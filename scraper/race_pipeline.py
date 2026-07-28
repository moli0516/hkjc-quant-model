import asyncio
import sys
import traceback
from typing import List, Optional, Dict, Any, Tuple

from config.settings import settings
from scraper.hook import Hook
from scraper.parser.calander_parser import get_all_date_async
from scraper.parser.result_parser import ResultParser
from scraper.parser.sectional_parser import SectionalParser
from scraper.data_manager import DataManager


class RaceScrapingPipeline:
    """賽事爬蟲調度管道 (Pipeline Layer)"""

    def __init__(self, max_concurrent_days: int = 20):
        self.semaphore = asyncio.Semaphore(max_concurrent_days)
        self.races_db = DataManager(json_path=settings.raw_races_json_dir)
        self.sectional_db = DataManager(json_path=settings.raw_sectional_json_dir)

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """
        深層 Exception 紀錄器：自動擷取引發 Error 的最深層檔案、函式名稱與行號
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # 追蹤 Traceback 到最深層的發源點
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown File"
        func_name = last_tb.tb_frame.f_code.co_name if last_tb else "Unknown Func"
        line_no = last_tb.tb_lineno if last_tb else "Unknown Line"

        print(
            f"\n💥 [Pipeline 崩潰] {context_msg}\n"
            f"   ├─ 類型: {exc_type.__name__ if exc_type else type(error).__name__}\n"
            f"   ├─ 訊息: {error}\n"
            f"   └─ 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )
        # 印出精簡後的 Call Stack，方便追蹤最底層原因
        tb_summary = traceback.format_exception(exc_type, exc_value, exc_tb)
        print("   🔍 完整調用鏈追蹤 (Traceback):")
        for line in tb_summary[-3:]:  # 只印出最後 3 層最關鍵的堆疊
            print(f"      {line.strip()}")

    async def fetch_and_parse_race(
        self, hook: Hook, race_no: int, date_str: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        race_info: Optional[Dict[str, Any]] = None
        sectional_info: Optional[Dict[str, Any]] = None

        try:
            # 1. 並發請求基本賽果與分段時間頁面
            basic_task = hook.get_result_tree(race_no, date_str)
            sectional_task = hook.get_sectional_tree(
                race_no, f"{date_str[-2:]}/{date_str[5:7]}/{date_str[:4]}"
            )

            basic_result, sectional_result = await asyncio.gather(
                basic_task, sectional_task, return_exceptions=True
            )

            # 檢查並處理非同步任務中拋出的 Task 例外
            if isinstance(basic_result, Exception):
                self._log_exception(f"[{date_str} R{race_no}] 抓取基本賽果 Task 異常", basic_result)
            elif basic_result:
                basic_tree, basic_url = basic_result
                result_parser = ResultParser(basic_tree, basic_url)
                race_info = result_parser.parse_single_race(race_no)

            if isinstance(sectional_result, Exception):
                self._log_exception(f"[{date_str} R{race_no}] 抓取分段數據 Task 異常", sectional_result)
            elif sectional_result:
                sec_tree, sec_url = sectional_result
                sec_parser = SectionalParser(sec_tree, sec_url)
                sec_rows = sec_parser.parse_sectional_row()

                if sec_rows:
                    sectional_info = {
                        "race_no": race_no,
                        "sectional_data": sec_rows
                    }

            return race_info, sectional_info

        except Exception as e:
            self._log_exception(f"[{date_str} R{race_no}] fetch_and_parse_race 執行失敗", e)
            return None, None

    async def process_day(self, hook: Hook, date_str: str) -> None:
        async with self.semaphore:
            if self.races_db.check_file_exist(date_str) and self.sectional_db.check_file_exist(date_str, "race"):
                print(f"⏩ [跳過] 日期 {date_str} 本地賽果與分段檔案均已存在。")
                return

            try:
                init_result = await hook.get_result_tree("1", date_str)
                if not init_result:
                    print(f"⚠️ [Pipeline 警告] 無法取得 {date_str} 第一場數據 (可能是空頁面或網路請求失敗)")
                    return

                tree, url = init_result
                parser = ResultParser(tree, url)

                if parser.is_oversea():
                    print(f"🌍 [海外賽事] {date_str} 跳過處理。")
                    return

                total_races = parser.parse_race_length() or 0
                venue = (parser.parse_venue() or "")[:-1]

                print(f"🏇 正在爬取: {date_str} | 場地: {venue} | 共 {total_races} 場")

                if total_races == 0:
                    print(f"⚠️ [Pipeline 警告] {date_str} 判讀場次總數為 0，跳過該日處理。")
                    return

                race_tasks = [
                    self.fetch_and_parse_race(hook, i, date_str)
                    for i in range(1, total_races + 1)
                ]
                results = await asyncio.gather(*race_tasks, return_exceptions=True)

                races_data = []
                sectionals_data = []

                for idx, res in enumerate(results, 1):
                    if isinstance(res, Exception):
                        self._log_exception(f"[{date_str} R{idx}] 非同步子任務拋出未捕捉例外", res)
                    elif res:
                        r_info, s_info = res
                        if r_info is not None:
                            races_data.append(r_info)
                        if s_info is not None:
                            sectionals_data.append(s_info)

                race_payload = {
                    "date": date_str,
                    "venue": venue,
                    "races": races_data
                }

                sectional_payload = {
                    "date": date_str,
                    "venue": venue,
                    "sectionals": sectionals_data
                }

                await self.races_db.save_races_json(date_str, race_payload)
                await self.sectional_db.save_races_json(date_str, sectional_payload)

                print(f"✅ [成功] {date_str} 賽果與分段數據已分開存檔。")

            except Exception as e:
                self._log_exception(f"處理賽事日 [{date_str}] 時發生嚴重錯誤", e)

    async def run(self, years: List[int]) -> None:
        """Pipeline 進入點：執行批量日期下載"""
        try:
            target_dates = await get_all_date_async(*years)
            print(f"📅 共找到 {len(target_dates)} 個待處理賽事日期。")

            if not target_dates:
                print("❌ 沒有可執行的日期，程式結束。")
                return

            async with Hook() as hook:
                tasks = [self.process_day(hook, date_str) for date_str in target_dates]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception("全域 process_day 非同步任務發生錯誤", res)

        except Exception as e:
            self._log_exception("Pipeline run 進入點發生嚴重崩潰", e)