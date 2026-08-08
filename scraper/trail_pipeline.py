# scraper/trail_pipeline.py
import asyncio
import datetime
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from scraper.data_manager import DataManager
from scraper.hook import Hook
from scraper.parser.trail_parser import TrailParser

logger = logging.getLogger(__name__)


class TrailScrapingPipeline:
    """HKJC 試閘 (Barrier Trials) 數據爬蟲調度管道 (Pipeline Layer)
    
    職責：
    1. 產生指定年份區間內包含的所有日曆天 (YYYY-MM-DD)。
    2. 使用 Hook.get_trail_tree 非同步擷取網頁 HTML。
    3. 透過 TrailParser 解析各組試閘之場地、距離、完成時間、分段時間與馬匹數據。
    4. 將解析結果以 JSON 形式持久化儲存至 data/raw_json/trials/ 目錄。
    """

    def __init__(self, max_concurrent_days: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent_days)
        self.save_dir = settings.raw_trails_json_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_manager = DataManager(json_path=self.save_dir)

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """深層 Exception 紀錄器：自動擷取引發 Error 的最深層檔案、函式名稱與行號"""
        exc_type, exc_value, exc_tb = sys.exc_info()

        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown File"
        func_name = last_tb.tb_frame.f_code.co_name if last_tb else "Unknown Func"
        line_no = last_tb.tb_lineno if last_tb else "Unknown Line"

        print(
            f"\n💥 [Trail Pipeline 崩潰] {context_msg}\n"
            f"   ├─ 類型: {exc_type.__name__ if exc_type else type(error).__name__}\n"
            f"   ├─ 訊息: {error}\n"
            f"   └─ 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )
        tb_summary = traceback.format_exception(exc_type, exc_value, exc_tb)
        print("   🔍 完整調用鏈追蹤 (Traceback):")
        for line in tb_summary[-3:]:
            print(f"      {line.strip()}")

    async def fetch_and_parse_trail(
        self, hook: Hook, date_str: str
    ) -> Optional[Dict[str, Any]]:
        """非同步請求與解析單一日期的試閘結果"""
        try:
            # 轉換為 HKJC 試閘查詢格式 (YYYY/MM/DD 或 YYYY-MM-DD)
            formatted_date = date_str.replace("-", "/")
            result = await hook.get_trail_tree(formatted_date)

            if isinstance(result, Exception):
                self._log_exception(f"[{date_str}] 抓取試閘 HTML Task 異常", result)
                return None

            if not result:
                return None

            tree, url = result
            parser = TrailParser(tree, url)
            trials_data = parser.parse_trials()

            if trials_data:
                return {
                    "date": date_str,
                    "total_groups": len(trials_data),
                    "trials": trials_data,
                }

            return None

        except Exception as e:
            self._log_exception(f"[{date_str}] fetch_and_parse_trail 執行失敗", e)
            return None

    async def process_day(self, hook: Hook, date_str: str) -> None:
        """單日處理流程（含 Semaphore 控制與本地防重爬檢查）"""
        clean_file_key = f"trail_{date_str.replace('-', '').replace('/', '')}"

        async with self.semaphore:
            # 本地已存在則跳過
            if (self.save_dir / f"{clean_file_key}.json").is_file():
                print(f"⏩ [跳過] 日期 {date_str} 本地試閘 Raw JSON 已存在。")
                return

            try:
                trail_payload = await self.fetch_and_parse_trail(hook, date_str)

                if trail_payload and trail_payload.get("trials"):
                    await self.data_manager.save_normal_json(
                        clean_file_key, trail_payload
                    )
                    print(
                        f"✅ [成功] 日期 {date_str} 試閘數據已存檔 (共 {trail_payload['total_groups']} 組試閘)。"
                    )
                else:
                    print(f"ℹ️ [無數據] 日期 {date_str} 無試閘紀錄或網頁內容為空。")

            except Exception as e:
                self._log_exception(f"處理試閘日期 [{date_str}] 時發生嚴重錯誤", e)

    async def run(self, start_year: int, end_year: int) -> None:
        """Pipeline 進入點：輸入起始年份與結束年份，生成此之間的所有日期並迭代爬取
        
        :param start_year: 起始年份 (例如 2024)
        :param end_year: 結束年份 (例如 2026)
        """
        if start_year > end_year:
            print(f"❌ [錯誤] 無效年份區間: start_year ({start_year}) > end_year ({end_year})")
            return

        # 生成雙年份區間內包含的所有日曆天 (YYYY-MM-DD)
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)
        
        # 限制爬取日期不得晚於今日，避免無效請求
        today = datetime.date.today()
        if end_date > today:
            end_date = today

        total_days = (end_date - start_date).days + 1
        all_dates = [
            (start_date + datetime.timedelta(days=i)).strftime("%Y/%m/%d")
            for i in range(total_days)
        ]

        print(
            f"📅 開始執行試閘數據爬取，目標年份: {start_year} ~ {end_year}，共計 {len(all_dates)} 個日曆天。"
        )

        try:
            async with Hook() as hook:
                tasks = [self.process_day(hook, date_str) for date_str in all_dates]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception("全域 process_day 非同步任務發生錯誤", res)

            print("✨ 試閘數據批量爬取完成！")

        except Exception as e:
            self._log_exception("TrailScrapingPipeline run 進入點發生嚴重崩潰", e)


if __name__ == "__main__":
    pipeline = TrailScrapingPipeline(max_concurrent_days=10)
    asyncio.run(pipeline.run(2026, 2026))