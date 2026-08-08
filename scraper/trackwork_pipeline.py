import asyncio
import datetime
import logging
import sys
from typing import Any, Dict, List, Optional

from config.settings import settings
from scraper.data_manager import DataManager
from scraper.hook import Hook

logger = logging.getLogger(__name__)


class TrackworkScrapingPipeline:
    """香港賽馬會晨操 (Trackwork One Day Records) 非同步爬蟲調度管道
    
    規格特性：
    - 端點格式: TrackworkOneDayRecords/{YYYYMMDD}{VENUE}.aspx?PageNum={page}
    - 動態分頁機制: 讀取回應 JSON 中的 "next" 指針 (當 "next": 0 代表到達最後一頁)。
    - 數據預設儲存於: data/raw_json/trackwork/ 目錄。
    """

    def __init__(self, max_concurrent_days: int = 5, max_page_limit: int = 999):
        self.semaphore = asyncio.Semaphore(max_concurrent_days)
        self.max_page_limit = max_page_limit

        # 建立專門儲存 trackwork raw json 的 DataManager
        self.save_dir = settings.raw_json_dir / "trackwork"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_manager = DataManager(json_path=self.save_dir)

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """深層 Exception 紀錄器：精準定位異常發生之檔案與行號"""
        exc_type, exc_value, exc_tb = sys.exc_info()
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        file_name = (
            last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown File"
        )
        func_name = (
            last_tb.tb_frame.f_code.co_name if last_tb else "Unknown Func"
        )
        line_no = last_tb.tb_lineno if last_tb else "Unknown Line"

        logger.error(
            f"💥 [Trackwork Pipeline 錯誤] {context_msg} | "
            f"型態: {exc_type.__name__ if exc_type else type(error).__name__} | "
            f"訊息: {error} | 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )

    async def fetch_single_day_trackwork(
        self, hook: Hook, date_str: str, 
    ) -> List[Dict[str, Any]]:
        """針對單一日期與特定場地代碼，利用 "next" 指針精準執行分頁遞歸抓取
        
        中斷條件：
        1. 收到 None 或 HTTP 網路請求失敗。
        2. API 回應中的 'Records' 為空陣列。
        3. API 回應中的 'next' 鍵值為 0 (已到達最終頁)。
        """
        all_day_records: List[Dict[str, Any]] = []
        clean_date = date_str.replace("-", "").replace("/", "").strip()

        current_page = 1

        while current_page > 0 and current_page <= self.max_page_limit:
            try:
                # 呼叫 Hook 中新增的 get_trackwork_json API 介面
                json_response = await hook.get_trackwork_json(
                    date_str=clean_date,
                    page_num=current_page,
                )

                # 🛑 中斷條件 1：請求失敗或網絡異常
                if json_response is None or not isinstance(json_response, dict):
                    logger.debug(
                        f"ℹ️ [{clean_date}] 第 {current_page} 頁請求失敗或回應無效，結束該場地循環。"
                    )
                    break

                records = json_response.get("Records", [])

                # 🛑 中斷條件 2：Records 陣列為空
                if not records or len(records) == 0:
                    logger.debug(
                        f"ℹ️ [{clean_date}] 第 {current_page} 頁 Records 為空，結束該場地循環。"
                    )
                    break

                # 注入元數據 (Metadata) 方便後續 Data Cleaning / DTO 轉換
                for item in records:
                    item["_crawled_date"] = clean_date
                    item["_page_num"] = current_page

                all_day_records.extend(records)

                # 🛑 中斷條件 3：讀取 JSON 的 "next" 指針 (next: 0 代表已無下一頁)
                next_page = json_response.get("next", 0)
                if next_page == 0 or next_page <= current_page:
                    logger.debug(
                        f"🏁 [{clean_date}] 第 {current_page} 頁指示 next={next_page}，到達最終頁。"
                    )
                    break

                # 更新至下一頁指標
                current_page = next_page

            except Exception as e:
                self._log_exception(
                    f"抓取 [{clean_date} Page {current_page}] 時發生例外",
                    e,
                )
                break

        return all_day_records

    async def process_day(self, hook: Hook, date_str: str) -> None:
        """單日處理管線 (包含多場地併發與本地防重複抓取機制)"""
        clean_date = date_str.replace("-", "").replace("/", "").strip()
        file_key = f"trackwork_{clean_date}"

        async with self.semaphore:
            # 防重複爬取機制 (本地已存在則跳過)
            if (self.save_dir / f"{file_key}.json").is_file():
                logger.info(f"⏩ [跳過] 日期 {clean_date} 本地晨操 Raw JSON 已存在。")
                return

            try:
                logger.info(f"🐎 正在爬取晨操數據: {clean_date}...")
                day_records: List[Dict[str, Any]] = []

                page_records = await self.fetch_single_day_trackwork(
                    hook, clean_date
                )
                if page_records:
                    day_records.extend(page_records)

                if day_records:
                    # 使用 DataManager 持久化儲存 JSON
                    await self.data_manager.save_normal_json(
                        file_key,
                        {
                            "date": clean_date,
                            "total_records": len(day_records),
                            "records": day_records,
                        },
                    )
                    logger.info(
                        f"✅ [成功] 日期 {clean_date} 晨操數據已存檔 (共 {len(day_records)} 條記錄)。"
                    )
                else:
                    logger.warning(
                        f"⚠️ [Pipeline 警告] 日期 {clean_date} 未抓取到任何晨操數據 (可能無試閘/晨操或官網未留存)。"
                    )

            except Exception as e:
                self._log_exception(
                    f"處理晨操日期 [{clean_date}] 時發生嚴重錯誤", e
                )

    async def run(self, start_year: int, end_year: int) -> None:
        """Pipeline 入口：輸入兩個年份，生成期間包含的所有日曆天進行迭代與晨操數據爬取
        
        :param start_year: 起始年份 (例如 2024)
        :param end_year: 結束年份 (例如 2026)
        """
        if start_year > end_year:
            logger.error(
                f"❌ 無效的年份區間: start_year ({start_year}) 不能大於 end_year ({end_year})。"
            )
            return

        # 1. 生成雙年份區間內包含的所有日曆天 (YYYYMMDD)
        start_date = datetime.date(start_year, 1, 1)
        end_date = datetime.date(end_year, 12, 31)

        date_delta = (end_date - start_date).days + 1
        all_dates = [
            (start_date + datetime.timedelta(days=i)).strftime("%Y%m%d")
            for i in range(date_delta)
        ]

        logger.info(
            f"📅 開始執行晨操數據爬取，年份範圍: {start_year} ~ {end_year}，共計 {len(all_dates)} 個日曆天..."
        )

        try:
            async with Hook() as hook:
                tasks = [self.process_day(hook, d) for d in all_dates]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception("全域 process_day 非同步任務發生錯誤", res)

            logger.info("✨ 跨年份晨操數據批量爬取完成！")

        except Exception as e:
            self._log_exception("TrackworkScrapingPipeline run 進入點發生嚴重崩潰", e)
            
if __name__ == "__main__":
    pipeline = TrackworkScrapingPipeline()
    asyncio.run(pipeline.run(2026,2026))