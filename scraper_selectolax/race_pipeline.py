import asyncio
from typing import List, Optional, Dict, Any

from config.settings import settings
from hook import Hook
from parser.calander_parser import get_all_date_multithread
from parser.rating_parser import data_process
from parser.result_parser import ResultParser
from data_manager import DataManager


class RaceScrapingPipeline:
    """賽事爬蟲調度管道 (Pipeline Layer)"""

    def __init__(self, data_manager: DataManager, max_concurrent_days: int = 5):
        self.db = data_manager
        self.semaphore = asyncio.Semaphore(max_concurrent_days)
        self.base_url = "https://racing.hkjc.com/zh-hk/local/information/localresults"

    async def fetch_and_parse_race(
        self, hook: Hook, race_no: int, date_str: str
    ) -> Optional[Dict[str, Any]]:
        """處理單一場次"""
        result = await hook.get_result_tree(str(race_no), date_str)
        if not result:
            return None
        tree, url = result
        parser = ResultParser(tree, url)
        return parser.parse_single_race(race_no)

    async def process_day(self, hook: Hook, date_str: str) -> None:
        """處理單一賽事日（包含跳過邏輯、並發抓取該日所有場次）"""
        async with self.semaphore:
            # 1. 檢查 DataManager 是否已經有資料
            if self.db.is_race_downloaded(date_str):
                print(f"⏩ [跳過] 日期 {date_str} 本地已存在。")
                return

            try:
                # 2. 獲取第 1 場頁面做初始判斷
                init_result = await hook.get_result_tree("1", date_str)
                if not init_result:
                    print(f"⚠️ 無法取得 {date_str} 第一場數據")
                    return

                tree, url = init_result
                parser = ResultParser(tree, url)

                if parser.is_oversea():
                    print(f"🌍 [海外賽事] {date_str} 跳過處理。")
                    return

                total_races = parser.parse_race_length() or 0
                venue = (parser.parse_venue() or "")[:-1]

                print(f"🏇 正在爬取: {date_str} | 場地: {venue} | 共 {total_races} 場")

                # 3. 並發抓取該日所有的場次 (Race 1 ~ N)
                race_tasks = [
                    self.fetch_and_parse_race(hook, i, date_str)
                    for i in range(1, total_races + 1)
                ]
                races_data = await asyncio.gather(*race_tasks)

                # 4. 組裝完整 JSON payload
                payload = {
                    "date": date_str,
                    "venue": venue,
                    "total_races": total_races,
                    "races": [r for r in races_data if r is not None]
                }

                # 5. 交給 DataManager 進行儲存
                await self.db.save_race_data(date_str, payload)
                print(f"✅ [成功] {date_str} 已完整下載並存檔。")

            except Exception as e:
                print(f"💥 [Pipeline 異常] 處理 {date_str} 時發生錯誤: {e}")

    async def run(self, target_dates: List[str]) -> None:
        """Pipeline 進入點：執行批量日期下載"""
        async with Hook(self.base_url) as hook:
            tasks = [self.process_day(hook, date_str) for date_str in target_dates]
            await asyncio.gather(*tasks)