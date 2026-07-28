import asyncio
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from scraper.data_manager import DataManager
from scraper.hook import Hook
from scraper.parser.horse_parser import HorseProfileParser


class HorseScrapingPipeline:
    """馬匹詳細資料爬蟲調度管道 (Pipeline Layer)"""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.horse_db = DataManager(
            json_path=settings.raw_horses_json_dir
        )

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """深層 Exception 紀錄器：自動擷取引發 Error 的最深層檔案、函式名稱與行號"""
        exc_type, exc_value, exc_tb = sys.exc_info()

        # 追蹤 Traceback 到最深層的發源點
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

        print(
            f"\n💥 [Horse Pipeline 崩潰] {context_msg}\n"
            f"   ├─ 類型: {exc_type.__name__ if exc_type else type(error).__name__}\n"
            f"   ├─ 訊息: {error}\n"
            f"   └─ 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )
        tb_summary = traceback.format_exception(exc_type, exc_value, exc_tb)
        print("   🔍 完整調用鏈追蹤 (Traceback):")
        for line in tb_summary[-3:]:  # 只印出最後 3 層最關鍵的堆疊
            print(f"      {line.strip()}")

    async def fetch_and_parse_horse(
        self, hook: Hook, horse_code: str
    ) -> Optional[Dict[str, Any]]:
        """單一馬匹資料請求與解析"""
        try:
            result = await hook.get_horse_tree(horse_code)

            if isinstance(result, Exception):
                self._log_exception(
                    f"[{horse_code}] 抓取馬匹 Profile Task 異常", result
                )
                return None

            if not result:
                print(
                    f"⚠️ [Horse Pipeline 警告] 無法取得馬匹 [{horse_code}] HTML 頁面"
                )
                return None

            tree, url = result
            parser = HorseProfileParser(tree, url)
            profile_data = parser.parse_horse_profile()

            if profile_data:
                profile_data["horse_code"] = horse_code
                return profile_data

            return None

        except Exception as e:
            self._log_exception(
                f"[{horse_code}] fetch_and_parse_horse 執行失敗", e
            )
            return None

    async def process_horse(self, hook: Hook, horse_code: str) -> None:
        """單一馬匹處理流程（含 Semaphore 與存檔檢查）"""
        async with self.semaphore:
            # 防重複爬取機制
            if self.horse_db.check_file_exist(horse_code):
                print(
                    f"⏩ [跳過] 馬匹代號 {horse_code} 本地 Profile 檔案已存在。"
                )
                return

            try:
                print(f"🐴 正在爬取馬匹資料: {horse_code}")
                profile_data = await self.fetch_and_parse_horse(
                    hook, horse_code
                )

                if profile_data:
                    await self.horse_db.save_normal_json(
                        horse_code, profile_data
                    )
                    print(f"✅ [成功] 馬匹 {horse_code} 資料已存檔。")
                else:
                    print(
                        f"⚠️ [Horse Pipeline 警告] 馬匹 {horse_code} 未能成功提取 valid profile。"
                    )

            except Exception as e:
                self._log_exception(
                    f"處理馬匹代號 [{horse_code}] 時發生嚴重錯誤", e
                )

    async def run(self, horse_codes: List[str]) -> None:
        """Pipeline 進入點：執行批量馬匹資料爬取

        :param horse_codes: 馬匹烙號/代號列表 (例: ['E123', 'G045', 'H112'])
        """
        try:
            print(f"📋 共收到 {len(horse_codes)} 個待處理馬匹代號。")

            if not horse_codes:
                print("❌ 沒有可執行的馬匹代號，程式結束。")
                return

            async with Hook() as hook:
                tasks = [
                    self.process_horse(hook, code) for code in horse_codes
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception(
                            "全域 process_horse 非同步任務發生錯誤", res
                        )

        except Exception as e:
            self._log_exception("HorsePipeline run 進入點發生嚴重崩潰", e)


if __name__ == "__main__":
    # 測試運行範例
    test_horse_codes = ["HK_2025_L397"]
    pipeline = HorseScrapingPipeline(max_concurrent=5)
    asyncio.run(pipeline.run(test_horse_codes))