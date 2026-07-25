import asyncio
import random
from typing import Optional, Tuple, Any, Dict
import aiohttp
from selectolax.parser import HTMLParser


class Hook:
    """非同步網路請求模組，基於 aiohttp 與 selectolax"""

    def __init__(self, url: str = ""):
        self.url = url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """進入 context manager 時建立 aiohttp session"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager 時自動關閉 session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _fetch(self, params: Optional[Dict[str, Any]] = None) -> Optional[Tuple[HTMLParser, str]]:
        """私有核心非同步請求函式"""
        if not self.session or self.session.closed:
            raise RuntimeError("🚨 Hook 必須在 `async with` 語境內使用！例如: `async with Hook(url) as hook:`")

        try:
            # 🌟 非同步隨機延遲，防止阻塞 Event Loop，同時降低觸發 IP 封鎖的機率
            await asyncio.sleep(random.uniform(0.5, 1.8))

            async with self.session.get(self.url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                
                final_url = str(response.url)
                print(f"🔗 正在請求: {final_url}")

                # 🌟 核心優化：非同步讀取 Raw Bytes 並直接喂給 selectolax
                content = await response.read()
                tree = HTMLParser(content)

                print("✨ 成功獲取 DOM 樹，交給 Selectolax Parser 處理...")
                return tree, final_url

        except aiohttp.ClientError as e:
            print(f"❌ 網絡請求發生異常錯誤: {e}")
            return None
        except asyncio.TimeoutError:
            print(f"⏰ 請求超時 ({self.url})")
            return None
        except Exception as e:
            print(f"💥 未知錯誤: {e}")
            return None

    async def get_result_tree(self, no: str, race_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取特定賽日與場次的賽果頁面"""
        actual_no = "1" if str(no) == "0" else str(no)
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        return await self._fetch(params)

    async def get_calendar_tree(self, year: int, month: int) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取賽事日曆頁面"""
        params = {
            "calyear": str(year),
            "calmonth": str(month).zfill(2)
        }
        return await self._fetch(params)

    async def get_no_params_tree(self) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取不帶參數的預設頁面"""
        return await self._fetch()
    
