import asyncio
import random
from typing import Optional, Tuple, Any, Dict
import aiohttp
from selectolax.parser import HTMLParser


class Hook:
    """非同步網路請求模組，基於 aiohttp 與 selectolax (內部封裝馬會 URL 與 API 邏輯)"""

    def __init__(self):
        # Base URL 與各種 Endpoint 私有變數
        self._domain = "https://racing.hkjc.com"
        self._result_endpoint = f"{self._domain}/zh-hk/local/information/localresults"
        self._sectional_endpoint = f"{self._domain}/zh-hk/local/information/displaysectionaltime"
        self._calendar_endpoint = f"{self._domain}/zh-hk/local/information/fixture"
        self._rating_endpoint = f"{self._domain}/racing/info/mcs/Chinese/Horses/clas/?&rf=http://racing.hkjc.com/zh-hk/local/information/latestonhorse?View=Horses/clas/&pageid=racing/local"
        self._horse_endpoint = f"{self._domain}/zh-hk/local/information/horse"
        self._trail_endpoint = f"{self._domain}/zh-hk/local/information/btresult"
        
        # 🌟 新增：單日晨操 JSON API Endpoint
        self._trackwork_json_endpoint = f"{self._domain}/racing/information/json/TrackworkOneDayRecords"

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

    async def _fetch(self, target_url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Tuple[HTMLParser, str]]:
        """私有核心非同步請求函式 (針對 HTML DOM 解析)"""
        if not self.session or self.session.closed:
            raise RuntimeError("🚨 Hook 必須在 `async with` 語境內使用！例如: `async with Hook() as hook:`")

        try:
            # 非同步隨機延遲，降低觸發 IP 封鎖的機率
            await asyncio.sleep(random.uniform(0.5, 1.8))

            async with self.session.get(target_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                
                final_url = str(response.url)
                print(f"🔗 正在請求 (HTML): {final_url}")

                # 非同步讀取 Raw Bytes 並直接餵給 selectolax
                content = await response.read()
                tree = HTMLParser(content)

                return tree, final_url

        except aiohttp.ClientError as e:
            print(f"❌ 網絡請求異常 [{target_url}]: {e}")
            return None
        except asyncio.TimeoutError:
            print(f"⏰ 請求超時 [{target_url}]")
            return None
        except Exception as e:
            print(f"💥 未知錯誤 [{target_url}]: {e}")
            return None

    async def _fetch_json(self, target_url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """🌟 新增：私有非同步 JSON 請求函式 (專門處理返回原生 JSON 的 API Endpoint)"""
        if not self.session or self.session.closed:
            raise RuntimeError("🚨 Hook 必須在 `async with` 語境內使用！例如: `async with Hook() as hook:`")

        try:
            # 非同步隨機延遲，降低觸發 IP 封鎖的機率
            await asyncio.sleep(random.uniform(0.3, 1.2))

            # 為 JSON API 設置專屬 Header 請求頭
            json_headers = {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }

            async with self.session.get(
                target_url, 
                params=params, 
                headers=json_headers, 
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                response.raise_for_status()
                
                final_url = str(response.url)
                print(f"🔗 正在請求 (JSON): {final_url}")

                # content_type=None 避開伺服器未設置 content-type: application/json 的 Header 檢查
                data = await response.json(content_type=None)
                return data

        except aiohttp.ClientError as e:
            print(f"❌ JSON API 網絡請求異常 [{target_url}]: {e}")
            return None
        except asyncio.TimeoutError:
            print(f"⏰ JSON API 請求超時 [{target_url}]")
            return None
        except Exception as e:
            print(f"💥 JSON 解析或未知錯誤 [{target_url}]: {e}")
            return None

    # -------------------------------------------------------------
    # 對外暴露的介面 (由 Hook 內部控制要打哪一個 Endpoint)
    # -------------------------------------------------------------

    async def get_result_tree(self, no: str | int, race_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [基本賽果] 頁面"""
        actual_no = "1" if str(no) == "0" else str(no)
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        return await self._fetch(self._result_endpoint, params=params)

    async def get_sectional_tree(self, no: str | int, race_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [分段時間與走位] 頁面"""
        actual_no = "1" if str(no) == "0" else str(no)
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        return await self._fetch(self._sectional_endpoint, params=params)

    async def get_calendar_tree(self, year: int, month: int) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [賽事日曆] 頁面"""
        params = {
            "calyear": str(year),
            "calmonth": str(month).zfill(2)
        }
        return await self._fetch(self._calendar_endpoint, params=params)
    
    async def get_horse_tree(self, horse_id: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [馬匹 Profile] 頁面"""
        params = {
            "horseid": horse_id
        }
        return await self._fetch(self._horse_endpoint, params=params)

    async def get_rating_tree(self) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取預設首頁"""
        return await self._fetch(self._rating_endpoint)

    async def get_trail_tree(self, trail_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取試閘結果頁面"""
        params = {
            "Date": trail_date,
        }
        return await self._fetch(self._trail_endpoint,params=params)

    async def get_trackwork_json(self, date_str: str, venue_code: str = "1C", page_num: int = 1) -> Optional[Dict[str, Any]]:
        """
        🌟 新增：非同步獲取 [單日晨操紀錄] JSON API 數據
        
        :param date_str: 格式為 'YYYYMMDD' 或 'YYYY-MM-DD' (例: '20260802' 或 '2026-08-02')
        :param venue_code: 場地代碼 (預設 '1C' 代表沙田晨操/草地試閘)
        :param page_num: 頁碼 (預設 1)
        :return: 回傳原始 Python 字典 (dict)，若失敗則回傳 None
        """
        # 清理日期格式: 將 '2026-08-02' 轉為 '20260802'
        clean_date = str(date_str).replace("-", "").replace("/", "").strip()
        
        # 拼接 URL 端點: 例 https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202608021C.aspx
        target_url = f"{self._trackwork_json_endpoint}/{clean_date}3C.aspx"
        params = {
            "PageNum": str(page_num)
        }
        
        return await self._fetch_json(target_url, params=params)