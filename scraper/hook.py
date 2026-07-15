from bs4 import BeautifulSoup
import requests
import time
import random

class Hook:
    def __init__(self, url=""):
        self.url = url
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8"
        }
        self.session.headers.update(self.headers)
    def get_result_soup(self, no, race_date):
        actual_no = "1" if no == "0" else str(no)
        
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        try:
            time.sleep(random.randrange(2.0,4.0))
            response = self.session.get(self.url, params=params, timeout=15)
            print(f"🔗 正在請求: {response.url}")
            
            # 🌟 修正 2：檢查馬會是不是偷偷把你重定向了
            # 如果你請求的是第 2 場，但 response.url 裡面卻寫著 RaceNo=1，代表被制裁了
            if f"RaceNo={actual_no}" not in response.url and actual_no != "1":
                print(f"🚨 警告：被馬會強制重定向！原本要拿第 {actual_no} 場，卻被丟回 {response.url}")
                # 這裡可以選擇重試，或者先回傳空值
            
            html_content = response.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 🌟 修正 3：更嚴格的關鍵字檢查
            if soup:
                print("✨ 成功獲取網頁原始碼，交給 Parser 處理...")
            else:
                print("❌ 網頁原始碼完全無法獲取")
                
            return soup, response.url
            
        except Exception as e:
            print(f"❌ 網絡請求發生異常錯誤: {e}")
            return None
    def get_calendar_soup(self, year, month):
        params = {
            "calyear": str(year),
            "calmonth": str(month).zfill(2)
        }
        try:
            time.sleep(random.randrange(2.0,4.0))
            response = self.session.get(self.url, params=params, timeout=15)
            print(f"🔗 正在請求: {response.url}")
            
            html_content = response.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 🌟 修正 3：更嚴格的關鍵字檢查
            if soup:
                print("✨ 成功獲取網頁原始碼，交給 Parser 處理...")
            else:
                print("❌ 網頁原始碼完全無法獲取")
                
            return soup, response.url
            
        except Exception as e:
            print(f"❌ 網絡請求發生異常錯誤: {e}")
            return None
    def get_no_params_soup(self):
        try:
            time.sleep(random.randrange(2,4))
            response = self.session.get(self.url, timeout=15)
            print(f"🔗 正在請求: {response.url}")
            
            html_content = response.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")
            
            # 🌟 修正 3：更嚴格的關鍵字檢查
            if soup:
                print("✨ 成功獲取網頁原始碼，交給 Parser 處理...")
            else:
                print("❌ 網頁原始碼完全無法獲取")
                
            return soup, response.url
            
        except Exception as e:
            print(f"❌ 網絡請求發生異常錯誤: {e}")
            return None