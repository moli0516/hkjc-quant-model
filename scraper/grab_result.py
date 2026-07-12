from bs4 import BeautifulSoup
import requests
import re
import time
import random
import sys

class Hook:
    def __init__(self):
        self.url = "https://racing.hkjc.com/zh-hk/local/information/localresults"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8"
        }
        self.session.headers.update(self.headers)
    def get_soup(self, no, race_date):
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
            
class get_data_from_soup:
    def __init__(self, soup, current_url=""):
        self.soup = soup
        self.current_url = current_url
    def parse_race_tab(self):
        try:
            element = self.soup.find("div", class_="race_tab")
            target_table = element.select_one('table')
            race_params = {
                "race_id": "",
                "basic_info": "",
                "track_condition": "",
                "track_info": "",
                "cumulative_finish_time": [],
                "sectional_finish_time": []
            }
            for row in target_table.find_all("tr"):
                for col in row.find_all("td"):
                    if any(item in col.get_text(strip=True) for item in ["班", "關", "新"]):
                        race_params["basic_info"] = col.get_text(strip=True)
                    if "場地" in col.get_text(strip=True):
                        race_params["track_condition"] = col.find_next_siblings("td")[0].get_text(strip=True)
                    if "賽道 :" in col.get_text(strip=True):
                        race_params["track_info"] = col.find_next_sibling("td").get_text(strip=True)
                    if "分段時間 :" in col.get_text(strip=True):
                        race_params["sectional_finish_time"] = [x.get_text(strip=True)[0:6] for x in col.find_next_siblings("td")]
                    elif "時間 :" in col.get_text(strip=True):
                        race_params["cumulative_finish_time"] = [re.sub(r"[()\[\]{}]", "", x.get_text(strip=True)) for x in col.find_next_siblings("td")]
            return race_params
        except:
            print("Error occured, no race_tab element in the soup!")
    
    def parse_results(self):
        try:
            element = self.soup.select_one('div[class*="performance"]')
            if element is None:
                element = self.soup.select_one('[class*="performance"]')
                
            # 💡 在這裡加一個安全閥
            if element is None:
                print("⚠️ Parser 警告：這個 soup 裡面真的完全沒有任何帶有 performance class 的標籤！")
                return None
                
            target_table = element.select_one('table')
            horses_params_list = []
            for row in target_table.find_all("tr")[1:]:
                horse_params = {
                    "placing": "",
                    "horse_id": "",
                    "horse_name": "",
                    "jockey": "",
                    "trainer": "",
                    "weight": "",
                    "rank_weight": "",
                    "draw": "",
                    "head_horse_dist": "",
                    "race_position": "",
                    "finished_time": "",
                    "odds": ""
                }
                key_list = list(horse_params.keys())
                for i, col in enumerate(row.find_all("td")):
                    val = col.get_text(strip=True)
                    if i == 9  and val !="---":
                        first_div = col.select_one('div')
                        position_list = [int(x.get_text(strip=True)) for x in first_div.find_all("div")]
                        horse_params["race_position"] = position_list
                    elif i == 1:
                        continue
                    else:
                        # 2. Try to convert to a float (real number)
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                        horse_params[key_list[i]] = val
                        
                horses_params_list.append(horse_params)
            return horses_params_list
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
             # tb_lineno 屬性即為錯誤行號
            line_number = exc_tb.tb_lineno
            print(f"Error occured, no performance element in the soup! at line {line_number}. \n {e}")
            
    def parse_race_length(self):
        try:
            element = self.soup.find("div", class_="top_races")
            target_table = element.select_one('table')
            first_row = target_table.select_one("tr")
            all_td = first_row.find_all("td")
            empty_cnt = 0
            for td in all_td:
                if len(td.contents) == 0:
                    empty_cnt += 1
            return len(all_td) - 2 - empty_cnt
        except:
            print("Error occured, no race_tab element in the soup!")
        
    def parse_all_date(self):
        try:
            element = self.soup.find("div", class_="raceMeeting_select")
            target_select = element.select_one("select")
            options = target_select.find_all("option")
            raw_dates = [x.get_text(strip=True) for x in options]
            return [f"{x[6:]}/{x[3:5]}/{x[:2]}" for x in raw_dates]
        except Exception as e:
            print("Error occured, no date element in the soup!\n",{e})
    
    def parse_venue(self):
        try:
            element = self.soup.find("div", class_="top_races")
            target_table = element.select_one('table')
            first_row = target_table.select_one("tr")
            first_td = first_row.select_one("td")
            return first_td.get_text(strip=True)
        except:
            print("Error occured, no venue element in the soup!")
    
    def is_oversea(self):
        if not self.soup or "overseas" in self.current_url.lower() or self.soup.find("div", id="race_top_banner_container") or not self.soup.find("div", class_="top_races"):
            return True
        return False
    
    def parse_single_race(self, i):
        race_info = self.parse_race_tab()
        race_performance = self.parse_results()
        race_info["race_id"] = i
        race_info["horses"] = race_performance
        return race_info

if __name__ == "__main__":
    hook = Hook()
    init_soup, init_url = hook.get_soup("1", "2026/07/08")
    init_parser = get_data_from_soup(init_soup, init_url)
    dates = init_parser.parse_all_date()[1:51]
    print(dates)
    '''
    print(dates)
    for date in dates:
        print(date)
        day_init_soup, day_init_url = hook.get_soup("1", date)
        day_init_parser = get_data_from_soup(day_init_soup, day_init_url)
        if day_init_parser.is_oversea():
            print("OVERSEA! SKIP")
            continue
        length_of_race = day_init_parser.parse_race_length()
        venue = day_init_parser.parse_venue()
        print(f"場地: {venue}, 日期: {date}")
        for i in range(1, length_of_race+1):
            print(str(i))
            race_soup, race_url = hook.get_soup(str(i), date)
            race_parser = get_data_from_soup(race_soup, race_url)
            race_info = race_parser.parse_race_tab()
            race_info["venue"] = venue
            race_performance = race_parser.parse_results()
            print(race_performance)
            #f = open(pathlib.Path(__file__).parent / "niga.txt", "a")
            #f.write(str(race_info))
            #f.writelines([x for x in race_performance)])
            #f.close()    
    '''