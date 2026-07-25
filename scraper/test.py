from bs4 import BeautifulSoup
import requests
import sys

class Hook:
    def __init__(self):
        self.url = "https://racing.hkjc.com/zh-hk/local/information/localresults"
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8"
        }
    def get_soup(self, no, race_date):
        params = {
            "RaceDate": race_date,
            "RaceNo": no
        }
        try:
            response = requests.get(self.url, headers=self.headers, params=params, timeout=10)
            html_content = response.content.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")
            print("Get soup successfully!")
            return soup
        except Exception as e:
        # 印出真正的錯誤詳細原因，不要被自訂訊息誤導
            print(f"程式執行發生異常錯誤: {e}")
            
class get_data_from_soup:
    def __init__(self, soup):
        self.soup = soup
    def parse_table(self):
        all_tables = self.soup.find_all('table')
    
        if not all_tables:
            print("【警告】網頁中完全沒有任何 <table> 標籤！可能被反爬蟲阻擋或網址無效。")
        else:
            print(f"【成功】在網頁中找到了 {len(all_tables)} 個表格結構。")
            print("-" * 50)
    def parse_race_tab(self):
        element = soup.find("div", class_="race_tab")
        target_table = element.select_one('table')
        for row in target_table.find_all("tr"):
            for col in row.find_all("td"):
                print(col.get_text(strip=True), end=" ")
            print("")
            
    def parse_top_races(self):
        element = self.soup.find("div", class_="top_races")
        target_table = element.select_one('table')
        first_row = target_table.select_one("tr")
        all_td = first_row.find_all("td")
        empty_cnt = 0
        for td in all_td:
            print(td.get_text(strip=True))
            if len(td.contents) == 0:
                empty_cnt += 1
        return len(all_td) - 2 - empty_cnt
        
    def parse_all_date(self):
        element = self.soup.find("div", class_="raceMeeting_select")
        target_select = element.select_one("select")
        options = target_select.find_all("option")
        return [x for x in options.get_text(strip=True)]

if __name__ == "__main__":
    hook = Hook()
    soup = hook.get_soup("1", "2026/07/04")
    data_parser = get_data_from_soup(soup)
    length_of_race = data_parser.parse_top_races()
    for i in range(1, length_of_race+1):
        soup = hook.get_soup(str(i), "2026/07/04")
        data_parser = get_data_from_soup(soup)
        length_of_race = data_parser.parse_race_tab()
    