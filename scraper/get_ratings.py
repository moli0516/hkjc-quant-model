from hook import Hook
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class parse_rating_soup:
    def __init__(self, soup, current_url=""):
        self.soup = soup
        self.current_url = current_url
    def parse_ratings(self):
        ratings = []
        target_tables = self.soup.find_all("table", class_="report_body_small")
        print(len(target_tables))
        for table in target_tables:
            rows = table.find_all("tr")
            for row in rows[1:]:
                horse_rating = {}
                col = row.find_all("td")
                horse_rating["horse_id"] = col[2].get_text(strip=True)
                horse_rating["rating"] = int(col[3].get_text(strip=True))
                ratings.append(horse_rating)
        return ratings

def data_process():
    hook = Hook("https://racing.hkjc.com/racing/info/mcs/Chinese/Horses/clas/?&rf=http://racing.hkjc.com/zh-hk/local/information/latestonhorse?View=Horses/clas/&pageid=racing/local")
    soup, url = hook.get_no_params_soup()
    rating_parser = parse_rating_soup(soup=soup,current_url=url)
    ratings = rating_parser.parse_ratings()
    return ratings


def get_all_date_multithread():
    hook = Hook("https://racing.hkjc.com/en-us/local/information/fixture")
    all_racedays = set()
    tasks = [(y, m) for y in range(2020,2026) for m in range(1, 13)]
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_date = {executor.submit(date_process, y, m): (y, m) for y, m in tasks}
        for future in as_completed(future_to_date):
            y, m = future_to_date[future]
            try:
                month_result = future.result()
                all_racedays.update(month_result)
            except Exception as e:
                print(f"[失敗] 處理 {y}年{m}月 的線程發生異常: {e}")

    return sorted(list(all_racedays))

if __name__ == "__main__":
    start_time = time.time()
    
    rating_list = data_process()
    
    end_time = time.time()
    
    print("\n" + "="*30)
    print(f"總共花費時間: {end_time - start_time:.2f} 秒")
    print(f"總共成功獲取: {len(rating_list)} 個賽馬Rating。")
    if rating_list:
        print("前 5 個Rating: ", rating_list[:5])
        print("最後 5 個Rating: ", rating_list[-5:])
    else:
        print("未抓取到任何Rating，請檢查是否被防火牆阻擋 (403 錯誤)。")