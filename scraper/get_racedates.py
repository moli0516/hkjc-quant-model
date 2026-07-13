from hook import Hook
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class parse_calendar_soup:
    def __init__(self, soup, current_url=""):
        self.soup = soup
        self.current_url = current_url
    def parse_day(self):
        days = []
        target_days = self.soup.find_all("td", class_="calendar")
        for day_td in target_days:
            days.append(day_td.find("p").get_text(strip=True))
        return days

def date_process(year, month):
    hook = Hook("https://racing.hkjc.com/en-us/local/information/fixture")
    soup, url = hook.get_calendar_soup(year, month)
    calendar_parser = parse_calendar_soup(soup=soup,current_url=url)
    days = calendar_parser.parse_day()
    return set([f'{year}/{str(month).zfill(2)}/{str(day).zfill(2)}' for day in days])


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
    
    hkjc_racedays_list = get_all_date_multithread()
    
    end_time = time.time()
    
    print("\n" + "="*30)
    print(f"總共花費時間: {end_time - start_time:.2f} 秒")
    print(f"總共成功獲取: {len(hkjc_racedays_list)} 個賽馬日。")
    if hkjc_racedays_list:
        print("前 5 個日期: ", hkjc_racedays_list[:5])
        print("最後 5 個日期: ", hkjc_racedays_list[-5:])
    else:
        print("未抓取到任何日期，請檢查是否被防火牆阻擋 (403 錯誤)。")