from grab_result import parse_result_soup
from get_racedates import get_all_date_multithread
import pathlib
import json
from hook import Hook
from concurrent.futures import ThreadPoolExecutor, as_completed

class Data_manager:
    def __init__(self, db_path = pathlib.Path(__file__).parent.parent / "database" / "hkjc_racing.db", json_path = pathlib.Path(__file__).parent.parent / "data" / "raw_json"):
        self.db_path = db_path
        self.json_path = json_path
    def save_json(self, date_str, params):
        
        file_path = self.json_path / f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:]}.json"
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)

def process_single_day(date, data_manager, hook):
    try:
        day_init_soup, day_init_url = hook.get_result_soup("1", date)
        day_init_parser = parse_result_soup(day_init_soup, day_init_url)
        if day_init_parser.is_oversea():
            print("OVERSEA! SKIP")
            return
        length_of_race = day_init_parser.parse_race_length()
        venue = day_init_parser.parse_venue()
        races_params = {"venue": venue[:-1],
                            "date": date,
                            "races": []}
        print(f"場地: {venue}, 日期: {date}")
        for i in range(1, length_of_race+1):
            print(str(i))
            race_soup, race_url = hook.get_result_soup(str(i), date)
            race_parser = parse_result_soup(race_soup, race_url)
            race_info = race_parser.parse_single_race(i)
            races_params["races"].append(race_info)
        data_manager.save_json(date,races_params)
        print("Saved to json")
    except:
        print("Error occured")
            
if __name__ == "__main__":
    dates = get_all_date_multithread()
    data_manager = Data_manager()
    hook = Hook("https://racing.hkjc.com/zh-hk/local/information/localresults")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_date = {executor.submit(process_single_day, date, data_manager, hook):date for date in dates}
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            try:
                result_message = future.result()
                # 這裡可以追蹤每個線程回傳的狀態
            except Exception as e:
                print(f"[系統致命錯誤] 日期 {date} 的執行緒完全崩潰: {e}")
    
        