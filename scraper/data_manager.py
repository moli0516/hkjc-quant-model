from grab_result import Hook, get_data_from_soup
import pathlib
import json
import random

class Data_manager:
    def __init__(self, db_path = pathlib.Path(__file__).parent.parent / "database" / "hkjc_racing.db", json_path = pathlib.Path(__file__).parent.parent / "data" / "raw_json"):
        self.db_path = db_path
        self.json_path = json_path
    def save_json(self, date_str, params):
        
        file_path = self.json_path / f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:]}.json"
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    hook = Hook()
    data_manager = Data_manager()
    init_soup, init_url = hook.get_soup("1", "2026/07/08")
    init_parser = get_data_from_soup(init_soup, init_url)
    dates = init_parser.parse_all_date()[1:]
    for date in dates:
        try:
            day_init_soup, day_init_url = hook.get_soup("1", date)
            day_init_parser = get_data_from_soup(day_init_soup, day_init_url)
            if day_init_parser.is_oversea():
                print("OVERSEA! SKIP")
                continue
            length_of_race = day_init_parser.parse_race_length()
            venue = day_init_parser.parse_venue()
            races_params = {"venue": venue[:-1],
                            "date": date,
                            "races": []}
            print(f"場地: {venue}, 日期: {date}")
            for i in range(1, length_of_race+1):
                print(str(i))
                race_soup, race_url = hook.get_soup(str(i), date)
                race_parser = get_data_from_soup(race_soup, race_url)
                race_info = race_parser.parse_single_race(i)
                races_params["races"].append(race_info)
            print(races_params)
            data_manager.save_json(date,races_params)
        except:
            pass