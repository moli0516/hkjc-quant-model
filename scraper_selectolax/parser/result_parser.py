import re
import sys
from typing import Dict, List, Any, Optional
from selectolax.parser import HTMLParser

class ResultParser:
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def parse_race_tab(self) -> Optional[Dict[str, Any]]:
        try:
            element = self.tree.css_first("div.race_tab")
            if not element:
                print("Error occured, no race_tab element in the tree!")
                return None

            target_table = element.css_first("table")
            race_params = {
                "race_id": "",
                "basic_info": "",
                "track_condition": "",
                "track_info": "",
                "cumulative_finish_time": [],
                "sectional_finish_time": []
            }
            if not target_table:
                return race_params

            for row in target_table.css("tr"):
                cols = row.css("td")
                for idx, col in enumerate(cols):
                    text = col.text(strip=True)
                    if any(item in text for item in ["班", "關", "新"]):
                        race_params["basic_info"] = text
                    if "場地" in text and idx + 1 < len(cols):
                        race_params["track_condition"] = cols[idx + 1].text(strip=True)
                    if "賽道 :" in text and idx + 1 < len(cols):
                        race_params["track_info"] = cols[idx + 1].text(strip=True)
                    if "分段時間 :" in text:
                        siblings = cols[idx + 1:]
                        race_params["sectional_finish_time"] = [x.text(strip=True)[0:6] for x in siblings]
                    elif "時間 :" in text and "分段時間 :" not in text:
                        siblings = cols[idx + 1:]
                        race_params["cumulative_finish_time"] = [
                            re.sub(r"[()\[\]{}]", "", x.text(strip=True)) for x in siblings
                        ]
            return race_params
        except Exception as e:
            print("Error occured, no race_tab element in the tree!", e)
            return None

    def parse_results(self) -> Optional[List[Dict[str, Any]]]:
        try:
            element = self.tree.css_first('div[class*="performance"]')
            if element is None:
                element = self.tree.css_first('[class*="performance"]')

            if element is None:
                print("⚠️ Parser 警告：這個 tree 裡面真的完全沒有任何帶有 performance class 的標籤！")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            horses_params_list = []
            rows = target_table.css("tr")

            for row in rows[1:]:
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
                cols = row.css("td")

                for i, col in enumerate(cols):
                    val = col.text(strip=True)
                    
                    if i == 9 and val != "---":
                        first_div = col.css_first("div")
                        if first_div:
                            pos_divs = first_div.css("div")
                            position_list = [
                                int(x.text(strip=True)) 
                                for x in pos_divs 
                                if x.text(strip=True).isdigit()
                            ]
                            horse_params["race_position"] = position_list
                        else:
                            horse_params["race_position"] = val
                    elif i == 1:
                        continue
                    else:
                        try:
                            val = float(val)
                        except ValueError:
                            pass

                        if i < len(key_list):
                            horse_params[key_list[i]] = val

                horses_params_list.append(horse_params)
            return horses_params_list

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            line_number = exc_tb.tb_lineno if exc_tb else "Unknown"
            print(f"Error occured, no performance element in the tree! at line {line_number}. \n {e}")
            return None

    def parse_race_length(self) -> Optional[int]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("Error occured, no race_tab element in the tree!")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            first_row = target_table.css_first("tr")
            if not first_row:
                return None

            all_td = first_row.css("td")
            empty_cnt = 0
            for td in all_td:
                if not td.text(strip=True) and len(td.children) == 0:
                    empty_cnt += 1
            return len(all_td) - 2 - empty_cnt
        except Exception as e:
            print("Error occured, no race_tab element in the tree!", e)
            return None

    def parse_all_date(self) -> Optional[List[str]]:
        try:
            element = self.tree.css_first("div.raceMeeting_select")
            if not element:
                return None

            target_select = element.css_first("select")
            if not target_select:
                return None

            options = target_select.css("option")
            raw_dates = [x.text(strip=True) for x in options]
            return [f"{x[6:]}/{x[3:5]}/{x[:2]}" for x in raw_dates if len(x) >= 10]
        except Exception as e:
            print(f"Error occured, no date element in the tree!\n{e}")
            return None

    def parse_venue(self) -> Optional[str]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("Error occured, no venue element in the tree!")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            first_row = target_table.css_first("tr")
            if not first_row:
                return None

            first_td = first_row.css_first("td")
            return first_td.text(strip=True) if first_td else ""
        except Exception as e:
            print("Error occured, no venue element in the tree!", e)
            return None

    def is_oversea(self) -> bool:
        if (
            not self.tree
            or "overseas" in self.current_url.lower()
            or self.tree.css_first("div#race_top_banner_container") is not None
            or self.tree.css_first("div.top_races") is None
        ):
            return True
        return False

    def parse_single_race(self, i: int) -> Dict[str, Any]:
        race_info = self.parse_race_tab() or {}
        race_performance = self.parse_results()
        race_info["race_id"] = i
        race_info["horses"] = race_performance
        return race_info