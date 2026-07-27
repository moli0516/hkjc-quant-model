import re
import sys
from typing import Dict, List, Any, Optional
from selectolax.parser import HTMLParser


class ResultParser:
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def _log_exception(self, method_name: str, error: Exception) -> None:
        """統一例外 Log 輸出格式"""
        _, _, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
        print(f"💥 [ResultParser 錯誤] {method_name} (Line {line_no}): {error}")

    def parse_race_tab(self) -> Optional[Dict[str, Any]]:
        try:
            element = self.tree.css_first("div.race_tab")
            if not element:
                print("⚠️ [ResultParser 警告] parse_race_tab: 找不到 div.race_tab 元素")
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
            self._log_exception("parse_race_tab", e)
            return None

    def parse_results(self) -> Optional[List[Dict[str, Any]]]:
        try:
            element = self.tree.css_first('div[class*="performance"]')
            if element is None:
                element = self.tree.css_first('[class*="performance"]')

            if element is None:
                print("⚠️ [ResultParser 警告] parse_results: 未找到包含 performance 的元素")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            horses_params_list = []
            rows = target_table.css("tr")

            for row in rows[1:]:
                horse_params = {
                    "placing": None,          # 名次 (int / None)
                    "horse_no": None,         # 馬號/布號 (int)
                    "horse_name": "",         # 馬名 (str)
                    "jockey": "",             # 騎師 (str)
                    "trainer": "",            # 練馬師 (str)
                    "actual_weight": None,    # 實際負磅 (int / float)
                    "body_weight": None,      # 排位體重 (int / float)
                    "draw": None,             # 檔位 (int / None)
                    "margin": "",             # 與頭馬距離 / 勝負距離 (str)
                    "finish_time": "",        # 完成時間 (str)
                    "odds": None,              # 獨贏賠率 (float / None)
                    "horse_id": ""
                }
                key_list = list(horse_params.keys())
                cols = row.css("td")

                for i, col in enumerate(cols):
                    val = col.text(strip=True)
                    if i == 2:
                        a = col.css_first('a')
                        if a:
                            href = a.attributes.get('href')
                            horse_params["horse_id"] = href.split("=")[1]
                    if i != 9:
                        try:
                            val = float(val)
                        except ValueError:
                            pass

                        if i < 9:
                            horse_params[key_list[i]] = val
                        elif i > 9 and i <= len(key_list):
                            horse_params[key_list[i-1]] = val

                horses_params_list.append(horse_params)
            return horses_params_list

        except Exception as e:
            self._log_exception("parse_results", e)
            return None

    def parse_race_length(self) -> Optional[int]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("⚠️ [ResultParser 警告] parse_race_length: 找不到 div.top_races 元素")
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
                # 檢查無文字且無子節點的空白排版格
                if not td.text(strip=True) and td.child is None:
                    empty_cnt += 1

            return len(all_td) - 2 - empty_cnt

        except Exception as e:
            self._log_exception("parse_race_length", e)
            return None

    def parse_venue(self) -> Optional[str]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("⚠️ [ResultParser 警告] parse_venue: 找不到 div.top_races 元素")
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
            self._log_exception("parse_venue", e)
            return None

    def is_oversea(self) -> bool:
        try:
            if (
                not self.tree
                or "overseas" in self.current_url.lower()
                or self.tree.css_first("div#race_top_banner_container") is not None
                or self.tree.css_first("div.top_races") is None
            ):
                return True
            return False
        except Exception as e:
            self._log_exception("is_oversea", e)
            return True

    def parse_single_race(self, i: int) -> Dict[str, Any]:
        try:
            race_info = self.parse_race_tab() or {}
            race_performance = self.parse_results()
            race_info["race_id"] = i
            race_info["horses"] = race_performance
            return race_info
        except Exception as e:
            self._log_exception("parse_single_race", e)
            return {"race_id": i, "horses": None}