import re
import sys
from typing import Dict, Optional
from selectolax.parser import HTMLParser


class HorseProfileParser:

    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def _log_exception(self, method_name: str, error: Exception) -> None:
        """統一例外 Log 輸出格式"""
        _, _, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
        print(f"💥 [ResultParser 錯誤] {method_name} (Line {line_no}): {error}")

    def parse_horse_profile(self) -> Optional[Dict[str, Optional[str]]]:
        horse_profile_params = {
            "origin_age": None,  # 出生地 / 馬齡 (例: 澳洲 / 5)
            "color_sex": None,  # 毛色 / 性別 (例: 栗 / 閹)
            "import_type": None,  # 進口類別 (例: 自購新馬)
            "season_stakes": None,  # 今季獎金*
            "total_stakes": None,  # 總獎金*
            "placing_records": None,  # 冠-亞-季-總出賽次數* (例: 1-7-4-27)
            "recent_10_races_count": None,  # 最近十個賽馬日出賽場數
            "current_location": None,  # 現在位置
            "location_arrival_date": None,  # 到達日期
            "import_date": None,  # 進口日期
            "trainer": None,  # 練馬師
            "owner": None,  # 馬主
            "current_rating": None,  # 現時評分
            "season_start_rating": None,  # 季初評分
            "sire": None,  # 父系
            "dam": None,  # 母系
            "damsire": None,  # 外祖父
        }

        # 精確對應清單 (確保不會誤觸「同父系馬」)
        field_keyword_map = {
            "出生地": "origin_age",
            "毛色": "color_sex",
            "進口類別": "import_type",
            "今季獎金": "season_stakes",
            "總獎金": "total_stakes",
            "冠-亞-季": "placing_records",
            "最近十個賽馬日": "recent_10_races_count",
            "練馬師": "trainer",
            "馬主": "owner",
            "現時評分": "current_rating",
            "季初評分": "season_start_rating",
            "外祖父": "damsire",  # 優先度高於「父系」
            "父系": "sire",
            "母系": "dam",
            "進口日期": "import_date",
        }

        try:
            tables = self.tree.css("table[class*='table_top_right']")
            if not tables:
                print(
                    f"⚠️ [parse_horse_profile] 找不到對應表格: {self.current_url}"
                )
                return horse_profile_params

            trs = [tr for table in tables for tr in table.css("tr")]

            for tr in trs:
                try:
                    tds = tr.css("td")
                    if not tds:
                        continue

                    # 雙欄佈局：每 3 個 td 為一組
                    for idx in range(0, len(tds), 3):
                        if idx + 2 >= len(tds):
                            break

                        label_text = tds[idx].text(strip=True)
                        val_text = tds[idx + 2].text(strip=True)

                        if not label_text:
                            continue

                        # 🛑 防護機制：完全跳過「同父系馬」選單欄位
                        if "同父系" in label_text:
                            continue

                        # 特殊處理：「現在位置 / 到達日期」
                        if "現在位置" in label_text and val_text:
                            match = re.search(
                                r"([^\s\(]+)(?:\s*\((.*?)\))?", val_text
                            )
                            if match:
                                horse_profile_params["current_location"] = (
                                    match.group(1)
                                )
                                if match.group(2):
                                    horse_profile_params[
                                        "location_arrival_date"
                                    ] = match.group(2)
                            else:
                                horse_profile_params["current_location"] = (
                                    val_text
                                )
                            continue

                        if not val_text:
                            continue

                        # 關鍵字匹配
                        for keyword, target_key in field_keyword_map.items():
                            if keyword in label_text:
                                horse_profile_params[target_key] = val_text
                                break

                except Exception as e:
                    self._log_exception("parse_horse_profile [Row Error]", e)

            return horse_profile_params

        except Exception as e:
            self._log_exception("parse_horse_profile (Global)", e)
            return horse_profile_params