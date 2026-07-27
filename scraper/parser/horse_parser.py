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

        try:
            tables = self.tree.css("table[class*='table_top_right']")
            if not tables:
                print(f"⚠️ [parse_horse_profile] 找不到對應表格: {self.current_url}")
                return horse_profile_params

            trs = [tr for table in tables for tr in table.css("tr")]
            keys = list(horse_profile_params.keys())

            for i, key in enumerate(keys):
                try:
                    # 防護機制: 檢查 tr 索引是否存在
                    if i >= len(trs):
                        break

                    tds = trs[i].css("td")

                    # 防護機制: 檢查 td 數量是否足夠 (至少要 index 3 也就是第 4 個 td)
                    if len(tds) > 2:
                        val = tds[2].text(strip=True)
                        horse_profile_params[key] = val if val else None
                except IndexError as ie:
                    continue
                except Exception as e:
                    self._log_exception(
                        f"parse_horse_profile [Key: {key}]", e
                    )

            return horse_profile_params

        except Exception as e:
            # 捕捉全局未預期的極端錯誤（如 tree 為 None 或類型錯誤）
            self._log_exception("parse_horse_profile (Global)", e)
            return horse_profile_params