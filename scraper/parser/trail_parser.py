# scraper/parser/trail_parser.py
import re
import sys
from typing import Dict, List, Any, Optional
from selectolax.parser import HTMLParser


class TrailParser:
    """HKJC 試閘 (Barrier Trials) 數據解析器 (TrailParser)
    
    專門解析 HKJC 試閘結果 HTML (如從化/沙田試閘)，提取各組試閘之場地、距離、完成時間、
    分段時間及參賽馬匹之沿途走位與走勢評述。
    """

    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def _log_exception(self, method_name: str, error: Exception) -> None:
        """統一例外 Log 輸出格式"""
        _, _, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
        print(f"💥 [TrailParser 錯誤] {method_name} (Line {line_no}): {error}")

    def _clean_text(self, text: str) -> str:
        """標準化字串清理"""
        if not text:
            return ""
        return text.replace("\xa0", " ").strip()

    def parse_trials(self) -> List[Dict[str, Any]]:
        """解析 HTML 頁面內所有組別的試閘結果
        
        Returns:
            List[Dict[str, Any]]: 包含各組試閘基本資訊及參賽馬匹列表的字典串列
        """
        trials_list = []
        try:
            container = self.tree.css_first("div#divBtresult")
            if not container:
                # 備用選擇器
                container = self.tree.css_first("#divBtresult") or self.tree

            # ----------------------------------------------------------
            # 修正重點：
            # 原本 table:not(.bigborder) 會同時抓到「外層組別表格」與「巢狀的時間小表格」
            # 導致 info_tables 數量變成 12 個，result_tables 只有 6 個，配對時發生錯位。
            # 現在只選取真正有「第 X 組」subheader 的外層表格。
            # ----------------------------------------------------------
            all_non_bigborder = container.css("table:not(.bigborder)")
            info_tables = []
            for tbl in all_non_bigborder:
                subheader = tbl.css_first("td.subheader")
                if subheader:
                    text = self._clean_text(subheader.text())
                    # 必須包含「組」字，確保是真正的組別標題（例如：第 1 組 - 從化草地 - 1000米）
                    if "組" in text:
                        info_tables.append(tbl)

            result_tables = container.css("table.bigborder")

            # 若資訊與結果表格數量匹配，逐組解析
            for idx in range(min(len(info_tables), len(result_tables))):
                info_tbl = info_tables[idx]
                res_tbl = result_tables[idx]

                trial_info = self._parse_trial_info(info_tbl, group_index=idx + 1)
                horses = self._parse_trial_horses(res_tbl)

                trial_info["horses"] = horses
                trials_list.append(trial_info)

            return trials_list

        except Exception as e:
            self._log_exception("parse_trials", e)
            return trials_list

    def _parse_trial_info(self, info_table: HTMLParser, group_index: int) -> Dict[str, Any]:
        """解析單組試閘抬頭資訊 (組別、場地、距離、場地狀況、總時間、分段時間)"""
        info = {
            "group_no": group_index,
            "basic_info": "",
            "venue": "",
            "track_type": "",
            "distance": None,
            "track_condition": "",
            "finish_time": "",
            "sectional_times": []
        }
        try:
            # 1. 標頭 (例如: 第 1 組 - 從化草地 - 1000米)
            subheader = info_table.css_first("td.subheader")
            if subheader:
                basic_info = self._clean_text(subheader.text())
                info["basic_info"] = basic_info
                
                # 提取距離 (例如: 1000米 -> 1000)
                dist_match = re.search(r"(\d{3,4})\s*米", basic_info)
                if dist_match:
                    info["distance"] = int(dist_match.group(1))

                # 可選：從 basic_info 再拆出 venue / track_type
                # 例如：「第 1 組 - 從化草地 - 1000米」或「第 5 組 - 從化全天候跑道 - 1200米」
                # 這裡先保留簡單實作，如需更細可再擴充
                if "從化" in basic_info:
                    info["venue"] = "從化"
                elif "沙田" in basic_info:
                    info["venue"] = "沙田"
                elif "跑馬地" in basic_info:
                    info["venue"] = "跑馬地"

                if "全天候" in basic_info:
                    info["track_type"] = "全天候跑道"
                elif "草地" in basic_info:
                    info["track_type"] = "草地"

            # 2. 右側資訊欄 (場地狀況、時間、分段時間)
            # 注意：這些 font 位於巢狀的小表格內，但因為我們是從外層 info_table 出發，
            # selectolax 的 css() 會遞迴搜尋子孫節點，所以仍然可以正確取得。
            fonts = info_table.css("font")
            for font in fonts:
                text = self._clean_text(font.text())
                if "場地狀況:" in text:
                    info["track_condition"] = text.split(":", 1)[-1].strip()
                elif "時間:" in text and "分段時間" not in text:
                    info["finish_time"] = text.split(":", 1)[-1].strip()
                elif "分段時間:" in text:
                    sec_str = text.split(":", 1)[-1].strip()
                    # 分割多個分段時間 (例如 "14.2   22.5   22.6")
                    info["sectional_times"] = [s for s in re.split(r"\s+", sec_str) if s]

            return info

        except Exception as e:
            self._log_exception("_parse_trial_info", e)
            return info

    def _parse_trial_horses(self, result_table: HTMLParser) -> List[Dict[str, Any]]:
        """解析單組試閘之參賽馬匹詳細結果數據"""
        horses = []
        try:
            rows = result_table.css("tr")
            if len(rows) <= 1:
                return horses

            # 第一列為 Subheader (馬名、騎師、練馬師...)，從第二列開始解析
            for row in rows[1:]:
                cols = row.css("td")
                if not cols or len(cols) < 10:
                    continue

                # 提取馬名與 horse_id
                horse_name_td = cols[0]
                horse_name_raw = self._clean_text(horse_name_td.text())
                
                horse_id = ""
                a_tag = horse_name_td.css_first("a")
                if a_tag:
                    href = a_tag.attributes.get("href", "")
                    if "horseid=" in href:
                        # 例如: /zh-hk/local/information/horse?horseid=HK_2017_B234 -> HK_2017_B234
                        horse_id = href.split("horseid=")[-1].strip()

                # 檢查是否退出 (例如 沿途走位 td 為 "退出")
                running_pos_raw = self._clean_text(cols[6].text())
                is_withdrawn = "退出" in running_pos_raw or "退出" in horse_name_raw

                horse_param = {
                    "horse_name": horse_name_raw,
                    "horse_id": horse_id,
                    "jockey": self._clean_text(cols[1].text()),
                    "trainer": self._clean_text(cols[2].text()),
                    "draw": self._clean_text(cols[3].text()),
                    "gear": self._clean_text(cols[4].text()),
                    "margin": self._clean_text(cols[5].text()),
                    "running_position": running_pos_raw if not is_withdrawn else "退出",
                    "finish_time": self._clean_text(cols[7].text()),
                    "result_remark": self._clean_text(cols[8].text()),  # 例如: 及格 / 不及格
                    "performance_comment": self._clean_text(cols[9].text()),
                    "is_withdrawn": is_withdrawn
                }

                horses.append(horse_param)

            return horses

        except Exception as e:
            self._log_exception("_parse_trial_horses", e)
            return horses