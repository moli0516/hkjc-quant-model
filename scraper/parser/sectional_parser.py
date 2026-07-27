from typing import Dict, List, Any, Optional
from selectolax.parser import HTMLParser

class SectionalParser():
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def parse_sectional_row(self) -> Optional[List[Dict[str, Any]]]:
        race_table = self.tree.css_first('div[class*="race_table"]')
        if race_table is None:
            race_table = self.tree.css_first('[class*="race_table"]')
        if race_table is None:
            print("⚠️ Parser 警告：這個 tree 裡面真的完全沒有任何帶有 race_table 的標籤！")
            return None
        t_body = race_table.css_first('tbody')
        if t_body is None:
            print("⚠️ Parser 警告：這個 tree 裡面真的完全沒有任何帶有 tbody 的標籤！")
            return None
        trs = t_body.css('tr')
        sectional_params_list = []
        for tr in trs:
            sectional_params = {
                "horse_name": "",
                "horse_id": "", 
                "sectional_details": []
            }
            tds = tr.css("td")
            for i, td in enumerate(tds[:-1]):
                if i == 2:
                    sectional_params["horse_name"] = td.text(strip=True)
                    a = td.css_first('a')
                    if a:
                        href = a.attributes.get('href')
                        sectional_params["horse_id"] = href.split("=")[1]
                elif i > 2:
                    if td.css_first("img") is None:
                        sectional_detailed_params = {
                                "section_no": i - 2,
                                "position": td.css_first("span").text(strip=True),
                                "margin": td.css_first("i").text(strip=True),
                                "sectional_time": td.css("p")[1].text(strip=True),
                                "split_times": [x.text(strip=True) for x in td.css("p")[1].css("span[class*='color_blue2'] span")]
                        }
                        sectional_params["sectional_details"].append(sectional_detailed_params)
            sectional_params_list.append(sectional_params)
        return sectional_params_list
                    