from typing import List, Dict, Any, Optional
from selectolax.parser import HTMLParser
from scraper.hook import Hook

class RatingParser:
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def parse_ratings(self) -> List[Dict[str, Any]]:
        ratings = []
        if not self.tree:
            return ratings

        target_tables = self.tree.css("table.report_body_small")
        print(f"📊 找到 {len(target_tables)} 個評分表格")

        for table in target_tables:
            rows = table.css("tr")
            for row in rows[1:]:
                cols = row.css("td")
                if len(cols) >= 4:
                    horse_name = cols[1].text(strip=True)
                    horse_id = cols[2].text(strip=True)
                    raw_rating = cols[3].text(strip=True)
                    
                    try:
                        rating = int(raw_rating)
                    except ValueError:
                        rating = None

                    ratings.append({
                        "horse_name": horse_name,
                        "horse_id": horse_id,
                        "rating": rating
                    })

        return ratings


def data_process() -> Optional[List[Dict[str, Any]]]:
    hook = Hook()
    
    result = hook.get_rating_tree()
    if not result:
        print("❌ 無法取得評分頁面 DOM 樹")
        return None

    tree, final_url = result
    rating_parser = RatingParser(tree=tree, current_url=final_url)
    ratings = rating_parser.parse_ratings()
    
    print(f"✅ 成功解析 {len(ratings)} 筆馬匹評分資料")
    return ratings