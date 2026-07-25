from hook import Hook
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class parse_rating_soup:
    def __init__(self, soup, current_url=""):
        self.soup = soup
        self.current_url = current_url
    def parse_ratings(self):
        ratings = []
        target_tables = self.soup.find_all("table", class_="report_body_small")
        print(len(target_tables))
        for table in target_tables:
            rows = table.find_all("tr")
            for row in rows[1:]:
                horse_rating = {}
                col = row.find_all("td")
                horse_rating["horse_name"] = col[1].get_text(strip=True)
                horse_rating["horse_id"] = col[2].get_text(strip=True)
                horse_rating["rating"] = int(col[3].get_text(strip=True))
                ratings.append(horse_rating)
        return ratings

def data_process():
    hook = Hook("https://racing.hkjc.com/racing/info/mcs/Chinese/Horses/clas/?&rf=http://racing.hkjc.com/zh-hk/local/information/latestonhorse?View=Horses/clas/&pageid=racing/local")
    soup, url = hook.get_no_params_soup()
    rating_parser = parse_rating_soup(soup=soup,current_url=url)
    ratings = rating_parser.parse_ratings()
    return ratings
