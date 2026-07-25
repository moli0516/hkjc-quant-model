from hook import Hook
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from selectolax.parser import HTMLParser
from typing import List
class CalendarParser:
    """專門解析賽事日曆頁面的 Parser"""
    
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def parse_days(self) -> List[str]:
        """提取所有日曆表格中的日期文字"""
        if not self.tree:
            return []
            
        nodes = self.tree.css("td.calendar p")
        return [node.text(strip=True) for node in nodes if node.text(strip=True)]

def date_process(year, month):
    hook = Hook("https://racing.hkjc.com/en-us/local/information/fixture")
    tree, url = hook.get_calendar_tree(year, month)
    calendar_parser = CalendarParser(tree = tree,current_url=url)
    days = calendar_parser.parse_days()
    return set([f'{year}/{str(month).zfill(2)}/{str(day).zfill(2)}' for day in days])

def get_all_date_multithread():
    all_racedays = set()
    tasks = [(y, m) for y in range(2020,2027) for m in range(1, 8)]
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_date = {executor.submit(date_process, y, m): (y, m) for y, m in tasks}
        for future in as_completed(future_to_date):
            y, m = future_to_date[future]
            try:
                month_result = future.result()
                all_racedays.update(month_result)
            except Exception as e:
                print(f"[失敗] 處理 {y}年{m}月 的線程發生異常: {e}")

    return sorted(list(all_racedays))
