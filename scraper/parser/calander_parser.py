import asyncio
from typing import List, Set
from selectolax.parser import HTMLParser
from scraper.hook import Hook


class CalendarParser:
    """專門解析賽事日曆頁面的 Parser"""

    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def parse_days(self) -> List[str]:
        """提取所有日曆表格中的日期文字"""
        if not self.tree:
            return []

        nodes = self.tree.css("td.calendar p.f_clear")
        return [node.text(strip=True) for node in nodes if node.text(strip=True)]


async def _date_process_async(hook: Hook, year: int, month: int) -> Set[str]:
    """私有非同步單月處理邏輯"""
    try:
        result = await hook.get_calendar_tree(year, month)
        if not result:
            return set()

        tree, url = result
        calendar_parser = CalendarParser(tree=tree, current_url=url)
        days = calendar_parser.parse_days()

        return {f"{year}/{str(month).zfill(2)}/{str(day).zfill(2)}" for day in days}

    except Exception as e:
        print(f"❌ [失敗] 處理 {year}年{month}月 發生異常: {e}")
        return set()


async def get_all_date_async(start_year: int, end_year: int) -> List[str]:
    """
    非同步版本：獲取範圍內所有賽事日期
    :param start_year: 起始年份 (例如 2020)
    :param end_year: 結束年份 (例如 2025)
    :return: 排序後的日期列表, 例: ['2020/01/01', '2020/01/05', ...]
    """
    all_racedays: Set[str] = set()
    
    # 🌟 修正原本 range(1, 12) 漏掉 12 月的 Bug -> 改為 range(1, 13)
    tasks_params = [(y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)]

    # 共享同一個 Hook Session 進行高速並發請求
    async with Hook() as hook:
        tasks = [_date_process_async(hook, y, m) for y, m in tasks_params]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, set):
                all_racedays.update(res)

    return sorted(list(all_racedays))


def get_all_date_multithread(start_year: int, end_year: int) -> List[str]:
    """
    同步相容層：保持函數名稱與舊版一致，內部直接啟動 asyncio Event Loop
    """
    return asyncio.run(get_all_date_async(start_year, end_year))