import argparse
import sys
from pathlib import Path

# 假設專案模組引用
try:
    #from cleaners.horse_cleaner import HorseCleaner
    from cleaners.cleaner_pipeline import CleaningPipeline
    from config.settings import settings
    from database.db_manager import DBManager
    from scraper.horse_pipeline import HorseScrapingPipeline
    from scraper.race_pipeline import RaceScrapingPipeline
except ImportError as e:
    print(f"❌ 模組匯入失敗，請確認執行路徑與專案目錄結構: {e}")
    sys.exit(1)


class HKJCCLI:

    def __init__(self):
        self.db = DBManager()

    def check_db_has_races(self) -> bool:
        """檢查資料庫是否存在賽果數據 (race_results)"""
        has_data = self.db.has_race_results()
        if not has_data:
            print(
                "\n⚠️ [前置檢查失敗] 資料庫中找不到任何賽果數據 (race_results)！"
            )
            print("👉 請先執行：1. 賽果/分段爬蟲 ➔ 2. 賽果/分段數據清洗\n")
        return has_data

    # ---------------- 核心功能調用 ----------------
    def run_race_scraper(self, start_date=None, end_date=None):
        print("🚀 [Step 1] 開始執行：賽果與分段時間爬蟲...")
        scraper = RaceScrapingPipeline()
        scraper.run(start_date=start_date, end_date=end_date)
        print("✅ 賽果與分段時間爬蟲完成！\n")

    def run_race_cleaner(self):
        print("🧹 [Step 2] 開始執行：賽果與分段時間數據清洗...")
        cleaner = CleaningPipeline()
        cleaner.run()
        print("✅ 賽果與分段時間數據清洗完成，已寫入資料庫！\n")

    def run_horse_scraper(self):
        print("🐎 [Step 3] 檢查資料庫狀態以進行馬匹資料爬蟲...")
        if not self.check_db_has_races():
            return

        print("🚀 開始執行：馬匹資料爬蟲...")
        # 從 DB 讀取歷史所有出現過且尚未抓取/需要更新的 horse_id
        horse_ids = self.db.get_pending_horse_ids()
        print(f"📊 找到 {len(horse_ids)} 匹需要更新的馬匹資料。")

        if horse_ids:
            scraper = HorseScrapingPipeline()
            scraper.run(horse_ids)
            print("✅ 馬匹資料爬蟲完成！\n")
        else:
            print("ℹ️ 沒有需要爬取的馬匹 ID。\n")

    #def run_horse_cleaner(self):
    #    print("🧹 [Step 4] 開始執行：馬匹資料數據清洗...")
    #    cleaner = HorseCleaner()
    #    cleaner.run()
    #    print("✅ 馬匹資料數據清洗完成，已更新至資料庫！\n")

    # ---------------- 互動式選單 ----------------
    def interactive_menu(self):
        while True:
            print("=" * 45)
            print("🏇  HKJC 賽馬數據工程 CLI 工具")
            print("=" * 45)
            print("1. 執行賽果和分段時間爬蟲")
            print("2. 進行賽果和分段時間數據清洗")
            print("3. 執行馬匹資料爬蟲 (需要先有賽果資料庫)")
            print("4. 進行馬匹資料數據清洗")
            print("5. ⚡ 執行一鍵全套 Pipeline (1 ➔ 2 ➔ 3 ➔ 4)")
            print("0. 退出系統")
            print("=" * 45)

            choice = input("請選擇要執行的功能 (0-5): ").strip()

            if choice == "1":
                self.run_race_scraper()
            elif choice == "2":
                self.run_race_cleaner()
            elif choice == "3":
                self.run_horse_scraper()
            #elif choice == "4":
                #self.run_horse_cleaner()
            elif choice == "5":
                print("\n🔄 開始一鍵執行全套 Pipeline...")
                self.run_race_scraper()
                self.run_race_cleaner()
                if self.check_db_has_races():
                    self.run_horse_scraper()
                    self.run_horse_cleaner()
            elif choice == "0":
                print("👋 已退出 CLI 工具。")
                break
            else:
                print("❌ 無效選擇，請重新輸入！\n")


def main():
    parser = argparse.ArgumentParser(
        description="HKJC 賽馬量化數據爬蟲與清洗管道 (CLI Tool)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--scrape-races",
        action="store_true",
        help="執行賽果與分段時間爬蟲",
    )
    parser.add_argument(
        "--clean-races", action="store_true", help="執行賽果與分段數據清洗"
    )
    parser.add_argument(
        "--scrape-horses",
        action="store_true",
        help="執行馬匹資料爬蟲 (需先有賽果 DB)",
    )
    parser.add_argument(
        "--clean-horses", action="store_true", help="執行馬匹資料數據清洗"
    )
    parser.add_argument(
        "--all", action="store_true", help="依序執行全部步驟 (1 ➔ 2 ➔ 3 ➔ 4)"
    )

    parser.add_argument(
        "--start-date", type=str, help="賽事爬蟲起始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="賽事爬蟲結束日期 (YYYY-MM-DD)"
    )

    args = parser.parse_args()
    cli = HKJCCLI()

    # 如果沒有傳入任何參數，進入互動選單模式
    if not any(
        [
            args.scrape_races,
            args.clean_races,
            args.scrape_horses,
            args.clean_horses,
            args.all,
        ]
    ):
        cli.interactive_menu()
        return

    # 命令列參數直接觸發模式
    if args.all:
        cli.run_race_scraper(start_date=args.start_date, end_date=args.end_date)
        cli.run_race_cleaner()
        if cli.check_db_has_races():
            cli.run_horse_scraper()
            cli.run_horse_cleaner()
        return

    if args.scrape_races:
        cli.run_race_scraper(
            start_date=args.start_date, end_date=args.end_date
        )

    if args.clean_races:
        cli.run_race_cleaner()

    if args.scrape_horses:
        cli.run_horse_scraper()

    if args.clean_horses:
        cli.run_horse_cleaner()


if __name__ == "__main__":
    main()