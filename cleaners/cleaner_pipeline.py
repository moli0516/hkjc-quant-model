import sys
from cleaners.races_cleaner import RaceCleaner
from cleaners.sectional_cleaner import SectionalCleaner
from database.db_manager import DBManager


class CleaningPipeline:

    def __init__(self):
        self.db = DBManager()
        self.race_cleaner = RaceCleaner()
        self.sectional_cleaner = SectionalCleaner()

    def run(self):
        print("🚀 [Pipeline] 初始化資料庫...")
        self.db.init_db()

        print("\n🧹 [Pipeline] 步驟 1/2: 開始清洗賽果數據...")
        race_data = self.race_cleaner.process()

        print("\n🧹 [Pipeline] 步驟 2/2: 開始清洗分段數據...")
        df_sectionals = self.sectional_cleaner.process()

        # 彙整給 DBManager 的資料表 Dictionary
        tables = {
            "races": race_data["races"],
            "race_results": race_data["race_results"],
            "race_sectionals": df_sectionals,
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"賽事總數 (races): {len(tables['races'])}")
        print(f"馬匹賽果總數 (race_results): {len(tables['race_results'])}")
        print(f"分段數據總數 (race_sectionals): {len(tables['race_sectionals'])}")

        print("\n💾 正在寫入資料庫...")
        try:
            self.db.insert_dataframes(tables)
            print("✨ 全套數據清洗並成功寫入資料庫！")
        except Exception as e:
            print(f"❌ 寫入資料庫時發生錯誤: {e}")
            sys.exit(1)


if __name__ == "__main__":
    pipeline = CleaningPipeline()
    pipeline.run()