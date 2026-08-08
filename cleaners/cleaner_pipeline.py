# cleaners/cleaner_pipeline.py

import sys
from cleaners.races_cleaner import RaceCleaner
from cleaners.sectional_cleaner import SectionalCleaner
from cleaners.horses_cleaner import HorseCleaner
from cleaners.trackwork_cleaner import TrackworkCleaner
from cleaners.trails_cleaner import TrailsCleaner
from database.db_manager import DBManager


class CleaningPipeline:

    def __init__(self):
        self.db = DBManager()
        self.race_cleaner = RaceCleaner()
        self.sectional_cleaner = SectionalCleaner()
        self.horse_cleaner = HorseCleaner()
        self.trackwork_cleaner = TrackworkCleaner()  # 🔒 正確實例化[cite: 1]
        self.trails_cleaner = TrailsCleaner()

    def process_race_sectional(self):
        print("\n🧹 [Pipeline] 步驟 1/2: 開始清洗賽果數據...")
        race_data = self.race_cleaner.process()

        print("\n🧹 [Pipeline] 步驟 2/2: 開始清洗分段數據...")
        df_sectionals = self.sectional_cleaner.process()

        tables = {
            "races": race_data["races"],
            "race_results": race_data["race_results"],
            "race_sectionals": df_sectionals,
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"賽事總數 (races): {len(tables['races'])}")
        print(f"馬匹賽果總數 (race_results): {len(tables['race_results'])}")
        print(f"分段數據總數 (race_sectionals): {len(tables['race_sectionals'])}")
        return tables

    def process_horse(self):
        print("\n🧹 [Pipeline] 開始清洗馬匹數據...")
        df_horses = self.horse_cleaner.process()

        tables = {
            "horses": df_horses
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"馬匹 Profiles 總數 (horses): {len(tables['horses'])}")
        return tables

    def process_trackwork(self):
        """清洗晨操與試閘數據"""
        print("\n🧹 [Pipeline] 開始清洗試閘數據...")
        df_trackwork = self.trackwork_cleaner.process()

        tables = {
            "race_trackwork": df_trackwork
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"晨操紀錄總數 (race_trackwork): {len(tables['race_trackwork'])}")
        return tables
    
    def process_trail(self):
        """清洗晨操與試閘數據"""
        print("\n🧹 [Pipeline] 開始清洗試閘數據...")
        trails_dict = self.trails_cleaner.process()

        tables = {
            "trails": trails_dict["trials"],
            "trail_results": trails_dict["trial_results"]
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"試閘組別總數 (race_trackwork): {len(tables['trails'])}")
        print(f"試閘紀錄總數 (race_trackwork): {len(tables['trail_results'])}")
        return tables

    def run(self, action: str):
        print(f"🧹 [Pipeline] Action type: {action}")
        
        tables = {}
        if action == "race_sectional":
            tables = self.process_race_sectional()
        elif action == "horse":
            tables = self.process_horse()
        elif action == "trackwork":
            tables = self.process_trackwork()
        elif action == "trails":
            tables = self.process_trail()
        else:
            raise ValueError(f"未知的清洗動作 (Unknown action): {action}")

        print("\n💾 正在寫入資料庫...")
        try:
            self.db.insert_dataframes(tables)
            print("✨ 數據清洗並成功寫入資料庫！")
        except Exception as e:
            print(f"❌ 寫入資料庫時發生錯誤: {e}")
            sys.exit(1)


if __name__ == "__main__":
    pipeline = CleaningPipeline()
    pipeline.run("race_sectional")