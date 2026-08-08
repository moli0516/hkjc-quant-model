import argparse
import asyncio
from datetime import datetime
import gc
import importlib
import logging
import sys
from pathlib import Path

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 模組匯入
try:
    from cleaners.cleaner_pipeline import CleaningPipeline
    from config.settings import settings
    from database.db_manager import DBManager
    from features.feature_pipeline import FeaturesPipeline
    from scraper.horse_pipeline import HorseScrapingPipeline
    from scraper.race_pipeline import RaceScrapingPipeline

    # 模型管線與資料載入器匯入
    from models.data_loader import RaceDataLoader
    from models.model_pipeline import ModelPipeline
except ImportError as e:
    print(f"❌ 模組匯入失敗，請確認執行路徑與專案目錄結構: {e}")
    sys.exit(1)


class HKJCCLI:

    def __init__(self):
        self.db = DBManager()
        self.trained_model = None  # 儲存當前訓練好的模型記憶體指標

    def check_db_has_races(self) -> bool:
        """檢查資料庫是否存在賽果數據 (race_results)"""
        has_data = self.db.has_race_results()
        if not has_data:
            print(
                "\n⚠️ [前置檢查失敗] 資料庫中找不到任何賽果數據 (race_results)！"
            )
            print("👉 請先執行：1. 賽果/分段爬蟲 ➔ 2. 賽果/分段數據清洗\n")
        return has_data

    def check_db_has_features(self) -> bool:
        """檢查資料庫是否存在特徵矩陣 (feature_matrix)"""
        has_features = (
            self.db.has_feature_matrix()
            if hasattr(self.db, "has_feature_matrix")
            else True
        )
        if not has_features:
            print(
                "\n⚠️ [前置檢查失敗] 資料庫中找不到特徵矩陣數據 (feature_matrix)！"
            )
            print("👉 請先執行：Step 9 特徵工程 Pipeline (Features Pipeline)\n")
        return has_features

    def _prompt_year_range(self) -> tuple[int, int]:
        """互動式詢問爬蟲開始與結束年份，預設為今年"""
        current_year = datetime.now().year
        
        start_in = input(f"請輸入爬蟲開始年份 [預設 {current_year}]: ").strip()
        start_year = int(start_in) if start_in.isdigit() else current_year

        end_in = input(f"請輸入爬蟲結束年份 [預設 {current_year}]: ").strip()
        end_year = int(end_in) if end_in.isdigit() else current_year

        if start_year > end_year:
            print(f"⚠️ 開始年份 ({start_year}) 大於結束年份 ({end_year})，已自動調整為相同年份。")
            end_year = start_year

        return start_year, end_year

    def switch_settings(self, config_name: str = None) -> bool:
        """切換 settings 設定檔並觸發模組熱重載以套用新設定"""
        config_dir = settings.config_dir

        if config_name is None:
            # 互動式搜尋與選擇 JSON 設定檔
            json_files = sorted([f.name for f in config_dir.glob("*.json")])
            current_config = settings.config_path.name

            print("\n⚙️  [設定檔切換選單]")
            print(f"📌 當前生效設定檔: {current_config}")
            print("📁 可用的設定檔列表:")
            for idx, file_name in enumerate(json_files, 1):
                prefix = "👉 " if file_name == current_config else "   "
                print(f"  {prefix}{idx}. {file_name}")

            choice = input(
                f"\n請選擇設定檔編號 (1-{len(json_files)}) 或直接輸入檔名 [留空取消]: "
            ).strip()

            if not choice:
                print("ℹ️  已取消切換設定檔。\n")
                return False

            if choice.isdigit() and 1 <= int(choice) <= len(json_files):
                config_name = json_files[int(choice) - 1]
            else:
                config_name = choice

        try:
            active_name = settings.switch_config(config_name)
            print(f"✅ 設定檔已成功切換並持久化儲存為: {active_name}")
            # 切換設定檔後自動進行熱重載，確保所有 Module 與 Pipeline 重新載入新配置
            self.reload_modules()
            return True
        except Exception as e:
            print(f"❌ 切換設定檔失敗: {e}\n")
            return False

    def reload_modules(self):
        """動態熱重載 (Hot Reload) 所有專案模組、生成器與 Pipeline"""
        print("\n🔄 正在掃描並熱重載所有專案模組...")

        project_prefixes = (
            "cleaners",
            "config",
            "database",
            "features",
            "models",
            "scraper",
            "utils",
        )

        # 找出目前所有屬於專案且已載入的模組
        target_modules = [
            mod_name
            for mod_name in list(sys.modules.keys())
            if mod_name.startswith(project_prefixes)
            and sys.modules[mod_name] is not None
        ]

        reloaded_count = 0
        for mod_name in target_modules:
            try:
                importlib.reload(sys.modules[mod_name])
                reloaded_count += 1
            except Exception as e:
                logger.warning(f"⚠️ 模組 {mod_name} 重載失敗: {e}")

        # 重新更新當前 CLI 作用域內的類別與 settings 引用
        try:
            global CleaningPipeline, FeaturesPipeline, HorseScrapingPipeline
            global RaceScrapingPipeline, RaceDataLoader, ModelPipeline, DBManager, settings

            from cleaners.cleaner_pipeline import CleaningPipeline
            from config.settings import settings
            from database.db_manager import DBManager
            from features.feature_pipeline import FeaturesPipeline
            from models.data_loader import RaceDataLoader
            from models.model_pipeline import ModelPipeline
            from scraper.horse_pipeline import HorseScrapingPipeline
            from scraper.race_pipeline import RaceScrapingPipeline

            # 重新實例化 DBManager 並清空舊模型的記憶體快取
            self.db = DBManager()
            self.trained_model = None

            print(
                f"✅ 成功熱重載 {reloaded_count} 個模組！所有 Pipeline 類別與設定已更新至最新狀態。\n"
            )
        except Exception as e:
            print(f"❌ 類別重新繫結失敗: {e}\n")

    # ---------------- 核心功能調用 ----------------
    def run_race_scraper(self, start_year: int = None, end_year: int = None):
        print("🚀 [Step 1] 開始執行：賽果與分段時間爬蟲...")
        if start_year is None or end_year is None:
            start_year, end_year = self._prompt_year_range()

        print(f"📅 爬蟲目標年份區間: {start_year} 年 ~ {end_year} 年")
        scraper = RaceScrapingPipeline()
        years = list(range(start_year, end_year + 1))
        asyncio.run(scraper.run(years))
        print("✅ 賽果與分段時間爬蟲完成！\n")

    def run_race_cleaner(self):
        print("🧹 [Step 2] 開始執行：賽果與分段時間數據清洗...")
        cleaner = CleaningPipeline()
        cleaner.run(action="race_sectional")
        print("✅ 賽果與分段時間數據清洗完成，已寫入資料庫！\n")

    def run_horse_scraper(self):
        print("🐎 [Step 3] 檢查資料庫狀態以進行馬匹資料爬蟲...")
        if not self.check_db_has_races():
            return

        print("🚀 開始執行：馬匹資料爬蟲...")
        horse_ids = self.db.get_pending_horse_ids()
        print(f"📊 找到 {len(horse_ids)} 匹需要更新的馬匹資料。")

        if horse_ids:
            scraper = HorseScrapingPipeline()
            asyncio.run(scraper.run(horse_ids))
            print("✅ 馬匹資料爬蟲完成！\n")
        else:
            print("ℹ️ 沒有需要爬取的馬匹 ID。\n")

    def run_horse_cleaner(self):
        print("🧹 [Step 4] 開始執行：馬匹資料數據清洗...")
        cleaner = CleaningPipeline()
        cleaner.run(action="horse")
        print("✅ 馬匹資料數據清洗完成，已更新至資料庫！\n")

    def run_trackwork_scraper(self, start_year: int = None, end_year: int = None):
        """執行晨操 (Trackwork) 資料爬蟲"""
        print("🏇 [Step 5] 開始執行：晨操數據爬蟲 (Trackwork Scraper)...")
        if start_year is None or end_year is None:
            start_year, end_year = self._prompt_year_range()

        print(f"📅 晨操爬蟲目標年份區間: {start_year} 年 ~ {end_year} 年")
        try:
            from scraper.trackwork_pipeline import TrackworkScrapingPipeline
            scraper = TrackworkScrapingPipeline()
            asyncio.run(scraper.run(start_year, end_year))
            print("✅ 晨操數據爬蟲完成！\n")
        except ImportError:
            print("⚠️ 找不到 `scraper.trackwork_pipeline` 模組，請確認模組建立狀態。\n")
        except Exception as e:
            print(f"❌ 晨操數據爬蟲執行失敗: {e}\n")

    def run_trackwork_cleaner(self):
        """執行晨操 (Trackwork) 數據清洗"""
        print("🧹 [Step 6] 開始執行：晨操數據清洗 (Trackwork Cleaner)...")
        try:
            cleaner = CleaningPipeline()
            cleaner.run(action="trackwork")
            print("✅ 晨操數據清洗完成，已更新至資料庫！\n")
        except Exception as e:
            print(f"❌ 晨操數據清洗執行失敗: {e}\n")

    def run_trail_scraper(self, start_year: int = None, end_year: int = None):
        """執行試閘 (Barrier Trials) 資料爬蟲"""
        print("🏇 [Step 7] 開始執行：試閘數據爬蟲 (Trail Scraper)...")
        if start_year is None or end_year is None:
            start_year, end_year = self._prompt_year_range()

        print(f"📅 試閘爬蟲目標年份區間: {start_year} 年 ~ {end_year} 年")
        try:
            from scraper.trail_pipeline import TrailScrapingPipeline
            scraper = TrailScrapingPipeline()
            asyncio.run(scraper.run(start_year, end_year))
            print("✅ 試閘數據爬蟲完成！\n")
        except ImportError:
            print("⚠️ 找不到 `scraper.trail_pipeline` 模組，請確認模組建立狀態。\n")
        except Exception as e:
            print(f"❌ 試閘數據爬蟲執行失敗: {e}\n")

    def run_trail_cleaner(self):
        """新增：執行試閘 (Barrier Trials) 數據清洗"""
        print("🧹 [Step 8] 開始執行：試閘數據清洗 (Trail Cleaner)...")
        try:
            cleaner = CleaningPipeline()
            cleaner.run(action="trails")
            print("✅ 試閘數據清洗完成，已更新至資料庫！\n")
        except Exception as e:
            print(f"❌ 試閘數據清洗執行失敗: {e}\n")

    def run_features_pipeline(self):
        """執行全量量化特徵工程 Pipeline (一次性計算以避免冷啟動斷層與時間洩漏)"""
        print("⚙️ [Step 9] 開始執行：全量量化特徵工程 Pipeline...")
        if not self.check_db_has_races():
            return

        try:
            # 1. 一次性載入全量賽事數據
            print("📥 正在從資料庫載入全量賽事歷史數據...")
            raw_df = self.db.load_all_merged_race_data()

            if raw_df.empty:
                print("⚠️ 未找到任何賽事記錄，終止特徵工程。")
                return

            # 2. 嚴格依時間與賽事順序排序，確保 shift(1) 與滾動統計時序完全正確
            print("⏳ 正在進行數據時序排序 (date, race_id)...")
            raw_df = raw_df.sort_values(
                ["date", "race_id", "horse_id"]
            ).reset_index(drop=True)

            print(f"📊 載入完成，共計 {len(raw_df)} 筆數據。開始計算特徵矩陣...")

            # 3. 呼叫 FeaturesPipeline 生成特徵
            pipeline = FeaturesPipeline(key_cols=["race_id", "horse_id"])
            feature_df = pipeline.run(df=raw_df)

            # 4. 一次性覆蓋寫入資料庫
            print("💾 正在將全量特徵矩陣寫入資料庫 (feature_matrix)...")
            self.db.save_feature_matrix(
                df=feature_df,
                table_name="feature_matrix",
                if_exists="replace",  # 一次性全量覆蓋
            )

            print(
                f"✅ 全量特徵工程計算完成！已成功寫入 {len(feature_df)} 筆數據至資料庫。\n"
            )

            # 5. 記憶體回收
            del raw_df, feature_df
            gc.collect()

        except Exception as e:
            print(f"❌ 特徵工程執行失敗: {e}\n")

    def run_model_pipeline(
        self, model_type: str = "xgb_ranker", val_days: int = 90
    ):
        """[Step 10] 執行機器學習模型訓練與長週期 Out-of-Time 財務回測"""
        print(f"🤖 [Step 10] 開始執行 Model Pipeline (驗證窗口: {val_days} 天)...")
        if not self.check_db_has_features():
            return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            print(f"🎯 使用模型架構: {model_type.upper()}")

            model, metrics = model_pipe.run_train_pipeline(
                model_name=model_type, val_days=val_days
            )

            self.trained_model = model

            print(f"✅ 模型訓練與 {val_days} 天 Out-of-Time 驗證完成！")
            print(f"📊 評估指標詳細結果:")
            for metric_name, val in metrics.items():
                print(f"   ├─ {metric_name}: {val:.4f}" if isinstance(val, float) else f"   ├─ {metric_name}: {val}")
            print()
            return model, metrics

        except Exception as e:
            print(f"❌ 模型 Pipeline 執行失敗: {e}\n")

    def run_tune_pipeline(
        self,
        model_type: str = "xgb_ranker",
        n_trials: int = 30,
        val_days: int = 90,
        metric_name: str = "top1_win_rate",
    ):
        """[Step 11] 執行 Optuna 自動超參數尋優 Pipeline"""
        print("🎯 [Step 11] 開始執行：Optuna 自動超參數尋優 (Model Tuning)...")
        if not self.check_db_has_features():
            return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            print(f"🎯 目標模型: {model_type.upper()} | 嘗試次數: {n_trials} | 優化目標: {metric_name}")

            best_params, best_model = model_pipe.run_tune_pipeline(
                model_name=model_type,
                n_trials=n_trials,
                val_days=val_days,
                metric_name=metric_name,
                direction="maximize",
                retrain_best=True,
            )

            if best_model is not None:
                self.trained_model = best_model

            print("✅ Optuna 自動尋優與最佳模型重練完成！")
            print("💡 最佳超參數組合如下:")
            for k, v in best_params.items():
                print(f"   ├─ {k}: {v}")
            print()
            return best_params, best_model

        except Exception as e:
            print(f"❌ 超參數尋優 Pipeline 執行失敗: {e}\n")

    def run_predictions(self, target_date: str = None):
        """[Step 12] 執行未來/最新賽事預測推論"""
        print("🔮 [Step 12] 開始執行：賽事勝率預測推論 (Model Inference)...")

        if self.trained_model is None:
            print("⚠️ 尚未在此 CLI 會話中訓練模型，嘗試自動啟動預設訓練流程...")
            self.run_model_pipeline()
            if self.trained_model is None:
                print("❌ 無法獲取有效的訓練模型，取消預測流程。\n")
                return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            data_loader = RaceDataLoader(self.db)

            print(f"📥 正在載入推論資料 (日期條件: {target_date or '最新數據'})...")
            inference_df, _, _ = data_loader.load_dataset(include_odds=True)

            if target_date and "date" in inference_df.columns:
                inference_df = inference_df[
                    inference_df["date"] == target_date
                ]

            if inference_df.empty:
                print("⚠️ 找不到符合條件的推論數據，請檢查資料庫狀態。\n")
                return

            result_df = model_pipe.run_inference_pipeline(
                model=self.trained_model, inference_df=inference_df
            )

            print("\n📊 賽事預測排名結果 (Top 10 範例)：")
            display_cols = [
                col
                for col in [
                    "race_id",
                    "horse_id",
                    "pred_score",
                    "pred_rank",
                    "placing",
                ]
                if col in result_df.columns
            ]
            print(result_df[display_cols].head(10).to_string(index=False))
            print("\n✅ 推論計算完成！\n")

        except Exception as e:
            print(f"❌ 賽事預測執行失敗: {e}\n")

    # ---------------- 互動式選單 ----------------
    def interactive_menu(self):
        while True:
            try:
                print("=" * 55)
                print(
                    f"🏇  HKJC 賽馬數據工程與機器學習模型 CLI 工具 (當前設定: {settings.config_path.name})"
                )
                print("=" * 55)
                print("1.  執行賽果和分段時間爬蟲")
                print("2.  進行賽果和分段時間數據清洗")
                print("3.  執行馬匹資料爬蟲 (需要先有賽果資料庫)")
                print("4.  進行馬匹資料數據清洗")
                print("5.  執行晨操資料爬蟲 (Trackwork Scraper)")
                print("6.  進行晨操資料數據清洗 (Trackwork Cleaner)")
                print("7.  🏇 執行試閘資料爬蟲 (Trail Scraper)")
                print("8.  🧹 進行試閘資料數據清洗 (Trail Cleaner)")
                print("9.  ⚙️  生成量化特徵矩陣 (Features Pipeline)")
                print("10. 🤖 訓練賽馬預測模型 (Model Pipeline)")
                print("11. 🎯 Optuna 自動尋優超參數 (Model Tuning)")
                print("12. 🔮 執行賽事勝率預測 (Inference)")
                print("13. ⚡ 執行一鍵全套 ETL + 特徵工程 + 模型訓練 (1 ➔ 10)")
                print("14. ⚙️  切換並設定生效 settings.json (Switch Settings)")
                print("15. 🔄 熱重載所有模組與腳本 (Reload Modules)")
                print("0.  退出系統")
                print("=" * 55)

                choice = (
                    input(
                        "請選擇要執行的功能編號 (0-15，或按 Ctrl+C 退出): "
                    )
                    .strip()
                )

                if choice == "1":
                    self.run_race_scraper()
                elif choice == "2":
                    self.run_race_cleaner()
                elif choice == "3":
                    self.run_horse_scraper()
                elif choice == "4":
                    self.run_horse_cleaner()
                elif choice == "5":
                    self.run_trackwork_scraper()
                elif choice == "6":
                    self.run_trackwork_cleaner()
                elif choice == "7":
                    self.run_trail_scraper()
                elif choice == "8":
                    self.run_trail_cleaner()
                elif choice == "9":
                    self.run_features_pipeline()
                elif choice == "10":
                    val_days_str = input("請輸入 val days (預設 90 日): ").strip()
                    val_days = int(val_days_str) if val_days_str.isdigit() else 90
                    self.run_model_pipeline(val_days=val_days)
                elif choice == "11":
                    trials_in = input(
                        "請輸入 Optuna 搜尋輪數 (預設 30 次): "
                    ).strip()
                    n_trials = int(trials_in) if trials_in.isdigit() else 30
                    self.run_tune_pipeline(n_trials=n_trials)
                elif choice == "12":
                    date_input = (
                        input(
                            "輸入預測日期 (YYYY-MM-DD，留空則預測最新賽事): "
                        ).strip()
                        or None
                    )
                    self.run_predictions(target_date=date_input)
                elif choice == "13":
                    print(
                        "\n🔄 開始一鍵執行全套 Pipeline (從爬蟲到模型訓練)..."
                    )
                    start_y, end_y = self._prompt_year_range()
                    self.run_race_scraper(start_year=start_y, end_year=end_y)
                    self.run_race_cleaner()
                    if self.check_db_has_races():
                        self.run_horse_scraper()
                        self.run_horse_cleaner()
                        self.run_trackwork_scraper(start_year=start_y, end_year=end_y)
                        self.run_trackwork_cleaner()
                        self.run_trail_scraper(start_year=start_y, end_year=end_y)
                        self.run_trail_cleaner()
                        self.run_features_pipeline()
                        self.run_model_pipeline()
                elif choice == "14":
                    self.switch_settings()
                elif choice == "15":
                    self.reload_modules()
                elif choice == "0":
                    print("👋 已退出 CLI 工具。")
                    break
                else:
                    print("❌ 無效選擇，請重新輸入！\n")

            except KeyboardInterrupt:
                print("\n\n⚠️ 收到使用者中斷指令 (Ctrl+C)！已取消當前執行的動作。")
                print("🧹 正在清理記憶體並返回主選單...\n")
                gc.collect()
                continue


def main():
    parser = argparse.ArgumentParser(
        description="HKJC 賽馬量化數據爬蟲、清洗、特徵工程與模型訓練管道 (CLI Tool)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--scrape-races", action="store_true", help="執行賽果與分段時間爬蟲"
    )
    parser.add_argument(
        "--clean-races", action="store_true", help="執行賽果與分段數據清洗"
    )
    parser.add_argument(
        "--scrape-horses", action="store_true", help="執行馬匹資料爬蟲"
    )
    parser.add_argument(
        "--clean-horses", action="store_true", help="執行馬匹數據清洗"
    )
    parser.add_argument(
        "--scrape-trackwork", action="store_true", help="執行晨操數據爬蟲"
    )
    parser.add_argument(
        "--clean-trackwork", action="store_true", help="執行晨操數據清洗"
    )
    parser.add_argument(
        "--scrape-trails", action="store_true", help="執行試閘數據爬蟲 (Trail Scraper)"
    )
    parser.add_argument(
        "--clean-trails", action="store_true", help="執行試閘數據清洗 (Trail Cleaner)"
    )
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="執行特徵工程矩陣生成 (Features Pipeline)",
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="執行模型訓練 Pipeline (Model Pipeline)",
    )
    parser.add_argument(
        "--tune-model",
        action="store_true",
        help="執行 Optuna 模型超參數尋優 Pipeline",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="執行賽事預測 (Inference)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="依序執行全套流程 (1 ➔ 10)",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="切換使用的設定檔路徑或檔名 (例: settings_dev.json)",
    )
    parser.add_argument(
        "--start-year", type=int, help="爬蟲起始年份 (YYYY)"
    )
    parser.add_argument(
        "--end-year", type=int, help="爬蟲結束年份 (YYYY)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgb_ranker",
        help="選擇的模型類型 (例: xgb_ranker, lgb_ranker)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna 超參數尋優搜尋試驗次數 (預設: 30)",
    )

    args = parser.parse_args()
    cli = HKJCCLI()

    if args.config:
        cli.switch_settings(args.config)

    # 若無指定任何執行動作的 CLI 旗標，開啟互動式介面
    if not any(
        [
            args.scrape_races,
            args.clean_races,
            args.scrape_horses,
            args.clean_horses,
            args.scrape_trackwork,
            args.clean_trackwork,
            args.scrape_trails,
            args.clean_trails,
            args.generate_features,
            args.train_model,
            args.tune_model,
            args.predict,
            args.all,
        ]
    ):
        cli.interactive_menu()
        return

    # 命令列參數驅動模式
    if args.all:
        curr_y = datetime.now().year
        s_y = args.start_year if args.start_year else curr_y
        e_y = args.end_year if args.end_year else curr_y
        cli.run_race_scraper(start_year=s_y, end_year=e_y)
        cli.run_race_cleaner()
        if cli.check_db_has_races():
            cli.run_horse_scraper()
            cli.run_horse_cleaner()
            cli.run_trackwork_scraper(start_year=s_y, end_year=e_y)
            cli.run_trackwork_cleaner()
            cli.run_trail_scraper(start_year=s_y, end_year=e_y)
            cli.run_trail_cleaner()
            cli.run_features_pipeline()
            cli.run_model_pipeline(model_type=args.model_type)
        return

    if args.scrape_races:
        cli.run_race_scraper(start_year=args.start_year, end_year=args.end_year)

    if args.clean_races:
        cli.run_race_cleaner()

    if args.scrape_horses:
        cli.run_horse_scraper()

    if args.clean_horses:
        cli.run_horse_cleaner()

    if args.scrape_trackwork:
        cli.run_trackwork_scraper(start_year=args.start_year, end_year=args.end_year)

    if args.clean_trackwork:
        cli.run_trackwork_cleaner()

    if args.scrape_trails:
        cli.run_trail_scraper(start_year=args.start_year, end_year=args.end_year)

    if args.clean_trails:
        cli.run_trail_cleaner()

    if args.generate_features:
        cli.run_features_pipeline()

    if args.tune_model:
        cli.run_tune_pipeline(
            model_type=args.model_type, n_trials=args.n_trials
        )

    if args.train_model:
        cli.run_model_pipeline(model_type=args.model_type)

    if args.predict:
        cli.run_predictions()


if __name__ == "__main__":
    main()