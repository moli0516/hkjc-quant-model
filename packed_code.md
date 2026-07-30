# Project Codebase: .

## Directory Structure

```
./
├── cleaners
│   ├── __init__.py
│   ├── cleaner_pipeline.py
│   ├── horses_cleaner.py
│   ├── races_cleaner.py
│   └── sectional_cleaner.py
├── config
│   ├── __init__.py
│   ├── settings.json
│   ├── settings.json.old
│   └── settings.py
├── database
│   ├── db_manager.py
│   └── models.py
├── features
│   ├── generators
│   │   ├── __init__.py
│   │   ├── body_weight_recovery.py
│   │   ├── class_performance.py
│   │   ├── context_relative.py
│   │   ├── horse_profile.py
│   │   ├── horse_rolling.py
│   │   ├── human_sire.py
│   │   ├── injury_rest.py
│   │   ├── interaction.py
│   │   ├── jockey_trainer_alpha.py
│   │   ├── jockey_trainer_synergy.py
│   │   ├── jt_recent_form.py
│   │   ├── odds_market.py
│   │   ├── pace_strategy.py
│   │   ├── rating_class.py
│   │   ├── ratinn_momentum.py
│   │   ├── sectional_brust.py
│   │   ├── sectional_speed.py
│   │   ├── speed_feature.py
│   │   ├── synergy_fitness.py
│   │   └── track_distance.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── leak_guard.py
│   │   ├── scale.py
│   │   ├── smoother.py
│   │   ├── time_calc.py
│   │   └── track_bias.py
│   ├── __init__.py
│   ├── base_target.py
│   └── feature_pipeline.py
├── models
│   ├── hyperopy
│   │   ├── __init__.py
│   │   └── optuna_tuner.py
│   ├── metrics
│   │   ├── __init__.py
│   │   └── ranking.py
│   ├── validation
│   │   ├── __init__.py
│   │   └── time_split.py
│   ├── wrappers
│   │   ├── __init__.py
│   │   └── xgb_wrapper.py
│   ├── __init__.py
│   ├── base_model.py
│   ├── data_loader.py
│   ├── model_pipeline.py
│   └── registry.py
├── scraper
│   ├── parser
│   │   ├── __init__.py
│   │   ├── calander_parser.py
│   │   ├── horse_parser.py
│   │   ├── rating_parser.py
│   │   ├── result_parser.py
│   │   └── sectional_parser.py
│   ├── __init__.py
│   ├── data_manager.py
│   ├── hook.py
│   ├── horse_pipeline.py
│   └── race_pipeline.py
├── .gitignore
├── __init__.py
└── cli.py
```

---

## Source Code

### File: `.gitignore`

```text
.venv/
.venv.old/
*__pycache__/
*.db
*.parquet
data/
createFolder.py
notebooks
anaconda_projects
.virtual_documents
.vscode/
backtests/
config/*.json
scraper_old/
predict.py
tests/
features_old/
.VSCodeCounter/
backtests/
project_codebase.txt
file_create.py
code_packer.py
train.py
models_old/
```

---

### File: `__init__.py`

```py

```

---

### File: `cli.py`

```py
import argparse
import asyncio
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
            print("👉 請先執行：Step 5 特徵工程 Pipeline (Features Pipeline)\n")
        return has_features

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

        # 重新更新當前 CLI 作用域內的類別引用
        try:
            global CleaningPipeline, FeaturesPipeline, HorseScrapingPipeline
            global RaceScrapingPipeline, RaceDataLoader, ModelPipeline, DBManager

            from cleaners.cleaner_pipeline import CleaningPipeline
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
                f"✅ 成功熱重載 {reloaded_count} 個模組！所有 Pipeline 類別已更新至最新版本。\n"
            )
        except Exception as e:
            print(f"❌ 類別重新繫結失敗: {e}\n")

    # ---------------- 核心功能調用 ----------------
    def run_race_scraper(self, start_date=None, end_date=None):
        print("🚀 [Step 1] 開始執行：賽果與分段時間爬蟲...")
        scraper = RaceScrapingPipeline()
        scraper.run(start_date=start_date, end_date=end_date)
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

    def run_features_pipeline(self):
        """執行全量量化特徵工程 Pipeline (一次性計算以避免冷啟動斷層與時間洩漏)"""
        print("⚙️ [Step 5] 開始執行：全量量化特徵工程 Pipeline...")
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
        self, model_type: str = "xgb_ranker", val_days: int = 30
    ):
        """[Step 6] 執行機器學習模型訓練 Pipeline"""
        print("🤖 [Step 6] 開始執行：量化模型訓練與評估 (Model Pipeline)...")
        if not self.check_db_has_features():
            return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            print(f"🎯 使用模型架構: {model_type.upper()}")

            # 調用 ModelPipeline 中的 run_train_pipeline
            model, metrics = model_pipe.run_train_pipeline(
                model_name=model_type, val_days=val_days
            )

            # 保存已訓練的模型供後續推論直接使用
            self.trained_model = model

            print("✅ 模型訓練與驗證完成！")
            print(f"📊 評估指標詳細結果:")
            for metric_name, val in metrics.items():
                print(f"   ├─ {metric_name}: {val:.4f}")
            print()
            return model, metrics

        except Exception as e:
            print(f"❌ 模型 Pipeline 執行失敗: {e}\n")

    def run_tune_pipeline(
        self,
        model_type: str = "xgb_ranker",
        n_trials: int = 30,
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
    ):
        """[Step 6.5] 執行 Optuna 自動超參數尋優 Pipeline"""
        print("🎯 [Step 6.5] 開始執行：Optuna 自動超參數尋優 (Model Tuning)...")
        if not self.check_db_has_features():
            return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            print(f"🎯 目標模型: {model_type.upper()} | 嘗試次數: {n_trials} | 優化目標: {metric_name}")

            # 調用 ModelPipeline 中的 run_tune_pipeline
            best_params, best_model = model_pipe.run_tune_pipeline(
                model_name=model_type,
                n_trials=n_trials,
                val_days=val_days,
                metric_name=metric_name,
                direction="maximize",
                retrain_best=True,  # 自動以最佳參數重練並返回模型
            )

            # 保存最佳模型供後續推論使用
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
        """[Step 7] 執行未來/最新賽事預測推論"""
        print("🔮 [Step 7] 開始執行：賽事勝率預測推論 (Model Inference)...")

        if self.trained_model is None:
            print("⚠️ 尚未在此 CLI 會話中訓練模型，嘗試自動啟動預設訓練流程...")
            self.run_model_pipeline()
            if self.trained_model is None:
                print("❌ 無法獲取有效的訓練模型，取消預測流程。\n")
                return

        try:
            model_pipe = ModelPipeline(db_manager=self.db)
            data_loader = RaceDataLoader(self.db)

            # 載入推論用的賽事數據
            print(f"📥 正在載入推論資料 (日期條件: {target_date or '最新數據'})...")
            inference_df, _, _ = data_loader.load_dataset(include_odds=True)

            if target_date and "date" in inference_df.columns:
                inference_df = inference_df[
                    inference_df["date"] == target_date
                ]

            if inference_df.empty:
                print("⚠️ 找不到符合條件的推論數據，請檢查資料庫狀態。\n")
                return

            # 調用 ModelPipeline 推論方法
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
                print("=" * 50)
                print("🏇  HKJC 賽馬數據工程與機器學習模型 CLI 工具")
                print("=" * 50)
                print("1. 執行賽果和分段時間爬蟲")
                print("2. 進行賽果和分段時間數據清洗")
                print("3. 執行馬匹資料爬蟲 (需要先有賽果資料庫)")
                print("4. 進行馬匹資料數據清洗")
                print("5. ⚙️  生成量化特徵矩陣 (Features Pipeline)")
                print("6. 🤖 訓練賽馬預測模型 (Model Pipeline)")
                print("T. 🎯 Optuna 自動尋優超參數 (Model Tuning)")
                print("7. 🔮 執行賽事勝率預測 (Inference)")
                print("8. ⚡ 執行一鍵全套 ETL + 特徵工程 + 模型訓練 (1 ➔ 6)")
                print("R. 🔄 熱重載所有模組與腳本 (Reload Modules)")
                print("0. 退出系統")
                print("=" * 50)

                choice = (
                    input("請選擇要執行的功能 (0-8 / T / R，或按 Ctrl+C 退出): ")
                    .strip()
                    .upper()
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
                    self.run_features_pipeline()
                elif choice == "6":
                    self.run_model_pipeline()
                elif choice == "T":
                    trials_in = input(
                        "請輸入 Optuna 搜尋輪數 (預設 30 次): "
                    ).strip()
                    n_trials = int(trials_in) if trials_in.isdigit() else 30
                    self.run_tune_pipeline(n_trials=n_trials)
                elif choice == "7":
                    date_input = (
                        input(
                            "輸入預測日期 (YYYY-MM-DD，留空則預測最新賽事): "
                        ).strip()
                        or None
                    )
                    self.run_predictions(target_date=date_input)
                elif choice == "8":
                    print(
                        "\n🔄 開始一鍵執行全套 Pipeline (從爬蟲到模型訓練)..."
                    )
                    self.run_race_scraper()
                    self.run_race_cleaner()
                    if self.check_db_has_races():
                        self.run_horse_scraper()
                        self.run_horse_cleaner()
                        self.run_features_pipeline()
                        self.run_model_pipeline()
                elif choice == "R":
                    self.reload_modules()
                elif choice == "0":
                    print("👋 已退出 CLI 工具。")
                    break
                else:
                    print("❌ 無效選擇，請重新輸入！\n")

            except KeyboardInterrupt:
                # 💡 當使用者按下 Ctrl + C 時捕捉訊號
                print("\n\n⚠️ 收到使用者中斷指令 (Ctrl+C)！已取消當前執行的動作。")
                print("🧹 正在清理記憶體並返回主選單...\n")
                gc.collect()  # 清理可能因中斷產生的孤立物件
                continue  # 繼續下一次迴圈，重新顯示主選單


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
        help="依序執行全套流程 (1 ➔ 2 ➔ 3 ➔ 4 ➔ 5 ➔ 6)",
    )

    parser.add_argument(
        "--start-date", type=str, help="賽事爬蟲/預測起始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, help="賽事爬蟲結束日期 (YYYY-MM-DD)"
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

    # 若無指定 CLI 旗標，開啟互動式介面
    if not any(
        [
            args.scrape_races,
            args.clean_races,
            args.scrape_horses,
            args.clean_horses,
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
        cli.run_race_scraper(
            start_date=args.start_date, end_date=args.end_date
        )
        cli.run_race_cleaner()
        if cli.check_db_has_races():
            cli.run_horse_scraper()
            cli.run_horse_cleaner()
            cli.run_features_pipeline()
            cli.run_model_pipeline(model_type=args.model_type)
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

    if args.generate_features:
        cli.run_features_pipeline()

    if args.tune_model:
        cli.run_tune_pipeline(
            model_type=args.model_type, n_trials=args.n_trials
        )

    if args.train_model:
        cli.run_model_pipeline(model_type=args.model_type)

    if args.predict:
        cli.run_predictions(target_date=args.start_date)


if __name__ == "__main__":
    main()
```

---

### File: `cleaners\__init__.py`

```py

```

---

### File: `cleaners\cleaner_pipeline.py`

```py
import sys
from cleaners.races_cleaner import RaceCleaner
from cleaners.sectional_cleaner import SectionalCleaner
from cleaners.horses_cleaner import HorseCleaner
from database.db_manager import DBManager


class CleaningPipeline:

    def __init__(self):
        self.db = DBManager()
        self.race_cleaner = RaceCleaner()
        self.sectional_cleaner = SectionalCleaner()
        self.horse_cleaner = HorseCleaner()

    def process_race_sectional(self):
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
        return tables

    def process_horse(self):
        print("\n🧹 [Pipeline] 開始清洗馬匹數據...")
        df_horses = self.horse_cleaner.process()

        # 彙整給 DBManager 的資料表 Dictionary
        tables = {
            "horses": df_horses
        }

        print("\n📊 --- 清洗統計 summary ---")
        print(f"馬匹 Profiles 總數 (horses): {len(tables['horses'])}")
        return tables

    def run(self, action):
        print(f"🧹 [Pipeline] Action type: {action}")
        if action == "race_sectional":
            tables = self.process_race_sectional()
        elif action == "horse":
            tables = self.process_horse()

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
```

---

### File: `cleaners\horses_cleaner.py`

```py
import json
import pathlib
import re
from datetime import datetime
import pandas as pd
from config.settings import settings


class HorseCleaner:

    def __init__(self):
        pass

    # ==========================================
    # 工具函數 (Static Methods)
    # ==========================================
    @staticmethod
    def parse_origin_age(val: str | None) -> tuple[str | None, int | None]:
        """解析出生地與年齡（例："澳洲 / 3" -> ("澳洲", 3)）"""
        if not val or pd.isna(val):
            return None, None
        parts = str(val).split("/")
        origin = parts[0].strip() if len(parts) > 0 else None
        age = None
        if len(parts) > 1:
            try:
                age = int(parts[1].strip())
            except ValueError:
                age = None
        return origin, age

    @staticmethod
    def parse_color_sex(val: str | None) -> tuple[str | None, str | None]:
        """解析毛色與性別（例："棗 / 閹" -> ("棗", "閹")）"""
        if not val or pd.isna(val):
            return None, None
        parts = str(val).split("/")
        color = parts[0].strip() if len(parts) > 0 else None
        sex = parts[1].strip() if len(parts) > 1 else None
        return color, sex

    @staticmethod
    def parse_stakes(val: str | None) -> float | None:
        """解析金額（例："$2,650,450" -> 2650450.0）"""
        if not val or pd.isna(val):
            return None
        # 移除非數字字符（保留小數點）
        clean_val = re.sub(r"[^\d.]", "", str(val))
        try:
            return float(clean_val) if clean_val else 0.0
        except ValueError:
            return None

    @staticmethod
    def parse_placing_records(
        val: str | None,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """解析冠亞季冠總次數（例："1-7-4-27" -> (1, 7, 4, 27)）"""
        if not val or pd.isna(val):
            return None, None, None, None
        parts = str(val).split("-")
        if len(parts) == 4:
            try:
                return (
                    int(parts[0]),
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                )
            except ValueError:
                pass
        return None, None, None, None

    @staticmethod
    def parse_date(val: str | None) -> str | None:
        """轉換日期格式（例："21/05/2026" -> "2026-05-21"）"""
        if not val or pd.isna(val):
            return None
        clean_val = str(val).strip()
        try:
            return datetime.strptime(clean_val, "%d/%m/%Y").strftime(
                "%Y-%m-%d"
            )
        except ValueError:
            return None

    @staticmethod
    def clean_int(val) -> int | None:
        """安全清理整數"""
        if val is None or pd.isna(val):
            return None
        try:
            return int(float(str(val).strip()))
        except ValueError:
            return None

    # ==========================================
    # 主清洗入口
    # ==========================================
    def process(
        self, horses_dir=settings.raw_horses_json_dir
    ) -> pd.DataFrame:
        horses_dir = pathlib.Path(horses_dir)
        horses_list = []

        horse_files = list(horses_dir.glob("*.json"))
        print(
            f"🔍 [HorseCleaner] 找到 {len(horse_files)} 個馬匹 Raw JSON 檔案..."
        )

        for file_path in horse_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                origin, age = self.parse_origin_age(
                    raw_data.get("origin_age")
                )
                color, sex = self.parse_color_sex(raw_data.get("color_sex"))
                wins, seconds, thirds, total_runs = (
                    self.parse_placing_records(raw_data.get("placing_records"))
                )

                horses_list.append({
                    "horse_code": raw_data.get("horse_code"),
                    "origin": origin,
                    "age": age,
                    "color": color,
                    "sex": sex,
                    "import_type": raw_data.get("import_type"),
                    "season_stakes": self.parse_stakes(
                        raw_data.get("season_stakes")
                    ),
                    "total_stakes": self.parse_stakes(
                        raw_data.get("total_stakes")
                    ),
                    "wins": wins,
                    "seconds": seconds,
                    "thirds": thirds,
                    "total_runs": total_runs,
                    "recent_10_races_count": self.clean_int(
                        raw_data.get("recent_10_races_count")
                    ),
                    "current_location": raw_data.get("current_location"),
                    "location_arrival_date": self.parse_date(
                        raw_data.get("location_arrival_date")
                    ),
                    "import_date": self.parse_date(raw_data.get("import_date")),
                    "trainer": raw_data.get("trainer"),
                    "owner": raw_data.get("owner"),
                    "current_rating": self.clean_int(
                        raw_data.get("current_rating")
                    ),
                    "season_start_rating": self.clean_int(
                        raw_data.get("season_start_rating")
                    ),
                    "sire": raw_data.get("sire"),
                    "dam": raw_data.get("dam"),
                    "damsire": raw_data.get("damsire"),
                })

            except Exception as e:
                print(
                    f"❌ [HorseCleaner] 解析檔案失敗 [{file_path.name}]: {e}"
                )

        df_horses = pd.DataFrame(horses_list)
        if not df_horses.empty:
            df_horses = df_horses.drop_duplicates(subset=["horse_code"])

        return df_horses
```

---

### File: `cleaners\races_cleaner.py`

```py
import json
import pathlib
import re
import pandas as pd
from config.settings import settings


class RaceCleaner:

    def __init__(self, rating_json_path=settings.rating_path):
        self.rating_json_path = rating_json_path
        self.rating_df = self._load_rating_df()

    def _load_rating_df(self) -> pd.DataFrame | None:
        """載入並預處理 Rating 資料"""
        if not pathlib.Path(self.rating_json_path).exists():
            return None
        try:
            with open(self.rating_json_path, "r", encoding="utf-8") as f:
                rating_json = json.load(f)
            df = pd.json_normalize(rating_json)
            if "horse_name" in df.columns:
                df["clean_horse_name"] = df["horse_name"].apply(
                    lambda x: self.extract_horse_info(x)[0]
                )
            return df
        except Exception as e:
            print(f"【警告】載入 Rating 資料失敗: {e}")
            return None

    # ==========================================
    # 工具函數 (Static Methods)
    # ==========================================
    @staticmethod
    def extract_horse_info(raw_name: str) -> tuple[str | None, str | None]:
        """解析馬名與烙號/horse_id"""
        if not isinstance(raw_name, str) or pd.isna(raw_name):
            return None, None

        match = re.search(r"\(([A-Z0-9]{3,5})\)", raw_name)
        horse_id = match.group(1) if match else None

        clean_name = (
            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", raw_name)
            .replace("\xa0", "")
            .strip()
        )
        return clean_name, horse_id

    @staticmethod
    def clean_head_horse_dist(hhd) -> float | None:
        """勝負距離轉為 float 馬身數"""
        if pd.isna(hhd) or hhd is None:
            return None
        if isinstance(hhd, (int, float)):
            return float(hhd)

        hhd_str = str(hhd).strip()
        if "-" in hhd_str and "/" in hhd_str:
            try:
                parts = hhd_str.split("-")
                frac = parts[1].split("/")
                return float(parts[0]) + float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        elif "/" in hhd_str:
            try:
                frac = hhd_str.split("/")
                return float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        else:
            margin_map = {
                "---": 0.0,
                "平頭馬": 0.0,
                "鼻位": 0.05,
                "短馬頭位": 0.1,
                "頭位": 0.2,
                "頸位": 0.3,
                "多個馬身": 99.0,
                "多個馬位": 99.0,
                "未能完成賽事": None,
                "退出": None,
            }
            return margin_map.get(
                hhd_str, float(hhd_str) if hhd_str.isdigit() else 0.25
            )

    @staticmethod
    def convert_min_to_sec(time_val) -> float | None:
        """時間轉秒數 (解析 1:10.12 或 23.88)"""
        if pd.isna(time_val) or time_val in ["---", "", None]:
            return None
        time_str = str(time_val).strip()
        try:
            if ":" in time_str:
                parts = time_str.split(":")
                return float(parts[0]) * 60.0 + float(parts[1])
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def extract_basic_info(
        basic_info_str: str,
    ) -> tuple[str | None, int | None]:
        """解析 basic_info -> 班次 (class) 與 路程 (distance)"""
        if not isinstance(basic_info_str, str) or pd.isna(basic_info_str):
            return None, None

        length_match = re.search(r"(\d{3,4})\s*米", basic_info_str)
        distance = int(length_match.group(1)) if length_match else None

        class_map = {
            "第一班": "1",
            "第二班": "2",
            "第三班": "3",
            "第四班": "4",
            "第五班": "5",
            "一級賽": "G1",
            "二級賽": "G2",
            "三級賽": "G3",
            "國際一級賽": "G1",
            "國際二級賽": "G2",
            "國際三級賽": "G3",
        }
        race_class = None
        for k, v in class_map.items():
            if k in basic_info_str:
                race_class = v
                break

        return race_class, distance

    @staticmethod
    def extract_track_info(text: str) -> tuple[str, str]:
        """解析 track_info -> 跑道材質 & 賽道 (A/B/C)"""
        if not isinstance(text, str) or pd.isna(text):
            return "未知", "N/A"
        clean_text = (
            text.replace('"', "").replace("“", "").replace("”", "").strip()
        )
        if "全天候" in clean_text or "泥地" in clean_text:
            return "泥地", "N/A"
        match = re.search(r"([^\s-]+)\s*-\s*([A-Za-z0-9+]+)", clean_text)
        if match:
            return match.group(1), match.group(2)
        return clean_text, "N/A"

    @staticmethod
    def clean_draw_value(val) -> int | None:
        """檔位清洗範例（防空字串、float 轉型）"""
        if val is None:
            return None
        val_str = str(val).strip()
        if not val_str or val_str.lower() in ["none", "null", "-", "--", ""]:
            return None
        try:
            return int(float(val_str))
        except (ValueError, TypeError):
            return None

    # ==========================================
    # 主清洗入口
    # ==========================================
    def process(
        self, races_dir=settings.raw_races_json_dir
    ) -> dict[str, pd.DataFrame]:
        races_dir = pathlib.Path(races_dir)
        races_list = []
        results_list = []

        hkjc_special_codes = [
            "DISQ",
            "DNF",
            "FE",
            "ML",
            "PU",
            "TNP",
            "TO",
            "UR",
            "VOID",
            "WR",
            "WV",
            "WV-A",
            "WX",
            "WX-A",
            "WXNR",
            "退出",
        ]

        race_files = list(races_dir.glob("*.json"))
        print(f"🔍 [RaceCleaner] 找到 {len(race_files)} 個賽事 Raw JSON 檔案...")

        for file_path in race_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)

                if not raw_data:
                    continue

                date = raw_data.get("date")
                venue = raw_data.get("venue")

                for race in raw_data.get("races") or []:
                    race_no = int(race["race_id"])
                    race_key = f"{date}_{venue}_{race_no}"

                    race_class, distance = self.extract_basic_info(
                        race.get("basic_info")
                    )
                    track_texture, track_type = self.extract_track_info(
                        race.get("track_info")
                    )

                    races_list.append({
                        "race_id": race_key,
                        "date": date,
                        "venue": venue,
                        "race_no": race_no,
                        "race_class": race_class,
                        "distance": distance,
                        "track_condition": race.get("track_condition"),
                        "track_texture": track_texture,
                        "track_type": track_type,
                    })

                    for horse in race.get("horses") or []:
                        placing_raw = str(horse.get("placing", "")).strip()

                        if (
                            placing_raw in hkjc_special_codes
                            or not placing_raw
                        ):
                            continue

                        clean_name, horse_id = self.extract_horse_info(
                            horse.get("horse_name")
                        )
                        time_raw = horse.get("finish_time") or horse.get(
                            "finished_time"
                        )
                        margin_raw = horse.get("margin") or horse.get(
                            "head_horse_dist"
                        )

                        placing_match = re.search(r"(\d+)", placing_raw)
                        placing = (
                            int(placing_match.group(1))
                            if placing_match
                            else None
                        )

                        results_list.append({
                            "race_id": race_key,
                            "horse_id": horse.get("horse_id"),
                            "horse_name": clean_name,
                            "placing": placing,
                            "draw": self.clean_draw_value(horse.get("draw")),
                            "jockey": horse.get("jockey"),
                            "trainer": horse.get("trainer"),
                            "actual_weight": (
                                float(horse.get("actual_weight"))
                                if horse.get("actual_weight")
                                else None
                            ),
                            "declared_weight": (
                                float(horse.get("body_weight"))
                                if horse.get("body_weight")
                                else None
                            ),
                            "win_odds": (
                                float(horse.get("odds"))
                                if horse.get("odds")
                                else None
                            ),
                            "finish_time_sec": self.convert_min_to_sec(
                                time_raw
                            ),
                            "margin_len": self.clean_head_horse_dist(
                                margin_raw
                            ),
                        })
            except Exception as e:
                print(f"❌ [RaceCleaner] 解析檔案失敗 [{file_path.name}]: {e}")

        df_races = pd.DataFrame(races_list).drop_duplicates(subset=["race_id"])
        df_results = pd.DataFrame(results_list)

        # 進行 Rating 資料合併
        if self.rating_df is not None and not df_results.empty:
            df_results = df_results.merge(
                self.rating_df[["clean_horse_name", "rating"]],
                left_on="horse_name",
                right_on="clean_horse_name",
                how="left",
            ).drop(columns=["clean_horse_name"], errors="ignore")

        return {"races": df_races, "race_results": df_results}
```

---

### File: `cleaners\sectional_cleaner.py`

```py
import json
import pathlib
import re
import pandas as pd
from config.settings import settings


class SectionalCleaner:

    @staticmethod
    def extract_horse_info(raw_name: str) -> tuple[str | None, str | None]:
        if not isinstance(raw_name, str) or pd.isna(raw_name):
            return None, None

        match = re.search(r"\(([A-Z0-9]{3,5})\)", raw_name)
        horse_id = match.group(1) if match else None

        clean_name = (
            re.sub(r"\s*\([A-Z0-9]{3,5}\)", "", raw_name)
            .replace("\xa0", "")
            .strip()
        )
        return clean_name, horse_id

    @staticmethod
    def convert_min_to_sec(time_val) -> float | None:
        if pd.isna(time_val) or time_val in ["---", "", None]:
            return None
        time_str = str(time_val).strip()
        try:
            if ":" in time_str:
                parts = time_str.split(":")
                return float(parts[0]) * 60.0 + float(parts[1])
            else:
                return float(time_str)
        except (ValueError, IndexError):
            return None
        
    @staticmethod
    def clean_head_horse_dist(hhd) -> float | None:
        """勝負距離轉為 float 馬身數"""
        if pd.isna(hhd) or hhd is None:
            return None
        if isinstance(hhd, (int, float)):
            return float(hhd)

        hhd_str = str(hhd).strip()
        if "-" in hhd_str and "/" in hhd_str:
            try:
                parts = hhd_str.split("-")
                frac = parts[1].split("/")
                return float(parts[0]) + float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        elif "/" in hhd_str:
            try:
                frac = hhd_str.split("/")
                return float(frac[0]) / float(frac[1])
            except (ValueError, IndexError):
                return 0.0
        else:
            margin_map = {
                "---": 0.0,
                "DH": 0.0,    # Dead Heat (平頭馬)
                "N": 0.05,    # Nose (鼻位)
                "SH": 0.1,    # Short Head (短馬頭位)
                "H": 0.2,     # Head (頭位)
                "N": 0.3,     # Neck (頸位 - 如果怕跟 Nose 衝突，通常可以用 K 或 NK)
                "ML": 99.0,   # Multiple Lengths / Many Lengths (多個馬身)
                "DNF": None,  # Did Not Finish (未能完成賽事)
                "WV": None,   # Withdrawn (退出)
            }
            return margin_map.get(
                hhd_str, float(hhd_str) if hhd_str.isdigit() else 0.25
            )

    def process(
        self, sectionals_dir=settings.raw_sectional_json_dir
    ) -> pd.DataFrame:
        sectionals_dir = pathlib.Path(sectionals_dir)
        sec_list = []

        sec_files = list(sectionals_dir.glob("*.json"))
        print(
            f"🔍 [SectionalCleaner] 找到 {len(sec_files)} 個分段時間 Raw JSON 檔案..."
        )

        for file_path in sec_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    sec_data = json.load(f)

                if not sec_data or not isinstance(sec_data, dict):
                    continue

                date = sec_data.get("date")
                venue = sec_data.get("venue")

                items = (
                    sec_data.get("sectionals")
                    or sec_data.get("races")
                    or sec_data.get("sectional_data")
                    or []
                )

                for item in items:
                    sec_race_no = item.get("race_no") or item.get("race_id")
                    if not sec_race_no or not date or not venue:
                        continue

                    race_key = f"{date}_{venue}_{int(sec_race_no)}"

                    horse_list = (
                        item.get("sectional_data")
                        or item.get("horses")
                        or item.get("details")
                        or []
                    )

                    for horse_sec in horse_list:
                        clean_name, horse_id = self.extract_horse_info(
                            horse_sec.get("horse_name") or horse_sec.get("name")
                        )

                        details = (
                            horse_sec.get("sectional_details")
                            or horse_sec.get("sectionals")
                            or []
                        )

                        for idx, detail in enumerate(details, 1):
                            sec_no = detail.get("section_no") or idx
                            position = detail.get("position") or detail.get(
                                "pos"
                            )
                            sec_time = detail.get(
                                "sectional_time"
                            ) or detail.get("time")
                            margin = detail.get("margin") or detail.get(
                                "behind"
                            )

                            sec_list.append({
                                "race_id": race_key,
                                "horse_name": clean_name,
                                "horse_id": horse_sec.get("horse_id"),
                                "section_no": int(sec_no),
                                "position": (
                                    int(position)
                                    if str(position).isdigit()
                                    else None
                                ),
                                "sectional_time_sec": self.convert_min_to_sec(
                                    sec_time
                                ),
                                "margin_behind": (
                                    self.clean_head_horse_dist(margin)
                                ),
                            })
            except Exception as e:
                print(
                    f"❌ [SectionalCleaner] 解析分段檔案失敗 [{file_path.name}]: {e}"
                )

        return pd.DataFrame(sec_list)
```

---

### File: `config\__init__.py`

```py

```

---

### File: `config\settings.json`

```json
{
    "last_updated": "2026-07-16",
    "smoothing_params": {
        "default_alpha": 20,
        "alphas": {
            "jockey": 20,
            "trainer": 20,
            "jockey_trainer": 40,
            "horse_id": 8,
            "draw": 30,
            "horse_track": 25,
            "horse_env": 35,
            "horse_yield": 40,
            "jockey_track": 15,
            "jockey_env": 20,
            "jockey_yield": 25,
            "trainer_track": 20,
            "trainer_yield": 30
        }
    },
    "rating_map": {
        "1": 100, "2": 85, "3": 70, "4": 50, "5": 30
    },
    "track_bias_map": {
        "A": 1.2, "B": 1.1, "C": 0.8, "C+3": 0.7
    },
    "active_features": [
        "j_smoothed_place_rate", 
        "h_smoothed_place_rate", 
        "win_odds_inv", 
        "log_win_odds"
    ],
    "paths": {
        "raw_json_dir": "data/raw_json",
        "raw_races_json_dir": "races",
        "raw_sectional_json_dir": "sectional",
        "raw_horses_json_dir": "horses",
        "horses_sub_dir": "horses",
        "races_sub_dir": "races",
        "flattened_json_dir": "data/cleaned_json/flatten",
        "normalized_json_dir": "data/cleaned_json/normalized",
        "rating_json_path": "data/horses_rating.json",
        "features_parquet_path": "data/features.parquet",
        "today_rc_json_path": "data/today_rc.json"
    },
    "data_loader": {
        "id_cols": ["race_id", "horse_id", "horse_name"],
        "target_cols": ["placing", "is_win", "is_top3"],
        "eval_cols": ["win_odds", "draw", "jockey", "trainer", "date"],
        "categorical_cols": ["brand_prefix", "course_type", "track_draw_key"]
    },
    "base_features": [
        "j_smoothed_win_rate",
        "form_x_rank_weight",
        "j_smoothed_place_rate",
        "h_track_smoothed_place_rate",
        "j_track_smoothed_place_rate",
        "jockey_adaptability_x_rank_weight",
        "h_smoothed_rolling_5_texture_place_rate_sand",
        "h_smoothed_rolling_5_place_rate",
        "adj_draw",
        "jt_smoothed_place_rate"
    ],
    "candidate_features": [
        "j_yield_smoothed_place_rate",
        "h_env_smoothed_place_rate",
        "j_smoothed_rolling_30_place_rate",
        "jt_smoothed_rolling_15_place_rate",
        
        "h_race_count_history",
        "d_smoothed_win_rate",
        "d_smoothed_place_rate",
        "weight_delta",
        "j_env_smoothed_place_rate",
        "draw_speed_interaction",
        "jt_smoothed_win_rate"
    ],
    "banned_features": [
        "rating_strength_score",
        "z_rating_vs_race_avg",
        "h_smoothed_rolling_5_texture_win_rate_turf"
    ],
    "PHYSICAL_FEATURES": [],
    "target": "is_place"
}
```

---

### File: `config\settings.json.old`

```old
{
    "last_updated": "2026-07-16",
    "smoothing_params": {
        "default_alpha": 20,
        "alphas": {
            "jockey": 20,
            "trainer": 20,
            "jockey_trainer": 40,
            "horse_id": 8,
            "draw": 30,
            "horse_track": 25,
            "horse_env": 35,
            "horse_yield": 40,
            "jockey_track": 15,
            "jockey_env": 20,
            "jockey_yield": 25,
            "trainer_track": 20,
            "trainer_yield": 30
        }
    },
    "rating_map": {
        "1": 100, "2": 85, "3": 70, "4": 50, "5": 30
    },
    "track_bias_map": {
        "A": 1.2, "B": 1.1, "C": 0.8, "C+3": 0.7
    },
    "active_features": [
        "j_smoothed_place_rate", 
        "h_smoothed_place_rate", 
        "win_odds_inv", 
        "log_win_odds"
    ],
    "paths": {
        "raw_json_dir": "data/raw_json",
        "horses_sub_dir": "horses",
        "races_sub_dir": "races",
        "flattened_json_dir": "data/cleaned_json/flatten",
        "normalized_json_dir": "data/cleaned_json/normalized",
        "rating_json_path": "data/horses_rating.json",
        "features_parquet_path": "data/features.parquet",
        "today_rc_json_path": "data/today_rc.json"
    },
    "latest_features": [
        "h_mean_speed_z_15",
        "h_speed_z_momentum",
        "h_rolling_2_speed_z_std",
        "h_rolling_15_speed_z_std",
        "h_race_count_history",
        "rating",
        "rating_is_real",
        "adj_draw",
        "draw_speed_interaction",
        "rating_vs_race_avg",
        "rating_strength_score",
        "win_odds_inv",
        "log_win_odds",
        "j_smoothed_win_rate",
        "t_smoothed_win_rate",
        "jt_smoothed_win_rate",
        "h_smoothed_win_rate",
        "d_smoothed_win_rate",
        "j_smoothed_place_rate",
        "t_smoothed_place_rate",
        "jt_smoothed_place_rate",
        "h_smoothed_place_rate",
        "d_smoothed_place_rate",
        "h_track_smoothed_win_rate",
        "h_track_smoothed_place_rate",
        "h_env_smoothed_win_rate",
        "h_env_smoothed_place_rate",
        "h_yield_smoothed_win_rate",
        "h_yield_smoothed_place_rate",
        "j_track_smoothed_place_rate",
        "j_env_smoothed_place_rate",
        "j_yield_smoothed_place_rate",
        "t_track_smoothed_place_rate",
        "t_yield_smoothed_place_rate",
        "j_smoothed_rolling_30_win_rate",
        "t_smoothed_rolling_30_win_rate",
        "jt_smoothed_rolling_15_win_rate",
        "h_smoothed_rolling_5_win_rate",
        "j_smoothed_rolling_30_place_rate",
        "t_smoothed_rolling_30_place_rate",
        "jt_smoothed_rolling_15_place_rate",
        "h_smoothed_rolling_5_place_rate"
    ],
    "meta_columns": [
        "race_unique_id",
        "actual_rank_score",
        "date",
        "races.race_id",
        "horse_id",
        "placing"
    ],
    "PHYSICAL_FEATURES": ["weight", "rank_weight"]
}
```

---

### File: `config\settings.py`

```py
import json
import pathlib

class Settings:
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parent.parent
        self.config_path = self.root_dir / "config" / "settings.json"
        self._data = self._load()

    def _load(self):
        with open(self.config_path, 'r', encoding="utf-8") as f:
            return json.load(f)
        
    @property
    def active_features(self):
        return self._data.get("active_features", [])
    
    @property
    def base_features(self):
        return self._data.get("base_features", [])
    
    @property
    def candidate_features(self):
        return self._data.get("candidate_features", [])
    
    @property
    def smoothing_alphas(self):
        return self._data.get("smoothing_params", {}).get("alphas", {})
    
    @property
    def raw_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("raw_json_dir", "")
    
    @property
    def raw_races_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_races_json_dir", "")
        
    @property
    def raw_sectional_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_sectional_json_dir", "")
    
    @property
    def raw_horses_json_dir(self):
        return self.raw_json_dir / self._data.get("paths", {}).get("raw_horses_json_dir", "")

    @property
    def flattened_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("flattened_json_dir", "")

    @property
    def normalized_json_dir(self):
        return self.root_dir / self._data.get("paths", {}).get("normalized_json_dir", "")

    @property
    def horses_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get("horsess_dir", "")

    @property
    def races_dir(self):
        return self.normalized_json_dir / self._data.get("paths", {}).get("races_dir", "")

    @property
    def rating_path(self):
        return self.root_dir / self._data.get("paths", {}).get("rating_json_path", "")
    
    @property
    def features_parquet_path(self):
        return self.root_dir / self._data.get("paths", {}).get("features_parquet_path", "")
    
    @property
    def today_rc_path(self):
        return self.root_dir / self._data.get("paths", {}).get("today_rc_json_path", "")
    
    @property
    def target(self):
        return self._data.get("target", [])
        
    def get_feature_group(self, group_name):
        return self._data.get("feature_groups", {}).get(group_name, [])

    # ------------------ 新增: DataLoader 屬性 ------------------

    @property
    def data_loader_config(self):
        """獲取完整 data_loader 設定字典"""
        return self._data.get("data_loader", {})

    @property
    def id_cols(self):
        """主鍵與識別欄位"""
        return self.data_loader_config.get("id_cols", ["race_id", "horse_id", "horse_name"])

    @property
    def target_cols(self):
        """目標/標籤欄位"""
        return self.data_loader_config.get("target_cols", ["placing", "is_win", "is_top3"])

    @property
    def eval_cols(self):
        """評估與特徵排除欄位"""
        return self.data_loader_config.get("eval_cols", ["win_odds", "draw", "jockey", "trainer", "date"])

    @property
    def categorical_cols(self):
        """類別型特徵欄位"""
        return self.data_loader_config.get("categorical_cols", ["brand_prefix", "course_type", "track_draw_key"])

settings = Settings()
```

---

### File: `database\db_manager.py`

```py
import gc
import pathlib
import pandas as pd
from database.models import Base  # 從 models 載入 Base
from sqlalchemy import Column, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker


class DBManager:

    def __init__(
        self, db_path=pathlib.Path(__file__).parent / "hkjc_racing.db"
    ):
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self):
        """初始化資料庫表格"""
        try:
            Base.metadata.create_all(self.engine)
            print("【成功】SQLAlchemy 資料庫表格已成功初始化！")
        except Exception as e:
            print(f"【錯誤】初始化資料庫失敗: {e}")

    def insert_dataframes(self, tables_dict: dict[str, pd.DataFrame]):
        if not tables_dict:
            return

        for table_name, df in tables_dict.items():
            if df is not None and not df.empty:
                df.to_sql(
                    table_name, con=self.engine, if_exists="replace", index=False
                )
        print("【成功】資料庫數據已覆蓋寫入！")

    def has_race_results(self) -> bool:
        """檢查 race_results 表格是否存在且有資料"""
        inspector = inspect(self.engine)
        if not inspector.has_table("race_results"):
            return False

        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM race_results")
            ).scalar()
            return result > 0

    def get_pending_horse_ids(self) -> list[str]:
        """從 race_results 表中提取所有不重複，且尚未存在於 horses 表格中的 horse_id"""
        if not self.has_race_results():
            return []

        inspector = inspect(self.engine)
        has_horses_table = inspector.has_table("horses")

        with self.engine.connect() as conn:
            if has_horses_table:
                query = text("""
                    SELECT DISTINCT r.horse_id 
                    FROM race_results r
                    LEFT JOIN horses h ON r.horse_id = h.horse_code
                    WHERE r.horse_id IS NOT NULL 
                      AND r.horse_id != '' 
                      AND h.horse_code IS NULL
                    ORDER BY r.horse_id DESC
                """)
            else:
                query = text("""
                    SELECT DISTINCT horse_id 
                    FROM race_results
                    WHERE horse_id IS NOT NULL AND horse_id != ''
                    ORDER BY horse_id DESC
                """)

            results = conn.execute(query).fetchall()
            return [row[0] for row in results if row[0]]

    # ---------------- 記憶體優化與分批讀寫增強 ----------------

    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """優化 DataFrame 記憶體佔用 (float64 -> float32, int64 -> int32)"""
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")
        return df

    def get_all_race_dates(self) -> list[str]:
        """取得資料庫中所有不重複的賽事日期 (按時間升序排序)"""
        if not self.has_race_results():
            return []

        with self.engine.connect() as conn:
            query = text("SELECT DISTINCT date FROM races ORDER BY date ASC")
            results = conn.execute(query).fetchall()
            return [row[0] for row in results if row[0]]

    def load_all_merged_race_data(self) -> pd.DataFrame:
        """一次性載入資料庫內所有賽事與馬匹數據"""
        query = text("""
            SELECT 
            r.race_id,
            r.date,
            r.venue,
            r.race_no,
            r.race_class,
            r.distance,
            r.track_condition,
            r.track_texture,
            r.track_type,
            
            res.horse_id,
            res.horse_name,
            res.placing,
            res.draw,
            res.jockey,
            res.trainer,
            res.actual_weight,
            res.declared_weight,
            res.win_odds,
            res.finish_time_sec,
            res.margin_len,
            res.rating,
            
            h.import_date,
            h.sire,

            -- 🌟 關鍵：將分段資料轉置成單欄（以 1200m ~ 2000m 常見的 3~4 個分段為例）
            MAX(CASE WHEN sec.section_no = 1 THEN sec.sectional_time_sec END) AS sec1_time,
            MAX(CASE WHEN sec.section_no = 2 THEN sec.sectional_time_sec END) AS sec2_time,
            MAX(CASE WHEN sec.section_no = 3 THEN sec.sectional_time_sec END) AS sec3_time,
            MAX(CASE WHEN sec.section_no = 4 THEN sec.sectional_time_sec END) AS sec4_time,
            MAX(CASE WHEN sec.section_no = 5 THEN sec.sectional_time_sec END) AS sec5_time,
            MAX(CASE WHEN sec.section_no = 6 THEN sec.sectional_time_sec END) AS sec6_time,
            
            -- 🌟 末腳（最後一段）時間與位置
            MAX(CASE WHEN sec.section_no = 1 THEN sec.position END) AS pos_sec1,
            MAX(CASE WHEN sec.section_no = 2 THEN sec.position END) AS pos_sec2,
            MAX(CASE WHEN sec.section_no = 3 THEN sec.position END) AS pos_sec3,
            MAX(CASE WHEN sec.section_no = 4 THEN sec.position END) AS pos_sec4,
            MAX(CASE WHEN sec.section_no = 5 THEN sec.position END) AS pos_sec5,
            MAX(CASE WHEN sec.section_no = 6 THEN sec.position END) AS pos_sec6

        FROM race_results res
        INNER JOIN races r ON res.race_id = r.race_id
        LEFT JOIN horses h ON res.horse_id = h.horse_code
        LEFT JOIN race_sectionals sec ON res.race_id = sec.race_id AND res.horse_id = sec.horse_id

        -- 🌟 必須加上 GROUP BY 確保一匹馬在該場比賽只有 1 列！
        GROUP BY 
            r.race_id, r.date, r.venue, r.race_no, r.race_class, r.distance, 
            r.track_condition, r.track_texture, r.track_type, res.horse_id, 
            res.horse_name, res.placing, res.draw, res.jockey, res.trainer, 
            res.actual_weight, res.declared_weight, res.win_odds, res.finish_time_sec, 
            res.margin_len, res.rating, h.import_date, h.sire

        ORDER BY r.date ASC, r.race_id ASC, res.horse_id ASC""")

        with self.engine.connect() as conn:
            df = pd.read_sql_query(query, conn)

        return self.optimize_memory(df)

    def save_feature_matrix(
        self,
        df: pd.DataFrame,
        table_name: str = "feature_matrix",
        if_exists: str = "append",
    ):
        """將特徵矩陣分批寫入資料庫"""
        if df is None or df.empty:
            return

        df = self.optimize_memory(df)

        with self.engine.begin() as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                index=False,
                dtype={"race_id": String(), "horse_id": String()},
            )
            
    def load_feature_result(self) -> pd.DataFrame:
        query = """
        SELECT 
            f.*,
            r.placing,
            r.win_odds,
            CASE WHEN r.placing = 1 THEN 1 ELSE 0 END AS is_win,
            CASE WHEN r.placing BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS is_top3
        FROM feature_matrix f
        INNER JOIN race_results r 
          ON f.race_id = r.race_id 
         AND f.horse_id = r.horse_id
        ORDER BY f.race_id ASC;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                query,
                conn
            )
            return df
```

---

### File: `database\models.py`

```py
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Race(Base):
    __tablename__ = "races"

    race_id = Column(String(50), primary_key=True)
    date = Column(String(20), nullable=False)
    venue = Column(String(10), nullable=False)
    race_no = Column(Integer, nullable=False)
    race_class = Column(String(10))
    distance = Column(Integer)
    track_condition = Column(String(20))
    track_texture = Column(String(20))
    track_type = Column(String(10))

    results = relationship(
        "RaceResult", back_populates="race", cascade="all, delete-orphan"
    )
    sectionals = relationship(
        "RaceSectional", back_populates="race", cascade="all, delete-orphan"
    )


class RaceResult(Base):
    __tablename__ = "race_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    horse_id = Column(String(10))
    horse_name = Column(String(50), nullable=False)
    placing = Column(Integer)
    draw = Column(Integer)
    jockey = Column(String(50))
    trainer = Column(String(50))
    actual_weight = Column(Float)
    declared_weight = Column(Float)
    win_odds = Column(Float)
    finish_time_sec = Column(Float)
    margin_len = Column(Float)
    rating = Column(Integer)

    race = relationship("Race", back_populates="results")


class RaceSectional(Base):
    __tablename__ = "race_sectionals"

    sec_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    horse_id = Column(String(10))
    horse_name = Column(String(50), nullable=False)
    section_no = Column(Integer, nullable=False)
    position = Column(Integer)
    sectional_time_sec = Column(Float)
    margin_behind = Column(String(20))

    race = relationship("Race", back_populates="sectionals")


class Horse(Base):
    __tablename__ = "horses"

    horse_code = Column(String(20), primary_key=True)
    origin = Column(String(50), nullable=True)
    age = Column(Integer, nullable=True)
    color = Column(String(20), nullable=True)
    sex = Column(String(20), nullable=True)
    import_type = Column(String(50), nullable=True)
    season_stakes = Column(Float, default=0.0)
    total_stakes = Column(Float, default=0.0)
    wins = Column(Integer, default=0)
    seconds = Column(Integer, default=0)
    thirds = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    recent_10_races_count = Column(Integer, nullable=True)
    current_location = Column(String(50), nullable=True)
    location_arrival_date = Column(Date, nullable=True)
    import_date = Column(Date, nullable=True)
    trainer = Column(String(50), index=True, nullable=True)
    owner = Column(String(100), nullable=True)
    current_rating = Column(Integer, index=True, nullable=True)
    season_start_rating = Column(Integer, nullable=True)
    sire = Column(String(100), index=True, nullable=True)
    dam = Column(String(100), nullable=True)
    damsire = Column(String(100), nullable=True)
```

---

### File: `features\__init__.py`

```py

```

---

### File: `features\base_target.py`

```py
import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
from features.utils.leak_guard import LeakageGuard


class BaseTargetBuilder:
    """專門負責建立特徵工程的核心基底（Skeleton DataFrame）與目標變數（Targets）。

    與 Schema 對齊：
    - 主表: races (race_id, date, venue, distance, track_type, track_condition, track_texture, race_class)
    - 賽果: race_results (race_id, horse_id, placing, draw, jockey, trainer, actual_weight, declared_weight, win_odds, finish_time_sec)
    - 馬匹: horses (horse_code, import_date, sire)
    """

    PRIMARY_KEYS = ["race_id", "horse_id"]
    TIME_KEYS = ["race_date"]

    # 保留特徵工程所需的最基本 Context 欄位
    CONTEXT_COLS = [
        "race_date",
        "race_id",
        "horse_id",
        "horse_name",
        "jockey",
        "trainer",
        "venue",
        "race_no",
        "race_class",
        "distance",
        "track_condition",
        "track_texture",
        "track_type",
        "draw",
        "actual_weight",
        "declared_weight",
        "win_odds",
        "finish_time_sec",
        "margin_len",
        "rating",
        "import_date",  # 🌟 補齊 SQL 選取的馬匹抵港日期
        "sire",         # 🌟 補齊 SQL 選取的父系/種馬
    ]

    @classmethod
    def build_from_dataframe(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """從傳入的 Raw DataFrame 建立骨架。"""
        df = df_raw.copy()

        # 1. 重命名欄位 (對齊系統標準命名)
        if "date" in df.columns and "race_date" not in df.columns:
            df = df.rename(columns={"date": "race_date"})

        # 2. 數據清洗與名次處理
        df = cls._sanitize_data(df)

        # 3. 按時間嚴格排序 (防止 Data Leakage)
        df = cls._sort_by_time(df)

        # 4. 嚴格防範 import_date 未來時間洩漏 (Temporal Leakage Guard)
        df = cls._apply_temporal_guard(df)

        # 5. 生成目標變數 (Targets)
        df = cls._create_targets(df)

        # 6. 保留 Context + Targets
        target_cols = ["target_win", "target_place", "target_rank_score"]
        keep_cols = [
            c for c in cls.CONTEXT_COLS if c in df.columns
        ] + target_cols
        df_base = df[keep_cols].copy()

        # 7. 資料品質檢查
        LeakageGuard.validate_feature_dataframe(
            df_base, required_keys=cls.PRIMARY_KEYS
        )

        return df_base

    @classmethod
    def build_from_sqlite(
        cls, db_path: str, query: Optional[str] = None
    ) -> pd.DataFrame:
        """根據傳入的 DB Schema 從 SQLite 資料庫做 JOIN 並載入數據。"""
        if query is None:
            query = """
                SELECT 
                    r.date AS race_date,
                    r.race_id,
                    r.venue,
                    r.race_no,
                    r.race_class,
                    r.distance,
                    r.track_condition,
                    r.track_texture,
                    r.track_type,
                    res.horse_id,
                    res.horse_name,
                    res.placing,
                    res.draw,
                    res.jockey,
                    res.trainer,
                    res.actual_weight,
                    res.declared_weight,
                    res.win_odds,
                    res.finish_time_sec,
                    res.margin_len,
                    res.rating,
                    h.import_date,
                    h.sire
                FROM race_results res
                INNER JOIN races r ON res.race_id = r.race_id
                LEFT JOIN horses h ON res.horse_id = h.horse_code
            """
        with sqlite3.connect(db_path) as conn:
            df_raw = pd.read_sql_query(query, conn)

        return cls.build_from_dataframe(df_raw)

    @staticmethod
    def _sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
        """清洗退跑、無效名次與 Null 值。"""
        df["placing_str"] = df["placing"].astype(str).str.upper().str.strip()

        invalid_pos_keywords = [
            "WV", "SCR", "DNF", "DISQ", "FE", "PU", "UR", "NAN", "NONE"
        ]
        valid_mask = ~df["placing_str"].isin(invalid_pos_keywords) & df["placing"].notnull()
        df = df[valid_mask].copy()

        df["craft_rank"] = pd.to_numeric(df["placing"], errors="coerce")
        df = df[df["craft_rank"].notnull() & (df["craft_rank"] > 0)].copy()

        df = df[df["race_id"].notnull() & df["horse_id"].notnull()].copy()
        return df

    @staticmethod
    def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
        """嚴格按 race_date, race_id, horse_id 時間升冪排序。"""
        df["race_date"] = pd.to_datetime(df["race_date"])
        df = df.sort_values(
            by=["race_date", "race_id", "horse_id"], ascending=[True, True, True]
        ).reset_index(drop=True)
        return df

    @staticmethod
    def _apply_temporal_guard(df: pd.DataFrame) -> pd.DataFrame:
        """🔒 防洩漏關鍵：若 import_date 晚於比賽日期，將其屏蔽為 NaT，避免未來的抵港紀錄滲透到過去賽事。"""
        if "import_date" in df.columns:
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            race_dt = pd.to_datetime(df["race_date"], errors="coerce")
            
            # 若抵港時間在比賽時間之後，判定為時間異常/未到港資訊洩漏，改為 NaT
            future_mask = import_dt > race_dt
            df.loc[future_mask, "import_date"] = pd.NaT
        return df

    @staticmethod
    def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """生成三個標準標籤。"""
        df["target_win"] = (df["craft_rank"] == 1).astype(int)
        df["target_place"] = (df["craft_rank"] <= 3).astype(int)
        df["target_rank_score"] = 1.0 / df["craft_rank"]
        return df
```

---

### File: `features\feature_pipeline.py`

```py
import gc
import os
import sqlite3
from typing import List, Optional
import pandas as pd

from features.generators import load_all_generators
from features.utils import LeakageGuard


class FeaturesPipeline:
    """HKJC Quant 量化特徵工程主管道 (動態掃描 Plugin 模式 + 零拷貝 & 零碎片化極速優化)。"""

    def __init__(self, key_cols: Optional[List[str]] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]
        # 🚀 自動掃描並依優先權排序載入所有 Generator
        self.generators = load_all_generators(key_cols=self.key_cols)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("[FeaturesPipeline] 輸入的 DataFrame 為空！")

        print(
            f"🚀 [Pipeline] 開始執行特徵生成 (共自動載入 {len(self.generators)} 個 Generators, 輸入筆數: {len(df)})"
        )

        # -------------------------------------------------------------------------
        # 🔒 [防洩漏靈魂步驟 1] 保留原始 Index 順序，並強制按【時間】嚴格排序
        # -------------------------------------------------------------------------
        original_index = df.index

        if "date" in df.columns:
            # 確保按照時間序列排序，避免 Rolling/Expanding 計算時發生未來的資料洩漏
            working_df = df.sort_values(["date", "race_id", "horse_id"]).copy()
        else:
            working_df = df.copy()

        # -------------------------------------------------------------------------
        # 🛡️ [防洩漏靈魂步驟 2] 驗證時間與 Target 資料合規性
        # -------------------------------------------------------------------------
        if hasattr(LeakageGuard, "check_dataframe"):
            LeakageGuard.check_dataframe(working_df)

        collected_feature_dfs: List[pd.DataFrame] = []

        # 💡 用於追蹤已存在的特徵欄位名稱 (包含原始 df 的欄位)，排除 key_cols 避免干擾
        existing_cols = set(working_df.columns)

        for gen in self.generators:
            gen_name = gen.__class__.__name__
            print(f"  ⚡ 正在執行 Generator: {gen_name}...")

            # 執行 Generator 生成特徵
            feat_df = gen.generate(working_df)

            if feat_df is None or feat_df.empty:
                print(f"  ⚠️ [Warning] {gen_name} 未產出任何特徵，跳過。")
                continue

            # ---------------------------------------------------------------------
            # 💡 [欄位防重名與數據清理]
            # ---------------------------------------------------------------------
            # 1. 只挑出非 Key 欄位且尚未在 existing_cols 中出現過的「新特徵欄位」
            new_feature_cols = [
                col
                for col in feat_df.columns
                if col not in self.key_cols and col not in existing_cols
            ]

            if not new_feature_cols:
                # 檢查是否因為缺少必要欄位而直接 returned 特徵空殼
                gen_cols_non_key = [
                    col for col in feat_df.columns if col not in self.key_cols
                ]
                if not gen_cols_non_key:
                    print(
                        f"  ⚠️ [Warning] {gen_name} 因缺少輸入必要欄位，未生成任何新特徵。"
                    )
                else:
                    print(
                        f"  ℹ️ {gen_name} 產出的欄位 ({gen_cols_non_key}) 皆已存在於輸入數據中，跳過。"
                    )
                continue

            # 僅保留純新特徵欄位 (Key 欄位將在最後統一拼合)
            clean_feat_df = feat_df[new_feature_cols].copy()

            # 2. 自動將 float64 轉為 float32 以節省記憶體並防止碎片化
            float64_cols = clean_feat_df.select_dtypes(
                include=["float64"]
            ).columns
            if len(float64_cols) > 0:
                clean_feat_df[float64_cols] = clean_feat_df[
                    float64_cols
                ].astype("float32")

            # 3. 更新已存在的欄位集合
            existing_cols.update(new_feature_cols)

            # 4. 收集結果 DataFrame
            collected_feature_dfs.append(clean_feat_df)

            # 5. 選擇性動態將新特徵併回 working_df，供後續有依賴關係的 Generator 使用
            # (例如 PaceStrategyGenerator 依賴 RunningPositionGenerator 的產出)
            working_df = pd.concat([working_df, clean_feat_df], axis=1)

            # 手動釋放暫存記憶體
            gc.collect()

        if not collected_feature_dfs:
            raise RuntimeError(
                "[FeaturesPipeline] 没有任何 Generator 成功生成特徵！"
            )

        # -------------------------------------------------------------------------
        # 🚀 [零碎片化特徵合併、Key 欄位保留與 Index 恢復]
        # -------------------------------------------------------------------------
        print("📦 [Pipeline] 正在高效併合所有特徵矩陣 (含 Key 欄位)...")

        # 1. 提取 Key 欄位 (確保包含 self.key_cols，例如 race_id, horse_id)
        present_keys = [
            col for col in self.key_cols if col in working_df.columns
        ]
        keys_df = working_df[present_keys].copy()

        # 2. 一次性併合 Keys 與所有生成出的特徵 DataFrame
        generated_features_df = pd.concat(collected_feature_dfs, axis=1)
        final_features_df = pd.concat([keys_df, generated_features_df], axis=1)

        # 🔒 [防洩漏與對齊靈魂步驟 3] 恢復為傳入時的原始 Index 順序
        final_features_df = final_features_df.reindex(original_index)

        print(
            f"✅ [Pipeline] 特徵工程完成！總共產出 {final_features_df.shape[1]} 個欄位 (含 Keys: {present_keys})，筆數: {len(final_features_df)}"
        )

        return final_features_df
```

---

### File: `features\generators\__init__.py`

```py
import importlib
import inspect
import os
import pkgutil


def load_all_generators(key_cols: list[str] = None):
    """
    動態掃描並載入當前目錄下所有的 Generator 類別，
    自動讀取 class 內部的 EXECUTION_ORDER 進行排序後回傳實例列表。
    """
    key_cols = key_cols or ["race_id", "horse_id"]
    generator_instances = []

    pkg_dir = os.path.dirname(__file__)
    pkg_name = __name__

    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name.startswith("_") or is_pkg:
            continue

        full_module_name = f"{pkg_name}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                obj.__module__ == full_module_name
                and name.endswith("Generator")
            ):
                generator_instances.append(obj(key_cols=key_cols))

    # 依照各 Generator 類別內部的 EXECUTION_ORDER 屬性進行排序 (未定義者預設值為 500)
    generator_instances.sort(
        key=lambda gen: getattr(gen, "EXECUTION_ORDER", 500)
    )

    return generator_instances
```

---

### File: `features\generators\body_weight_recovery.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class BodyWeightRecoveryGenerator:
    """馬匹體重變動與體能恢復特徵生成器 (完全不含賠率)"""

    EXECUTION_ORDER = 52

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 支援多種常見的馬匹體重欄位命名
        weight_col = next(
            (c for c in ["declared_weight", "horse_weight"] if c in df.columns),
            None,
        )

        if weight_col is None or "date" not in df.columns:
            features["horse_weight_vs_hist_mean"] = 0.0
            features["horse_weight_abs_change"] = 0.0
            features["is_heavy_workload_14d"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["race_dt"] = pd.to_datetime(work_df["date"])

        # 1. 馬匹過去 3 場的平均體重 (與當前體重比較)
        hist_weight_mean = BayesianSmoother.calc_rolling_stat(
            work_df,
            group_cols="horse_id",
            value_col=weight_col,
            window=3,
            stat="mean",
        )
        prev_weight = work_df.groupby("horse_id")[weight_col].shift(1)

        features["horse_weight_vs_hist_mean"] = (
            (df[weight_col] - hist_weight_mean)
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )
        
        # 2. 上場與本場體重的絕對變化量 (過胖或減過頭皆影響勝率)
        features["horse_weight_abs_change"] = (
            (df[weight_col] - prev_weight)
            .abs()
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )

        # 3. 14 天內連續出賽的高強度密集賽程標記
        prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)
        days_rest = (work_df["race_dt"] - prev_dt).dt.days
        features["is_heavy_workload_14d"] = (
            (days_rest <= 14).astype("float32").reindex(df.index).fillna(0.0)
        )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\class_performance.py`

```py
import numpy as np
import pandas as pd
import re
from features.utils import BayesianSmoother, LeakageGuard


class ClassPerformanceGenerator:
    """馬匹班次表現與升降班適應力特徵生成器 (僅限 Class 1-5)"""

    EXECUTION_ORDER = 15

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    @staticmethod
    def _parse_class_num(race_class_series: pd.Series) -> pd.Series:
        """只解析 Class 1 到 5，其他班次 (Group/Griffin/特殊賽事) 一律回傳 np.nan"""

        def parse_val(val):
            if pd.isna(val):
                return np.nan
            s = str(val).upper().strip()

            # 抓取班次數字 (例如: "CLASS 3" -> 3)
            match = re.search(r"(\d+)", s)
            if match:
                class_num = float(match.group(1))
                # 嚴格限制在 1 至 5 班之間
                if 1.0 <= class_num <= 5.0:
                    return class_num

            # 非 1-5 班賽事 (如 Group 1, Griffin 等) 返回 NaN
            return np.nan

        return race_class_series.apply(parse_val)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "race_class" not in df.columns:
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["class_num"] = self._parse_class_num(work_df["race_class"])
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        # 計算同班歷史勝率與上名率 (非 1-5 班的紀錄會自動忽略)
        class_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["horse_id", "class_num"],
            target_col="is_win",
            prior_alpha=2.0,
            baseline_rate=0.08,
        )
        features["horse_class_win_rate"] = class_win_rate.reindex(
            df.index
        ).astype("float32")

        class_top3_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["horse_id", "class_num"],
            target_col="is_top3",
            prior_alpha=2.0,
            baseline_rate=0.24,
        )
        features["horse_class_top3_rate"] = class_top3_rate.reindex(
            df.index
        ).astype("float32")

        # 升降班動態標籤計算
        prev_class = work_df.groupby("horse_id")["class_num"].shift(1)
        work_df["class_diff"] = work_df["class_num"] - prev_class

        # 只有兩場賽事都在 1-5 班內時，才會計算升降班
        work_df["is_class_up"] = (work_df["class_diff"] < 0).astype("float32")
        work_df["is_class_down"] = (work_df["class_diff"] > 0).astype("float32")

        features["is_class_up"] = (
            work_df["is_class_up"].reindex(df.index).fillna(0.0).astype("float32")
        )
        features["is_class_down"] = (
            work_df["is_class_down"]
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )

        # 歷史升降班次數統計
        shifted_up = work_df.groupby("horse_id")["is_class_up"].shift(1)
        shifted_down = work_df.groupby("horse_id")["is_class_down"].shift(1)

        cum_class_up_count = (
            shifted_up.groupby(work_df["horse_id"]).cumsum().fillna(0.0)
        )
        cum_class_down_count = (
            shifted_down.groupby(work_df["horse_id"]).cumsum().fillna(0.0)
        )

        features["horse_hist_class_up_count"] = (
            cum_class_up_count.reindex(df.index).astype("float32")
        )
        features["horse_hist_class_down_count"] = (
            cum_class_down_count.reindex(df.index).astype("float32")
        )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\context_relative.py`

```py
import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class ContextRelativeGenerator:
    """生成場次內相對特徵 (Context Relative Features)。
    
    將絕對數值（如負重、排位、賠率）轉換為在該場比賽（race_id）中的相對名次、Z-Score 或與平均值之差。
    """

    EXECUTION_ORDER = 120

    def __init__(self, key_cols: list[str] = None):
        # 🌟 修正：接收 key_cols 參數，與其他 Generator 保持一致
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "race_id" not in df.columns:
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        # 1. 負重相對特徵 (Actual Weight Relative)
        if "actual_weight" in df.columns:
            df["actual_weight_num"] = pd.to_numeric(df["actual_weight"], errors="coerce")
            race_mean_weight = df.groupby("race_id")["actual_weight_num"].transform("mean")
            features["weight_diff_from_race_avg"] = (
                (df["actual_weight_num"] - race_mean_weight).fillna(0.0).astype("float32")
            )
        else:
            features["weight_diff_from_race_avg"] = 0.0

        # 2. 獨贏賠率相對特徵 (Win Odds Relative & Rank within Race)
        if "win_odds" in df.columns:
            df["win_odds_num"] = pd.to_numeric(df["win_odds"], errors="coerce")
            
            # 賠率在同場比賽中的名次 (1 代表大熱門)
            features["odds_rank_in_race"] = (
                df.groupby("race_id")["win_odds_num"]
                .rank(method="min", ascending=True)
                .fillna(99.0)
                .astype("float32")
            )

            # 隱含勝率 (Implied Probability) 及其同場佔比
            implied_prob = 1.0 / df["win_odds_num"].replace(0, np.nan)
            total_prob = implied_prob.groupby(df["race_id"]).transform("sum")
            features["implied_prob_share"] = (
                (implied_prob / total_prob).fillna(0.0).astype("float32")
            )
        else:
            features["odds_rank_in_race"] = 99.0
            features["implied_prob_share"] = 0.0

        # 3. 檔位相對特徵 (Draw Z-score)
        if "draw" in df.columns:
            df["draw_num"] = pd.to_numeric(df["draw"], errors="coerce")
            race_draw_std = df.groupby("race_id")["draw_num"].transform("std").replace(0, np.nan)
            race_draw_mean = df.groupby("race_id")["draw_num"].transform("mean")
            
            features["draw_zscore_in_race"] = (
                ((df["draw_num"] - race_draw_mean) / race_draw_std).fillna(0.0).astype("float32")
            )
        else:
            features["draw_zscore_in_race"] = 0.0

        # 數據品質與安全檢查
        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\horse_profile.py`

```py
import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class HorseProfileGenerator:
    """生成馬匹服役資歷與烙印年資特徵 (Data Source: horses / race_results)"""

    EXECUTION_ORDER = 20

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "race_date" if "race_date" in df.columns else ("date" if "date" in df.columns else None)

        # 🔒 安全計算：馬匹在「該場賽事當下」在香港服役的實際年數
        if date_col and "import_date" in df.columns:
            race_dt = pd.to_datetime(df[date_col], errors="coerce")
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            
            # 計算比賽當天距離抵港日期的天數（若抵港日在比賽日之後則會被遮蔽為 NaN）
            days_in_hk = (race_dt - import_dt).dt.days
            
            # 僅保留比賽當下已抵港的合法紀錄 (>=0 天)
            valid_days = days_in_hk.where(days_in_hk >= 0, np.nan)
            features["est_years_in_hk"] = (valid_days / 365.25).clip(lower=0.0, upper=10.0).fillna(0.0).astype("float32")
        else:
            features["est_years_in_hk"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\horse_rolling.py`

```py
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class HorseRollingGenerator:

    """計算馬匹歷史賽事紀錄之滾動 (Rolling) 統計特徵（已嚴格防堵 Data Leakage）。"""

    EXECUTION_ORDER = 50

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 確保嚴格按馬匹與時間排序
        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        # 2. 關鍵防洩漏步驟：先對基礎欄位做 .shift(1)，將當場比賽結果隔離
        work_df["shifted_placing"] = work_df.groupby("horse_id")["placing"].shift(1)
        work_df["shifted_is_win"] = work_df.groupby("horse_id")["is_win"].shift(1)
        work_df["shifted_is_top3"] = work_df.groupby("horse_id")["is_top3"].shift(1)
        
        if "actual_weight" in work_df.columns:
            work_df["shifted_actual_weight"] = work_df.groupby("horse_id")["actual_weight"].shift(1)

        windows = [3, 5, 10]
        for w in windows:
            # 使用 shift 後的欄位計算 Rolling，徹底阻斷數據洩漏
            features[f"horse_rolling_pos_mean_{w}"] = (
                BayesianSmoother.calc_rolling_stat(
                    work_df,
                    group_cols="horse_id",
                    value_col="shifted_placing",
                    window_size=w,
                    stat_type="mean",
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_pos_std_{w}"] = (
                BayesianSmoother.calc_rolling_stat(
                    work_df,
                    group_cols="horse_id",
                    value_col="shifted_placing",
                    window_size=w,
                    stat_type="std",
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_win_rate_{w}"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="horse_id",
                    target_col="shifted_is_win",
                    window_size=w,
                    prior_alpha=3.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_top3_rate_{w}"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="horse_id",
                    target_col="shifted_is_top3",
                    window_size=w,
                    prior_alpha=3.0,
                    baseline_rate=0.24,
                )
                .reindex(df.index)
                .astype("float32")
            )

            if "actual_weight" in work_df.columns:
                features[f"horse_rolling_weight_mean_{w}"] = (
                    BayesianSmoother.calc_rolling_stat(
                        work_df,
                        group_cols="horse_id",
                        value_col="shifted_actual_weight",
                        window_size=w,
                        stat_type="mean",
                    )
                    .reindex(df.index)
                    .astype("float32")
                )

        if "date" in work_df.columns:
            work_df["race_dt"] = pd.to_datetime(work_df["date"])
            prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)
            days = (work_df["race_dt"] - prev_dt).dt.days.fillna(999)
            features["days_since_last_race"] = (
                days.reindex(df.index).astype("float32")
            )

        if "horse_weight" in work_df.columns:
            prev_horse_w = work_df.groupby("horse_id")["horse_weight"].shift(1)
            w_change = work_df["horse_weight"] - prev_horse_w
            features["horse_weight_change"] = (
                w_change.reindex(df.index).fillna(0.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\human_sire.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class HumanSireGenerator:
    """騎師/練馬師/種馬 (Sire) 歷史績效特徵生成器"""

    EXECUTION_ORDER = 60

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "date" if "date" in df.columns else ("race_date" if "race_date" in df.columns else None)
        if date_col:
            work_df = df.sort_values([date_col, "race_id", "horse_id"]).copy()
        else:
            work_df = df.copy()

        work_df["is_win"] = (work_df["placing"] == 1).astype("float32") if "placing" in work_df.columns else 0.0

        if "sire" in work_df.columns:
            sire_smooth_win = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols="sire", target_col="is_win", prior_alpha=15.0, baseline_rate=0.08
            )
            features["sire_global_win_rate"] = sire_smooth_win.reindex(df.index).astype("float32")

        if "jockey" in work_df.columns:
            jockey_rolling_win = BayesianSmoother.calc_rolling_smooth_rate(
                work_df, group_cols="jockey", target_col="is_win", window=10, prior_alpha=5.0, baseline_rate=0.08
            )
            features["jockey_rolling_win_rate_10"] = jockey_rolling_win.reindex(df.index).astype("float32")

        if "trainer" in work_df.columns:
            trainer_rolling_win = BayesianSmoother.calc_rolling_smooth_rate(
                work_df, group_cols="trainer", target_col="is_win", window=20, prior_alpha=10.0, baseline_rate=0.08
            )
            features["trainer_rolling_win_rate_20"] = trainer_rolling_win.reindex(df.index).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\injury_rest.py`

```py
import pandas as pd
from features.utils import LeakageGuard


class InjuryRestGenerator:
    """生成參賽節奏、久休復出與抵港參賽間隔特徵。"""

    EXECUTION_ORDER = 110

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        
        date_col = "race_date" if "race_date" in df.columns else ("date" if "date" in df.columns else None)
        
        if date_col is None:
            features["days_since_last_race"] = 999.0
            features["is_layoff_60d"] = 0.0
            features["is_layoff_90d"] = 0.0
            features["days_since_import"] = 999.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", date_col]).copy()

        # 1. 計算距離上場賽事天數 (嚴格按時間)
        work_df["race_dt"] = pd.to_datetime(work_df[date_col])
        prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)

        days_diff = (work_df["race_dt"] - prev_dt).dt.days
        features["days_since_last_race"] = (
            days_diff.reindex(df.index).fillna(999.0).astype("float32")
        )

        features["is_layoff_60d"] = (
            features["days_since_last_race"] >= 60.0
        ).astype("float32")
        features["is_layoff_90d"] = (
            features["days_since_last_race"] >= 90.0
        ).astype("float32")

        # 2. 🔒 安全計算抵港天數（嚴格濾除未來時間泄露）
        if "import_date" in df.columns:
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            race_dt = pd.to_datetime(df[date_col], errors="coerce")
            
            diff_days = (race_dt - import_dt).dt.days
            # 若抵港日晚於比賽日 (diff_days < 0)，視為未到港數據，填為 999.0
            valid_diff = diff_days.where(diff_days >= 0, 999.0)
            
            features["days_since_import"] = (
                valid_diff.reindex(df.index).fillna(999.0).astype("float32")
            )
        else:
            features["days_since_import"] = 999.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\interaction.py`

```py
import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class InteractionGenerator:

    """生成交叉權重與高級交互特徵。"""

    EXECUTION_ORDER = 999  # 交叉特徵 Generator 必須排在最後面！

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 負重比率
        hw_col = next(
            (
                c
                for c in ["horse_weight", "declared_weight", "rank_weight"]
                if c in df.columns
            ),
            None,
        )
        if "actual_weight" in df.columns and hw_col:
            valid_hw = df[hw_col].replace(0, np.nan)
            features["weight_to_horse_body_ratio"] = (
                df["actual_weight"] / valid_hw
            ).astype("float32")

        # 2. 人馬勝率乘積
        h_win_col = next(
            (
                c
                for c in [
                    "horse_rolling_win_rate_5",
                    "h_smoothed_rolling_5_win_rate",
                ]
                if c in df.columns
            ),
            None,
        )
        j_win_col = next(
            (
                c
                for c in [
                    "jockey_rolling_win_rate_50",
                    "j_smoothed_rolling_30_win_rate",
                ]
                if c in df.columns
            ),
            None,
        )

        if h_win_col and j_win_col:
            features["horse_jockey_win_rate_interaction"] = (
                df[h_win_col] * df[j_win_col]
            ).astype("float32")

        # 3. 賠率落差與隱含勝率
        if "win_odds" in df.columns:
            implied_prob = 1.0 / df["win_odds"].replace(0, np.nan)
            features["win_odds_inv"] = implied_prob.astype("float32")

            if h_win_col:
                features["odds_vs_history_win_rate_gap"] = (
                    implied_prob - df[h_win_col]
                ).astype("float32")

        # 4. 檔位與速度 Z-Score 交互
        if "draw" in df.columns and "h_mean_speed_z_15" in df.columns:
            features["draw_speed_interaction"] = (
                df["draw"] * df["h_mean_speed_z_15"]
            ).astype("float32")

        # 5. 評分優勢 / 負重變化與體重交互
        rating_col = (
            "rating_vs_race_avg"
            if "rating_vs_race_avg" in df.columns
            else "rating"
        )
        if rating_col in df.columns and hw_col:
            features["rating_x_rank_weight"] = (
                df[rating_col] * df[hw_col]
            ).astype("float32")

        if "weight_delta" in df.columns and hw_col:
            features["delta_x_rank"] = (df["weight_delta"] * df[hw_col]).astype(
                "float32"
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\jockey_trainer_alpha.py`

```py
import numpy as np
import pandas as pd
import re
from features.utils import BayesianSmoother, LeakageGuard


class JockeyTrainerAlphaGenerator:
    """騎練動態 Alpha 特徵生成器 (Jockey & Trainer Dynamics Alpha Generator)"""

    EXECUTION_ORDER = 65

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    @staticmethod
    def _is_class_1_to_5(race_class_series: pd.Series) -> pd.Series:
        def check_valid(val):
            if pd.isna(val):
                return False
            s = str(val).upper().strip()
            if any(kw in s for kw in ["G1", "G2", "G3", "GROUP", "HKG"]):
                return False
            match = re.search(r"(\d+)", s)
            if match:
                c = float(match.group(1))
                return 1.0 <= c <= 5.0
            return False

        return race_class_series.apply(check_valid)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        required_cols = ["jockey", "trainer", "date", "placing"]
        if not all(col in df.columns for col in required_cols):
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "race_class" in work_df.columns:
            work_df["is_target_class"] = self._is_class_1_to_5(
                work_df["race_class"]
            )
        else:
            work_df["is_target_class"] = True

        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        work_df["jt_combo"] = (
            work_df["jockey"].astype(str)
            + "_"
            + work_df["trainer"].astype(str)
        )

        jt_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jt_combo"],
            target_col="is_win",
            prior_alpha=3.0,
            baseline_rate=0.08,
        )

        jt_top3_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jt_combo"],
            target_col="is_top3",
            prior_alpha=3.0,
            baseline_rate=0.24,
        )

        jockey_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jockey"],
            target_col="is_win",
            prior_alpha=5.0,
            baseline_rate=0.08,
        )

        jt_win_alpha = jt_win_rate - jockey_win_rate

        prev_jockey = work_df.groupby("horse_id")["jockey"].shift(1)
        work_df["is_jockey_switched"] = (
            (work_df["jockey"] != prev_jockey) & (prev_jockey.notna())
        ).astype("float32")

        prev_jockey_win_rate = (
            work_df.groupby("horse_id")["jockey"]
            .shift(1)
            .map(jockey_win_rate)
        )
        work_df["jockey_upgrade_alpha"] = np.where(
            work_df["is_jockey_switched"] == 1.0,
            jockey_win_rate - prev_jockey_win_rate,
            0.0,
        )

        target_mask = work_df["is_target_class"]

        features["alpha_jt_combo_win_rate"] = np.where(
            target_mask, jt_win_rate, np.nan
        )
        features["alpha_jt_combo_top3_rate"] = np.where(
            target_mask, jt_top3_rate, np.nan
        )
        features["alpha_jt_synergy_alpha"] = np.where(
            target_mask, jt_win_alpha, np.nan
        )
        features["alpha_is_jockey_switched"] = np.where(
            target_mask, work_df["is_jockey_switched"], np.nan
        )
        features["alpha_jockey_upgrade_alpha"] = np.where(
            target_mask, work_df["jockey_upgrade_alpha"], np.nan
        )

        features = features.reindex(df.index)

        for col in features.columns:
            if col not in self.key_cols:
                features[col] = features[col].astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)

        return features
```

---

### File: `features\generators\jockey_trainer_synergy.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class JockeyTrainerSynergyGenerator:
    """騎練長期合作與人馬專屬勝率特徵生成器 (非賠率導向)"""

    EXECUTION_ORDER = 66

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        required_cols = ["jockey", "trainer", "placing"]
        if not all(col in df.columns for col in required_cols):
            features["jt_combo_win_rate_smooth"] = 0.0
            features["horse_jockey_combo_win_rate"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy() if "date" in df.columns else df.copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["jt_combo"] = work_df["jockey"].astype(str) + "_" + work_df["trainer"].astype(str)
        work_df["hj_combo"] = work_df["horse_id"].astype(str) + "_" + work_df["jockey"].astype(str)

        # 1. 騎練組合 (Jockey + Trainer) 歷史貝氏平滑勝率
        jt_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols="jt_combo",
            target_col="is_win",
            prior_alpha=3.0,
            baseline_rate=0.08,
        )
        features["jt_combo_win_rate_smooth"] = (
            jt_win_rate.reindex(df.index).astype("float32")
        )

        # 2. 人馬專屬組合 (Horse + Jockey) 歷史勝率 (如「潘頓策騎該馬」的表現)
        hj_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols="hj_combo",
            target_col="is_win",
            prior_alpha=1.5,
            baseline_rate=0.08,
        )
        features["horse_jockey_combo_win_rate"] = (
            hj_win_rate.reindex(df.index).astype("float32")
        )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\jt_recent_form.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class JTRecentFormGenerator:
    """騎師、練馬師及騎練組合 (J/T/JT) 近期狀態 (Recent Form) 特徵生成器"""

    EXECUTION_ORDER = 68

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        required_cols = ["date", "jockey", "trainer", "placing"]
        if not all(col in df.columns for col in required_cols):
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")
        work_df["jt_combo"] = work_df["jockey"].astype(str) + "_" + work_df["trainer"].astype(str)

        # 1. 騎師近 5 / 10 場滾動勝率
        rolling_j_win_5 = BayesianSmoother.calc_rolling_smooth_rate(
            work_df, group_cols="jockey", target_col="is_win", window_size=5, prior_alpha=2.0, baseline_rate=0.08
        )
        features["jockey_recent_win_rate_5"] = rolling_j_win_5.reindex(df.index).astype("float32")

        # 2. 騎練組合近 5 場滾動上名率
        rolling_jt_top3_5 = BayesianSmoother.calc_rolling_smooth_rate(
            work_df, group_cols="jt_combo", target_col="is_top3", window_size=5, prior_alpha=2.0, baseline_rate=0.24
        )
        features["jt_combo_recent_top3_rate_5"] = rolling_jt_top3_5.reindex(df.index).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\odds_market.py`

```py
import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class OddsMarketGenerator:

    """生成獨贏賠率、市場隱含勝率與熱門指標特徵。"""

    EXECUTION_ORDER = 100

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "win_odds" in df.columns:
            valid_odds = df["win_odds"].replace(0, np.nan)

            features["odds_implied_prob"] = (1.0 / valid_odds).astype(
                "float32"
            )
            features["is_market_favorite"] = (df["win_odds"] <= 3.0).astype(
                "float32"
            )

            odds_mean = df.groupby("race_id")["win_odds"].transform("mean")
            odds_std = df.groupby("race_id")["win_odds"].transform("std")
            features["odds_race_zscore"] = (
                (df["win_odds"] - odds_mean) / (odds_std + 1e-6)
            ).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\pace_strategy.py`

```py
import pandas as pd
import numpy as np
from features.utils import BayesianSmoother, LeakageGuard


class PaceStrategyGenerator:
    """跑法與賽事步速競爭特徵生成器"""

    EXECUTION_ORDER = 40

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        pos_series = None
        for col in ["pos_sec1", "running_position_avg", "position_1"]:
            if col in df.columns:
                pos_series = pd.to_numeric(df[col], errors="coerce")
                break

        if pos_series is None:
            features["is_front_runner"] = 0.0
            features["race_front_runner_count"] = 0.0
            features["horse_avg_sec1_pos_3"] = np.nan
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy() if "date" in df.columns else df.copy()
        work_df["_pos_clean"] = pos_series.reindex(work_df.index)
        
        rolling_pos = BayesianSmoother.calc_rolling_stat(
            work_df,
            group_cols="horse_id",
            value_col="_pos_clean",
            window_size=3,
            stat_type="mean"
        )
        features["horse_avg_sec1_pos_3"] = rolling_pos.reindex(df.index).astype("float32")

        is_front = (features["horse_avg_sec1_pos_3"].fillna(pos_series) <= 3.5).astype("float32")
        features["is_front_runner"] = is_front.reindex(df.index).astype("float32")

        df_temp = pd.DataFrame({"race_id": df["race_id"], "is_front": features["is_front_runner"]}, index=df.index)
        features["race_front_runner_count"] = (
            df_temp.groupby("race_id")["is_front"]
            .transform("sum")
            .astype("float32")
        )
        features["is_front_runner_race_front_runner_count_interaction"] = (features["is_front_runner"] * features["race_front_runner_count"]).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\rating_class.py`

```py
import pandas as pd
import numpy as np
from features.utils import LeakageGuard


class RatingClassGenerator:
    """生成馬匹班次升降 (Class Change) 特徵。"""

    EXECUTION_ORDER = 10

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        
        if "date" not in df.columns or "horse_id" not in df.columns:
            features["class_change"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()

        # 班次變動 (Class Change)
        if "race_class" in work_df.columns:
            class_num = (
                work_df["race_class"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

            work_df["_class_num"] = class_num
            prev_class = work_df.groupby("horse_id")["_class_num"].shift(1)

            class_change = work_df["_class_num"] - prev_class
            features["class_change"] = class_change.reindex(df.index).fillna(0.0).astype("float32")
        else:
            features["class_change"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\ratinn_momentum.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, RaceScaler


class RatingMomentumGenerator:
    """馬匹評分趨勢與同場評分優勢生成器 (純硬實力預測)"""

    EXECUTION_ORDER = 12

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "rating" not in df.columns:
            features["rating_diff_from_race_mean"] = 0.0
            features["rating_momentum_3"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        # 1. 同場賽事相對評分優勢 (與該場平均評分差額 & Z-Score)
        features["rating_diff_from_race_mean"] = RaceScaler.race_diff_from_mean(
            df, race_col="race_id", value_col="rating"
        ).astype("float32")

        features["rating_race_z"] = RaceScaler.race_z_score(
            df, race_col="race_id", value_col="rating"
        ).astype("float32")

        # 2. 評分動態上升/下降趨勢 (最近一次 rating vs 3場前 rating)
        if "date" in df.columns and "horse_id" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            prev_rating_1 = work_df.groupby("horse_id")["rating"].shift(1)
            prev_rating_3 = work_df.groupby("horse_id")["rating"].shift(3)

            # 近 3 場評分變化量
            rating_momentum = prev_rating_1 - prev_rating_3
            features["rating_momentum_3"] = (
                rating_momentum.reindex(df.index).fillna(0.0).astype("float32")
            )
        else:
            features["rating_momentum_3"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\sectional_brust.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, SpeedTimeCalculator


class SectionalBurstGenerator:
    """末腳衝刺爆發力與速度比率特徵生成器"""

    EXECUTION_ORDER = 32

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 檢查是否有末段分段時間
        sec_cols = [
            c for c in ["sec1_time", "sec2_time", "sec3_time", "sec4_time", "sec5_time", "sec6_time"]
            if c in df.columns
        ]

        if not sec_cols or "finish_time_sec" not in df.columns or "distance" not in df.columns:
            features["burst_ratio_last_sec"] = 0.0
            features["horse_rolling_burst_ratio_3"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        # 全場平均速度 (m/s)
        overall_speed = SpeedTimeCalculator.calc_speed_mps(
            df["distance"], df["finish_time_sec"]
        )

        # 取最後一段 400m 時間
        last_sec_time = df[sec_cols].ffill(axis=1).iloc[:, -1]
        last_sec_speed = SpeedTimeCalculator.calc_speed_mps(
            pd.Series(400.0, index=df.index), last_sec_time
        )

        # 1. 爆發力指標 (末段速度 / 全場平均速度) -> >1.0 代表末段加速能力強
        burst_ratio = last_sec_speed / (overall_speed + 1e-6)
        features["burst_ratio_last_sec"] = burst_ratio.fillna(0.0).astype("float32")

        # 2. 歷史近 3 場的平均末腳爆發比率 (Rolling 滾動計算，嚴格防洩漏)
        if "date" in df.columns and "horse_id" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            work_df["_burst_ratio"] = features.loc[work_df.index, "burst_ratio_last_sec"]

            rolling_burst = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="_burst_ratio",
                window=3,
                stat="mean",
            )
            features["horse_rolling_burst_ratio_3"] = (
                rolling_burst.reindex(df.index).fillna(0.0).astype("float32")
            )
        else:
            features["horse_rolling_burst_ratio_3"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\sectional_speed.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, SpeedTimeCalculator


class SectionalSpeedGenerator:
    """生成分段時間、末腳爆發力與走位卡位能力特徵 (適用於 SQL 轉置欄位結構 - 防洩漏與崩潰修正版)。"""

    EXECUTION_ORDER = 30

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 計算全場平均速度 (m/s)
        if "distance" in df.columns and "finish_time_sec" in df.columns:
            features["speed_mps_overall"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    df["distance"], df["finish_time_sec"]
                ).astype("float32")
            )

        # 2. 動態提取「最後一段分段時間 (sectional_time_last)」
        sec_cols = [
            c for c in ["sec1_time", "sec2_time", "sec3_time", "sec4_time", "sec5_time", "sec6_time"]
            if c in df.columns
        ]
        
        if sec_cols:
            # 安全防護：使用 ffill 提取最末有效分段時間
            sec_df = df[sec_cols].ffill(axis=1)
            if not sec_df.empty and sec_df.shape[1] > 0:
                features["sectional_time_last"] = sec_df.iloc[:, -1]

        # 3. 計算末腳衝刺速度 (m/s)
        if "sectional_time_last" in features.columns:
            features["speed_mps_last_sectional"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    pd.Series(400.0, index=df.index), features["sectional_time_last"]
                ).astype("float32")
            )

        # 4. 計算走位變化/衝刺追趕能力 (Position Gain)
        pos_cols = [
            c for c in ["pos_sec1", "pos_sec2", "pos_sec3", "pos_sec4", "pos_sec5", "pos_sec6"]
            if c in df.columns
        ]
        
        if len(pos_cols) >= 2:
            pos_df = df[pos_cols]
            first_pos = pos_df.bfill(axis=1).iloc[:, 0]
            last_pos = pos_df.ffill(axis=1).iloc[:, -1]
            features["position_gain_first_to_last"] = (first_pos - last_pos).astype("float32")

        # 5. 計算馬匹歷史近 3 場末腳平均速度 (Rolling Mean - 🔒 防洩漏與索引對齊修復)
        if "speed_mps_last_sectional" in features.columns:
            work_df = df.copy()
            work_df["speed_mps_last_sectional"] = features["speed_mps_last_sectional"]
            
            # 確保按照時間順序排序以進行歷史滾動
            if "date" in work_df.columns:
                work_df = work_df.sort_values(["horse_id", "date"])

            # 呼叫 BayesianSmoother (內部需包含 shift(1) 防止 Leakage)
            rolling_speed = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="speed_mps_last_sectional",
                window_size=3,
                stat_type="mean",
            )
            
            # 🔒 關鍵修復：使用 reindex 安全地按原始 df.index 對齊，避免 .loc 找不到索引或轉型失敗
            if isinstance(rolling_speed, pd.Series):
                features["horse_rolling_last_sec_speed_mean_3"] = (
                    rolling_speed.reindex(df.index).fillna(0.0).astype("float32")
                )
            else:
                features["horse_rolling_last_sec_speed_mean_3"] = np.float32(0.0)

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\speed_feature.py`

```py
import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, RaceScaler


class SpeedFeatureGenerator:
    """標準化速度指數與分段衝刺特徵生成器"""

    EXECUTION_ORDER = 35

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 同場標準化完賽時間 Z-Score
        if "finish_time_sec" in df.columns and "race_id" in df.columns:
            temp_df = pd.DataFrame(
                {
                    "race_id": df["race_id"],
                    "_neg_finish_time": df["finish_time_sec"] * -1.0,
                },
                index=df.index,
            )
            features["finish_time_race_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_neg_finish_time"
            ).astype("float32")

        # 2. 同場末腳衝刺速度 Z-Score
        last_sec_val = None
        if "speed_mps_last_sectional" in df.columns:
            last_sec_val = df["speed_mps_last_sectional"]
        elif "sectional_time_last" in df.columns:
            last_sec_val = df["sectional_time_last"] * -1.0

        if last_sec_val is not None and "race_id" in df.columns:
            temp_df = pd.DataFrame(
                {"race_id": df["race_id"], "_last_sec_val": last_sec_val},
                index=df.index,
            )
            features["last_400m_speed_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_last_sec_val"
            ).astype("float32")

        # 3. 同場早段搶放體力消耗 Z-Score
        if "sec1_time" in df.columns and "race_id" in df.columns:
            sec1_num = pd.to_numeric(df["sec1_time"], errors="coerce")
            temp_df = pd.DataFrame(
                {"race_id": df["race_id"], "_neg_sec1": sec1_num * -1.0},
                index=df.index,
            )
            features["early_pace_expenditure_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_neg_sec1"
            ).astype("float32")

        # 4. 馬匹歷史近 5 場 Speed Z-Score 滾動平均
        if "finish_time_race_z" in features.columns and "date" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            work_df["finish_time_race_z"] = features.loc[
                work_df.index, "finish_time_race_z"
            ]

            rolling_speed_z = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="finish_time_race_z",
                window=5,
                stat="mean",
            )
            features["horse_rolling_speed_z_mean_5"] = (
                rolling_speed_z.reindex(df.index).fillna(0.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\synergy_fitness.py`

```py
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class SynergyFitnessGenerator:

    """生成人馬默契與更換騎師特徵 (全面向量化優化)。"""

    EXECUTION_ORDER = 70

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "jockey" in work_df.columns:
            prev_jockey = work_df.groupby("horse_id")["jockey"].shift(1)
            is_changed = (work_df["jockey"] != prev_jockey).astype("float32")
            features["is_jockey_changed"] = (
                is_changed.reindex(df.index).fillna(0.0).astype("float32")
            )

            work_df["horse_jockey_pair"] = (
                work_df["horse_id"].astype(str)
                + "_"
                + work_df["jockey"].astype(str)
            )
            work_df["is_win"] = (work_df["placing"] == 1).astype("float32")

            pair_counts = work_df.groupby("horse_jockey_pair").cumcount()
            features["pair_ride_count"] = (
                pair_counts.reindex(df.index).astype("float32")
            )

            pair_win_rate = BayesianSmoother.calc_global_smooth_rate(
                work_df,
                group_cols="horse_jockey_pair",
                target_col="is_win",
                prior_alpha=2.0,
                baseline_rate=0.08,
            )
            features["pair_win_rate"] = (
                pair_win_rate.reindex(df.index).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\generators\track_distance.py`

```py
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class TrackDistanceGenerator:
    """路程與場地歷史特徵生成器"""

    EXECUTION_ORDER = 80

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "date" if "date" in df.columns else ("race_date" if "race_date" in df.columns else None)
        if date_col is None or "placing" not in df.columns:
            features["horse_dist_win_rate"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", date_col]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")

        if "distance" in work_df.columns:
            dist_smooth = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols=["horse_id", "distance"], target_col="is_win", prior_alpha=2.0, baseline_rate=0.08
            )
            features["horse_dist_win_rate"] = dist_smooth.reindex(df.index).fillna(0.0).astype("float32")

        if "track" in work_df.columns or "track_type" in work_df.columns:
            t_col = "track" if "track" in work_df.columns else "track_type"
            track_smooth = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols=["horse_id", t_col], target_col="is_win", prior_alpha=2.0, baseline_rate=0.08
            )
            features["horse_track_win_rate"] = track_smooth.reindex(df.index).fillna(0.0).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features
```

---

### File: `features\utils\__init__.py`

```py
from .leak_guard import LeakageGuard
from .scale import RaceScaler
from .smoother import BayesianSmoother
from .time_calc import SpeedTimeCalculator
from .track_bias import TrackEncoder

__all__ = [
    "BayesianSmoother",
    "RaceScaler",
    "SpeedTimeCalculator",
    "TrackEncoder",
    "LeakageGuard",
]
```

---

### File: `features\utils\leak_guard.py`

```py
import warnings
from typing import List
import numpy as np
import pandas as pd


class LeakageGuard:
    """自動化檢查產出的 Feature DataFrame 是否存在 Data Leakage 或異常值。"""

    @staticmethod
    def check_future_leakage(
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        threshold: float = 0.90,
    ) -> float:
        """檢查特徵與 Target 的相關性，若絕對值過高則自動發出警告。"""
        valid_df = df[[feature_col, target_col]].dropna()
        if len(valid_df) < 2:
            return 0.0

        corr = valid_df[feature_col].corr(valid_df[target_col])

        # 處理 Pandas corr() 回傳 NaN 的情況
        if pd.isna(corr):
            return 0.0

        corr_val = float(corr)

        if abs(corr_val) >= threshold:
            warnings.warn(
                f"🚨 [LEAKAGE WARNING] 特徵 `{feature_col}` 與 Target `{target_col}` 的相關係數高達 {corr_val:.4f}！請檢查是否漏寫 `.shift(1)`！",
                UserWarning,
            )
        return corr_val

    @staticmethod
    def assert_no_null_keys(df: pd.DataFrame, key_cols: List[str]):
        """確保 Primary Keys 完全沒有缺失值。"""
        for col in key_cols:
            null_count = df[col].isnull().sum()
            assert (
                null_count == 0
            ), f"❌ Key 欄位 `{col}` 包含 {null_count} 個 Null/NaN 缺失值！"

    @staticmethod
    def validate_feature_dataframe(
        df: pd.DataFrame, required_keys: List[str]
    ) -> bool:
        """檢驗生成後的 Feature DataFrame 是否符合管道規格。"""
        LeakageGuard.assert_no_null_keys(df, required_keys)
        assert len(df) > 0, "❌ 產出的 Feature DataFrame 筆數為 0！"
        return True
```

---

### File: `features\utils\scale.py`

```py
import numpy as np
import pandas as pd


class RaceScaler:
    """專門處理同場賽事（Race-Level）特徵相對化與標準化."""

    @staticmethod
    def race_z_score(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        feature_name: str = None,
    ) -> pd.Series:
        """計算欄位在同場賽事中的 Z-Score (X - Mean) / Std."""
        grp = df.groupby(race_col)[value_col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, 1e-6).fillna(1e-6)

        z_score = (df[value_col] - mean) / std
        return z_score.rename(
            feature_name if feature_name else f"{value_col}_race_z"
        )

    @staticmethod
    def race_diff_from_mean(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        feature_name: str = None,
    ) -> pd.Series:
        """計算欄位與同場平均值的差額 (X - Mean)."""
        mean = df.groupby(race_col)[value_col].transform("mean")
        diff = df[value_col] - mean
        return diff.rename(
            feature_name if feature_name else f"{value_col}_diff_mean"
        )

    @staticmethod
    def race_rank(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        ascending: bool = False,
        feature_name: str = None,
    ) -> pd.Series:
        """計算數值在同場賽事中的相對排名 (例如：賠率第幾高、負重第幾重)."""
        rank = df.groupby(race_col)[value_col].rank(
            ascending=ascending, method="min"
        )
        return rank.rename(
            feature_name if feature_name else f"{value_col}_race_rank"
        )
```

---

### File: `features\utils\smoother.py`

```py
# features/utils/smoother.py (加強參數相容與安全防護修正版)

import numpy as np
import pandas as pd


class BayesianSmoother:
    """貝氏平滑器與滾動統計工具庫 (防 Data Leakage、安全向量化與高相容性版)."""

    @staticmethod
    def calc_global_smooth_rate(
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        prior_alpha: float = 10.0,
        baseline_rate: float = 0.08,
    ) -> pd.Series:
        """計算歷史擴展窗口 (Expanding) 貝氏平滑率 (嚴格排除當場賽事)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 防洩漏：先使用原生 groupby.shift(1) 排除當場數據，避免使用 apply 導致 empty concat 崩潰
        shifted_target = df.groupby(group_cols)[target_col].shift(1)

        # 進行累積擴展窗口計算
        cum_sum = shifted_target.groupby([df[c] for c in group_cols]).cumsum()
        cum_count = shifted_target.groupby([df[c] for c in group_cols]).cumcount()

        smoothed_rate = (cum_sum.fillna(0.0) + prior_alpha * baseline_rate) / (
            cum_count.fillna(0.0) + prior_alpha
        )
        return smoothed_rate.fillna(baseline_rate).reindex(df.index)

    @staticmethod
    def calc_rolling_stat(
        df: pd.DataFrame,
        group_cols: str | list[str],
        value_col: str,
        window_size: int = 5,
        stat_type: str = "mean",
        min_periods: int = 1,
        window: int = None,  # 🔒 相容性修復：允許外部傳入 window 參數
        **kwargs,
    ) -> pd.Series:
        """計算動態滾動統計量 (嚴格排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # 若外部使用了 window 參數，自動覆蓋 window_size
        effective_window = window if window is not None else window_size

        if df.empty or value_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        # 🔒 1. 使用向量化的 groupby.shift(1) 排除當場數據
        shifted_series = df.groupby(group_cols)[value_col].shift(1)

        # 🔒 2. 對 shift 後的 Series 進行分組滾動計算
        grouped_shifted = shifted_series.groupby(
            [df[c] for c in group_cols]
            if len(group_cols) > 1
            else df[group_cols[0]]
        )

        if stat_type == "mean":
            res = grouped_shifted.rolling(
                window=effective_window, min_periods=min_periods
            ).mean()
        elif stat_type == "std":
            res = (
                grouped_shifted.rolling(
                    window=effective_window, min_periods=min_periods
                )
                .std()
                .fillna(0.0)
            )
        else:
            raise ValueError(f"不支援的 stat_type: {stat_type}")

        # 🔒 3. 重置 MultiIndex 並精確對齊原始 df.index
        if isinstance(res.index, pd.MultiIndex):
            res = res.reset_index(level=list(range(len(group_cols))), drop=True)

        return res.reindex(df.index)

    @staticmethod
    def calc_rolling_smooth_rate(
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        window_size: int = 5,
        prior_alpha: float = 5.0,
        baseline_rate: float = 0.08,
        window: int = None,  # 🔒 相容性修復：允許外部傳入 window 參數
        **kwargs,
    ) -> pd.Series:
        """計算滾動貝氏平滑勝率 (嚴格排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        effective_window = window if window is not None else window_size

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 向量化 Shift(1)
        shifted_target = df.groupby(group_cols)[target_col].shift(1)
        grouped_shifted = shifted_target.groupby(
            [df[c] for c in group_cols]
            if len(group_cols) > 1
            else df[group_cols[0]]
        )

        roll_sum = grouped_shifted.rolling(
            window=effective_window, min_periods=1
        ).sum()
        roll_count = grouped_shifted.rolling(
            window=effective_window, min_periods=1
        ).count()

        if isinstance(roll_sum.index, pd.MultiIndex):
            roll_sum = roll_sum.reset_index(
                level=list(range(len(group_cols))), drop=True
            )
            roll_count = roll_count.reset_index(
                level=list(range(len(group_cols))), drop=True
            )

        roll_sum = roll_sum.reindex(df.index).fillna(0.0)
        roll_count = roll_count.reindex(df.index).fillna(0.0)

        smoothed_rate = (roll_sum + prior_alpha * baseline_rate) / (
            roll_count + prior_alpha
        )
        return smoothed_rate.fillna(baseline_rate)
```

---

### File: `features\utils\time_calc.py`

```py
from typing import Optional
import pandas as pd


class SpeedTimeCalculator:
    """專門處理賽事時間、段速與標準化速度計算。"""

    @staticmethod
    def calc_speed_mps(
        distance_col: pd.Series,
        time_sec_col: pd.Series,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """計算平均每秒跑多少米 (Meters Per Second, m/s)。"""
        valid_time = time_sec_col.replace(0, pd.NA)
        speed = distance_col / valid_time
        return speed.rename(
            feature_name if feature_name else "speed_meters_per_sec"
        )

    @staticmethod
    def normalize_time_by_distance(
        time_sec_col: pd.Series,
        distance_col: pd.Series,
        target_dist: float = 1200.0,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """將不同路程的時間按比例折算至標準路程秒數（預設折算為 1200 米）。"""
        valid_dist = distance_col.replace(0, pd.NA)
        normalized_time = (time_sec_col / valid_dist) * target_dist
        return normalized_time.rename(
            feature_name
            if feature_name
            else f"norm_time_{int(target_dist)}m"
        )
```

---

### File: `features\utils\track_bias.py`

```py
from typing import Optional
import pandas as pd


class TrackEncoder:
    """跑道條件、路程與檔位組合的編碼與清洗器。"""

    @staticmethod
    def categorize_course_type(
        track_type_col: pd.Series, feature_name: Optional[str] = None
    ) -> pd.Series:
        """將 track_type 統一歸類為 TURF (草地) 或 AWT (全天候/泥地)。"""

        def _clean(val):
            if pd.isna(val):
                return "UNKNOWN"
            s = str(val).upper().strip()
            # 增強匹配條件，涵蓋 HKJC 常見的全天候/泥地標示
            if any(
                kw in s
                for kw in ["ALL WEATHER", "DIRT", "AWT", "ALL-WEATHER"]
            ):
                return "AWT"
            return "TURF"

        res = track_type_col.apply(_clean)
        return res.rename(
            feature_name if feature_name else "course_type_clean"
        )

    @staticmethod
    def create_track_draw_combo(
        venue_col: pd.Series,
        track_type_col: pd.Series,
        draw_col: pd.Series,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """建立賽場+跑道+檔位組合 Key（例如：ST_TURF_DRAW_1）。"""
        course_clean = TrackEncoder.categorize_course_type(track_type_col)
        combo = (
            venue_col.astype(str)
            + "_"
            + course_clean.astype(str)
            + "_DRAW_"
            + draw_col.astype(str)
        )
        return combo.rename(
            feature_name if feature_name else "track_draw_combo_key"
        )
```

---

### File: `models\__init__.py`

```py

```

---

### File: `models\base_model.py`

```py
import abc
import logging
from typing import List, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    所有機器學習模型的抽象基底類別 (Abstract Base Class)
    定義統一的介面規範，確保不同的演算法 (如 XGBRanker, LightGBMRanker)
    具有一致的訓練、預測與存檔行為。
    """

    def __init__(self, model_params: dict = None):
        """
        :param model_params: 模型的超參數字典 (Hyperparameters)
        """
        self.model_params = model_params or {}
        self.model = None
        self.feature_cols: List[str] = []

    @abc.abstractmethod
    def fit(
        self, 
        train_df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str, 
        groups: np.ndarray = None,
        eval_set: Tuple[pd.DataFrame, List[str], str, np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        模型訓練介面
        
        :param train_df: 訓練集的 DataFrame
        :param feature_cols: 參與訓練的特徵欄位名稱清單
        :param target_col: 目標標籤欄位名稱 (例如 'placing' 或 'is_win')
        :param groups: 排序模型專用的賽事群組陣列 (XGBRanker / LightGBMRanker 必填)
        :param eval_set: 驗證集資料 (val_df, feature_cols, target_col, val_groups)
        """
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        模型預測介面 (回傳預測分數或機率)
        
        :param df: 包含特徵的 DataFrame
        :return: 1D numpy array 預測結果
        """
        pass

    def save(self, filepath: str) -> None:
        """
        將訓練好的模型序列化並儲存至硬碟
        """
        import joblib
        try:
            joblib.dump(self, filepath)
            logger.info(f"✅ 模型已成功儲存至: {filepath}")
        except Exception as e:
            logger.error(f"❌ 模型儲存失敗 ({filepath}): {e}")
            raise e

    @classmethod
    def load(cls, filepath: str) -> Any:
        """
        從硬碟載入已序列化的模型
        """
        import joblib
        try:
            model_instance = joblib.load(filepath)
            logger.info(f"✅ 模型已成功從 {filepath} 載入")
            return model_instance
        except Exception as e:
            logger.error(f"❌ 模型載入失敗 ({filepath}): {e}")
            raise e
```

---

### File: `models\data_loader.py`

```py
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import settings

# 設定 Logging
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """賽馬訓練數據加載與預處理器 (Data Feed Provider) - 【完全不含賠率版本】

    職責：
    1. 從 DBManager 撈取 feature_matrix 與 race_results 整合數據。
    2. 自動由 placing 衍生二元標籤 (is_win, is_top3)（若資料庫中未包含）。
    3. 根據 config/settings.json 進行類別型態轉換 (astype('category'))。
    4. 嚴格清理並規範特徵型態，防止 XGBoost/LGBM 底層 C++ 引擎因 Category 索引含 float 而崩潰。
    5. 動態分離並提取 Feature、Target、ID 與 Evaluation 欄位，嚴格排除賽後洩漏欄位與所有賠率欄位。
    6. 確保資料按 race_id 排序，並計算 XGBRanker/LGBMRanker 所需的 group 陣列。
    7. 記憶體快取 (In-Memory Caching)：防止多次訓練或 Optuna Tuning 時重複載入與預處理。
    """

    def __init__(self, db_manager=None):
        """:param db_manager: DBManager 實例。若未傳入，將自動初始化新實例。"""
        if db_manager is None:
            from database.db_manager import DBManager

            self.db = DBManager()
        else:
            self.db = db_manager

        # 💡 快取儲存字典: Key 固定為 (include_odds: bool = False)，Value 為 (df, feature_cols, groups)
        self._cache = {}

    def clear_cache(self):
        """🧹 手動清空記憶體快取 (例如在重新執行 Step 5 特徵工程後使用)"""
        self._cache.clear()
        logger.info("🧹 已成功清空 RaceDataLoader 記憶體快取！")

    def get_feature_cols(
        self, df: pd.DataFrame, include_odds: bool = False
    ) -> List[str]:
        """嚴格特徵過濾：

        1. 無條件剔除「當場賽果特徵 (Post-race Data Leakage)」。
        2. 【全面無條件剔除】所有與賠率 (Odds) 及市場指標相關之特徵。
        """
        # 1. 基礎排除清單 (ID, Target, Evaluation)
        exclude_set = set(
            settings.id_cols + settings.target_cols + settings.eval_cols
        )

        # 2. 🚨【無條件絕對剔除】當場賽果數據 (Post-race Data Leakage)
        post_race_leakage_cols = [
            "placing",
            "finish_time_sec",
            "finish_time_race_z",  # 當場完賽時間 Z-Score
            "last_400m_speed_z",  # 當場末腳速度 Z-Score
            "early_pace_expenditure_z",  # 當場早段搶放 Z-Score
            "speed_mps_last_sectional",
            "sectional_time_last",
            "sec1_time",
            "sec2_time",
            "sec3_time",
            "sec4_time",
            "sec5_time",
            "sec6_time",
            "position",
            "plc",
            "margin_len",
        ]
        exclude_set.update(post_race_leakage_cols)

        # 3. 🛡️【完全無條件剔除】所有賠率與市場相關特徵 (Odds & Market Features)
        strict_odds_cols = [
            "win_odds",
            "win_odds_race_z",
            "win_odds_race_rank",
            "odds_implied_prob",
            "is_market_favorite",
            "odds_race_zscore",
            "win_odds_inv",
            "odds_vs_history_win_rate_gap",
            "odds_rank_in_race",
            "implied_prob_share",
        ]
        exclude_set.update(strict_odds_cols)

        # 額外掃描並無條件剔除欄位名稱中包含 'odds' 或 'market' 的動態欄位
        odds_features = [
            col
            for col in df.columns
            if "odds" in col.lower() or "market" in col.lower()
        ]
        exclude_set.update(odds_features)

        feature_cols = [c for c in df.columns if c not in exclude_set]
        return feature_cols

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """對原始 DataFrame 進行標籤衍生、形態轉換與排序等預處理步驟。

        :param df: 原始 merged DataFrame
        :return: 預處理後的 DataFrame
        """
        df = df.copy()

        # 1. 若原始資料僅有 placing，自動衍生二元分類目標標籤 (is_win, is_top3)
        if "placing" in df.columns:
            if "is_win" not in df.columns:
                df["is_win"] = (df["placing"] == 1).astype(int)
            if "is_top3" not in df.columns:
                df["is_top3"] = (df["placing"] <= 3).astype(int)

        # 2. 根據 settings 轉換類別型特徵，防禦 C++ 引擎的 Category float index 崩潰
        cat_cols = set(settings.categorical_cols)
        for col in df.columns:
            if col in cat_cols:
                # 先將 NaN 或混雜格式填補後轉字串，再轉 category，避免 category index 出現 float/NaN
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace({"nan": "missing", "None": "missing", "<NA>": "missing"})
                )
                df[col] = df[col].astype("category")
            elif (
                col not in settings.id_cols
                and col not in settings.eval_cols
                and col not in settings.target_cols
            ):
                # 非 ID/Target/Category 的純數值特徵，統一轉為 float32，避免 float64 或 object 型態殘留
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    "float32"
                )

        # 3. 確保資料嚴格按 race_id 排序 (Ranking 演算法必備)
        if "race_id" in df.columns:
            df = df.sort_values("race_id").reset_index(drop=True)

        return df

    @staticmethod
    def prepare_ranking_groups(df: pd.DataFrame) -> np.ndarray:
        """計算每場賽事 (race_id) 的馬匹數量陣列，供 XGBRanker / LightGBMRanker 使用。

        :param df: 已按 race_id 排序的 DataFrame
        :return: 包含每場賽事出賽馬匹數量的 1D numpy array
        """
        if "race_id" not in df.columns:
            raise KeyError(
                "DataFrame 中找不到 'race_id' 欄位，無法計算 ranking groups！"
            )

        groups = df.groupby("race_id", sort=False).size().to_numpy()
        return groups

    def load_dataset(
        self, include_odds: bool = False, force_reload: bool = False
    ) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
        """[主入口 API] 發起數據載入、標籤衍生、類型轉換與特徵提取流程。

        註：`include_odds` 預設強制為 False，永遠排除賠率數據。

        :param include_odds: 保留參數介面，固定為 False
        :param force_reload: 若為 True，會無視快取並重新從資料庫載入與預處理
        :return: (processed_df, feature_cols, groups)
        """
        cache_key = False  # 強制快取 Key 為不含賠率狀態

        # 💡 1. 檢查記憶體快取：若已有快取且未要求強制重載，直接返回
        if not force_reload and cache_key in self._cache:
            cached_df, cached_features, cached_groups = self._cache[cache_key]
            logger.info(
                "⚡ [Cache Hit] 直接從記憶體載入數據集 (完全排除賠率特徵)！"
            )
            return cached_df.copy(), list(cached_features), cached_groups.copy()

        logger.info("📦 開始從 DBManager 載入特徵矩陣與賽果數據...")
        raw_df = self.db.load_feature_result()

        if raw_df is None or raw_df.empty:
            raise ValueError(
                "【錯誤】資料庫中的 feature_matrix 或 race_results 為空，請先執行特徵工程！"
            )

        logger.info(
            f"📊 原始數據載入完成，共 {len(raw_df)} 條記錄。開始進行無賠率預處理..."
        )

        # 2. 預處理 (自動補齊標籤、類別轉型、排序)
        df = self.process_dataframe(raw_df)

        # 3. 提取特徵欄位清單 (強制剔除賠率特徵)
        feature_cols = self.get_feature_cols(df, include_odds=False)

        # 4. 計算 Ranking Groups
        groups = self.prepare_ranking_groups(df)

        # 💡 5. 寫入記憶體快取
        self._cache[cache_key] = (df, feature_cols, groups)

        logger.info(
            f"✅ DataLoader 處理完畢並已建立快取："
            f"記錄數={len(df)}, 賽事場數={len(groups)}, 純基本面特徵數={len(feature_cols)} (完全排除賠率)"
        )

        return df.copy(), list(feature_cols), groups.copy()
```

---

### File: `models\model_pipeline.py`

```py
import logging
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from database.db_manager import DBManager
from models.data_loader import RaceDataLoader
from models.metrics.ranking import RankingMetrics
from models.registry import ModelRegistry


import models.wrappers.xgb_wrapper
from models.validation.time_split import TimeSeriesSplitter

logger = logging.getLogger(__name__)


class ModelPipeline:
    """賽馬機器學習統籌工作流 (Model Pipeline)

    負責將資料載入、時間切分、模型訓練、超參數尋優、評估與推論串聯成標準化流程。
    """

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self.data_loader = RaceDataLoader(db_manager)
        self.splitter = TimeSeriesSplitter(date_col="date", group_col="race_id")

    def run_train_pipeline(
        self,
        model_name: str = "xgb_ranker",
        model_params: Optional[Dict[str, Any]] = None,
        val_days: int = 30,
        feature_cols: Optional[list] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """執行完整的訓練與驗證 Pipeline"""
        logger.info("🚀 開始執行訓練 Pipeline...")

        # 1. 載入並自動預處理數據
        df, default_feature_cols, _ = self.data_loader.load_dataset(
            include_odds=True
        )

        if df.empty:
            raise ValueError("【錯誤】訓練資料集為空，無法進行訓練！")

        # 解析日期：支援 YYYY/MM/DD 或 YYYY-MM-DD
        if "date" not in df.columns:
            if "race_date" in df.columns:
                df["date"] = df["race_date"]
            else:
                df["date"] = df["race_id"].astype(str).str.extract(
                    r"(\d{4}[/-]\d{2}[/-]\d{2})"
                )[0]

        # 轉為標準 datetime 格式
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 清理與驗證
        cleaned_len = len(df.dropna(subset=["date"]))
        df = df.dropna(subset=["date"]).copy()

        if cleaned_len == 0:
            raise ValueError(
                "【錯誤】無法從 race_id 解析出任何有效日期！"
            )

        logger.info(
            f"💡 成功解析 {cleaned_len} 筆賽事日期 (日期範圍: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')})"
        )

        # 若使用者未自訂 feature_cols，則使用 DataLoader 自動提取的預設特徵
        if feature_cols is None:
            feature_cols = list(default_feature_cols)

        # 定義禁止傳入模型的「未來官子/當場結果/當場賠率」欄位 (防 Leakage)
        forbidden_cols = {
            # --- 識別符與時間 ---
            "race_id",
            "horse_id",
            "date",
            "race_date",
            # --- 當場賽後結果 (Target & Race Result Leakage) ---
            "placing",
            "is_win",
            "is_top3",
            "relevance_score",
            "finish_time_sec",
            "margin_len",
            "sectional_time_last",  # 當場末腳時間 (當場結果)
            "position_gain_first_to_last",  # 當場走位變化 (當場結果)
            "speed_mps_overall",  # 當場平均速度 (當場結果)
            "speed_mps_last_sectional",  # 當場末段速度 (當場結果)
            # --- 當場臨場賠率 (Market Leakage) ---
            "win_odds",
            "win_odds_inv",
            "odds_implied_prob",
            "is_market_favorite",
            "win_odds_race_rank",
            "win_odds_race_z",
            "odds_race_zscore",
            "odds_vs_history_win_rate_gap",
            "rating_x_rank_weight",
        }

        # 雙重防洩漏保險：強制過濾禁用的欄位
        feature_cols = [
            col for col in feature_cols if col not in forbidden_cols
        ]

        if not feature_cols:
            raise ValueError(
                "【錯誤】經過禁用的欄位過濾後，有效特徵數為 0，無法進行訓練！"
            )

        # 確保訓練資料中包含模型所需的 relevance_score 標籤
        if "relevance_score" not in df.columns and "placing" in df.columns:
            df["relevance_score"] = df["placing"].apply(
                lambda p: max(0, 4 - p) if p <= 3 else 0
            )

        logger.info(
            f"📊 總樣本數: {len(df)}, 有效特徵數: {len(feature_cols)}"
        )

        # 2. 時間序列切分 (防止資料洩漏)
        train_df, val_df, train_groups, val_groups = (
            self.splitter.split_by_days(df, val_days=val_days)
        )

        # 3. 創建模型實例
        model = ModelRegistry.create(
            name=model_name, model_params=model_params
        )

        # 4. 執行模型訓練
        model.fit(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="relevance_score",
            groups=train_groups,
            eval_set=(val_df, feature_cols, "relevance_score"),
            eval_groups=val_groups,
        )

        # =========================================================================
        # 📊 特徵重要性 (Feature Importance) 提取與日誌輸出
        # =========================================================================
        self._log_feature_importance(model, feature_cols)

        # 5. 模型預測與評估
        logger.info("📈 正在計算驗證集評估指標...")
        val_preds = model.predict(val_df)
        val_df_evaluated = val_df.copy()
        val_df_evaluated["pred_score"] = val_preds

        top1_win_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_placing_col="placing",
            group_col="race_id",
            k=1,
        )
        top3_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_placing_col="placing",
            group_col="race_id",
            k=3,
        )
        ndcg = RankingMetrics.mean_ndcg_score(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_relevance_col="relevance_score",
            group_col="race_id",
            k=5,
        )

        metrics = {
            "top1_win_rate": top1_win_rate,
            "top3_rate": top3_rate,
            "ndcg@5": ndcg,
        }

        logger.info(f"🎯 驗證結果指標: {metrics}")
        return model, metrics

    def _get_default_search_space(self, model_name: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
        """針對不同模型提供預設的 Optuna 超參數尋優空間 (Search Space)"""
        
        if model_name == "xgb_ranker":
            def xgb_ranker_space(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    "objective": trial.suggest_categorical("objective", ["rank:pairwise", "rank:ndcg"]),
                    "eval_metric": "ndcg@5",
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 800, 1500, step=100),
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 150),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 25.0),
                    "random_state": 42,
                    "tree_method": "hist",
                    "enable_categorical": True,
                }
            return xgb_ranker_space

        elif model_name == "lgb_ranker":
            def lgb_ranker_space(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=100),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "random_state": 42,
                }
            return lgb_ranker_space

        else:
            raise ValueError(f"【錯誤】未定義該模型的超參數搜尋空間: {model_name}")

    def run_tune_pipeline(
        self,
        model_name: str = "xgb_ranker",
        n_trials: int = 30,
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
        feature_cols: Optional[list] = None,
        custom_param_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
        retrain_best: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """管線內自動超參數尋優 (Optuna Tuning)

        :param model_name: 模型名稱 ('xgb_ranker', 'lgb_ranker' 等)
        :param n_trials: 搜尋試驗輪數
        :param val_days: 驗證集切分天數
        :param metric_name: 優化目標指標 ('top1_win_rate', 'ndcg@5', 'top3_rate')
        :param direction: 'maximize' 或 'minimize'
        :param feature_cols: (可選) 自訂特徵欄位清單
        :param custom_param_fn: (可選) 自訂 Optuna Search Space 函數
        :param retrain_best: 尋優結束後，是否使用最佳參數自動重新訓練最終模型
        :return: (best_params, best_model_instance)
        """
        from models.hyperopt.optuna_tuner import OptunaTuner
        logger.info(f"🎯 開始執行管線自動尋優: [Model: {model_name}] [Target: {metric_name}] [Trials: {n_trials}]")

        # 1. 取得 Search Space
        param_fn = custom_param_fn or self._get_default_search_space(model_name)

        # 2. 實例化超參數尋優器
        tuner = OptunaTuner(
            pipeline=self,
            model_name=model_name,
            val_days=val_days,
            metric_name=metric_name,
            direction=direction,
        )

        # 3. 執行 Optuna 尋優
        study = tuner.optimize(
            param_fn=param_fn,
            n_trials=n_trials,
            study_name=f"{model_name}_tune",
        )

        best_params = study.best_params
        logger.info(f"🏆 管線尋優完成！最佳指標值 [{metric_name}]: {study.best_value:.4f}")
        logger.info(f"💡 最佳參數組合: {best_params}")

        # 4. 選項：自動以最佳參數重新訓練最終模型
        best_model = None
        if retrain_best:
            logger.info("🚀 正在使用最佳超參數重新訓練最終模型...")
            best_model, final_metrics = self.run_train_pipeline(
                model_name=model_name,
                model_params=best_params,
                val_days=val_days,
                feature_cols=feature_cols,
            )
            logger.info(f"✅ 最終模型重新訓練完畢，驗證集指標: {final_metrics}")

        return best_params, best_model

    def _log_feature_importance(self, model: Any, feature_cols: list, top_n: int = 20):
        """解析內部原生的模型物件並列印特徵重要性 (相容 Wrapper)"""
        try:
            # 1. 解開 Wrapper 取得底層的原生模型 (如 XGBRanker/LGBMRanker)
            raw_model = getattr(model, "model", model)

            # 2. 提取特徵重要性數值
            importances = None
            if hasattr(raw_model, "feature_importances_"):
                importances = raw_model.feature_importances_
            elif hasattr(raw_model, "get_score"):  # 原生 XGBoost Booster 結構
                score_dict = raw_model.get_score(importance_type="gain")
                importances = [score_dict.get(f"f{i}", score_dict.get(col, 0.0)) for i, col in enumerate(feature_cols)]

            if importances is None or len(importances) != len(feature_cols):
                logger.warning("⚠️ 無法讀取該模型的特徵重要性 (Feature Importance)。")
                return

            # 3. 組裝為 DataFrame 排序
            fi_df = (
                pd.DataFrame({"feature": feature_cols, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )

            # 4. 列印高亮日誌
            print("\n" + "=" * 60)
            print(f"🔥 [模型特徵權重排行榜]")
            print("=" * 60)
            for idx, row in fi_df.iterrows():
                print(f"  #{idx+1:02d} | {row['feature']:<35} | 權重: {row['importance']:.6f}")
            print("=" * 60 + "\n")

        except Exception as e:
            logger.warning(f"⚠️ 提取 Feature Importance 過程發生異常: {e}")

    def run_inference_pipeline(
        self, model: Any, inference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """執行推論 Pipeline：對給定的最新賽事特徵進行預測評分"""
        logger.info("🔮 開始執行推論 (Inference) Pipeline...")

        preds = model.predict(inference_df)
        result_df = inference_df.copy()
        result_df["pred_score"] = preds

        # 依照賽事 (race_id) 內部對 pred_score 進行排名
        result_df["pred_rank"] = result_df.groupby("race_id")["pred_score"].rank(
            ascending=False, method="min"
        )

        logger.info("✅ 推論完成！")
        return result_df
```

---

### File: `models\registry.py`

```py
import logging
from typing import Dict, Type, Any
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    模型工廠與註冊器 (Model Registry / Factory)
    用於動態註冊、管理與創建不同的機器學習模型包裝類別。
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        類別裝飾器：用於將模型類別註冊到工廠中
        
        用法範例:
            @ModelRegistry.register("xgb_ranker")
            class XGBRankerWrapper(BaseModel):
                ...
        """
        def decorator(subclass: Type[BaseModel]):
            if not issubclass(subclass, BaseModel):
                raise TypeError(f"【錯誤】被註冊的類別 '{subclass.__name__}' 必須繼承自 BaseModel！")
            
            if name in cls._registry:
                logger.warning(f"⚠️ 警告: 模型名稱 '{name}' 已存在於註冊表中，將會被覆蓋。")
                
            cls._registry[name] = subclass
            logger.info(f"📌 成功註冊模型: '{name}' -> {subclass.__name__}")
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, model_params: Any = None) -> BaseModel:
        """
        根據模型名稱動態創建模型實例
        
        :param name: 模型註冊名稱 (如 "xgb_ranker")
        :param model_params: 傳入模型的超參數字典或參數物件
        :return: 對應模型的實例 (BaseModel 的子類別)
        """
        if name not in cls._registry:
            available_models = list(cls._registry.keys())
            raise ValueError(f"【錯誤】找不到名為 '{name}' 的模型！現有可用的模型列表為: {available_models}")
        
        model_cls = cls._registry[name]
        logger.info(f"🔨 正在創建模型實例: '{name}' ({model_cls.__name__})")
        
        if model_params is not None:
            return model_cls(model_params=model_params)
        return model_cls()

    @classmethod
    def list_models(cls) -> list:
        """列出目前所有已註冊的模型名稱"""
        return list(cls._registry.keys())
```

---

### File: `models\hyperopy\__init__.py`

```py
from models.hyperopt.optuna_tuner import OptunaTuner

__all__ = ["OptunaTuner"]
```

---

### File: `models\hyperopy\optuna_tuner.py`

```py
import logging
from typing import Callable, Dict, Any, Optional
import optuna

from models.model_pipeline import ModelPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Optuna 自動超參數尋優器 (Hyperparameter Tuner)
    封裝對 ModelPipeline 的調用，防範資料洩漏並集中管理搜尋實驗。
    """

    def __init__(
        self,
        pipeline: ModelPipeline,
        model_name: str = "xgb_ranker",
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
    ):
        """
        :param pipeline: 已初始化的 ModelPipeline 實例
        :param model_name: 在 ModelRegistry 註冊的模型名稱 (例如 'xgb_ranker')
        :param val_days: 驗證集切分天數
        :param metric_name: 評估指標名稱 ('top1_win_rate', 'ndcg@5', 'top3_rate')
        :param direction: 優化方向 ('maximize' 或 'minimize')
        """
        self.pipeline = pipeline
        self.model_name = model_name
        self.val_days = val_days
        self.metric_name = metric_name
        self.direction = direction

    def _create_objective(self, param_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> Callable[[optuna.Trial], float]:
        """建立內部使用的 Objective 函數"""

        def objective(trial: optuna.Trial) -> float:
            # 1. 透過外部傳入的 param_fn 生成該輪 Trial 的超參數組合
            model_params = param_fn(trial)

            try:
                # 2. 調用 Pipeline 進行標準訓練與驗證 (自動處理 TimeSplit)
                _, metrics = self.pipeline.run_train_pipeline(
                    model_name=self.model_name,
                    model_params=model_params,
                    val_days=self.val_days,
                )

                # 3. 提取指定的評估指標
                score = metrics.get(self.metric_name, 0.0)
                return float(score)

            except Exception as e:
                # 防禦機制：若極端參數導致崩潰，給予低分並跳過
                logger.warning(f"⚠️ Trial #{trial.number} 執行異常: {e}")
                return 0.0 if self.direction == "maximize" else 999.0

        return objective

    def optimize(
        self,
        param_fn: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 30,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> optuna.Study:
        """
        執行自動調參流程
        
        :param param_fn: 接受 trial 並回傳 model_params 字典的函數
        :param n_trials: 試驗輪數
        :param timeout: 最大搜尋時間限制 (秒)
        :param study_name: 實驗名稱
        :return: 完成後的 Optuna Study 物件
        """
        # 隱藏 Optuna 過多的預設資訊
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_name = study_name or f"{self.model_name}_optimization"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(),
        )

        logger.info(
            f"🚀 開始執行 Optuna 自動超參數尋優 (模型: {self.model_name}, 輪數: {n_trials}, 優化指標: {self.metric_name})..."
        )

        objective_fn = self._create_objective(param_fn)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"🏆 尋優完成！最佳指標 [{self.metric_name}]: {study.best_value:.4f}")
        return study
```

---

### File: `models\metrics\__init__.py`

```py

```

---

### File: `models\metrics\ranking.py`

```py
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class RankingMetrics:
    """
    賽馬排序與預測能力評估指標計算器
    """

    @staticmethod
    @staticmethod
    def top_k_win_rate(
        df: pd.DataFrame, 
        pred_score_col: str = "pred_score", 
        target_placing_col: str = "placing", 
        group_col: str = "race_id",
        k: int = 1
    ) -> float:
        if group_col not in df.columns or pred_score_col not in df.columns or target_placing_col not in df.columns:
            raise ValueError(f"【錯誤】缺少必要欄位，請檢查是否包含 '{group_col}', '{pred_score_col}', '{target_placing_col}'")

        hits = 0
        total_races = 0

        for _, group in df.groupby(group_col):
            if len(group) == 0:
                continue
            
            total_races += 1
            # 取預測分數最高的前 K 匹馬
            top_k_preds = group.sort_values(by=pred_score_col, ascending=False).head(k)
            actual_placings = top_k_preds[target_placing_col].values
            
            if k == 1:
                # 獨贏/冠中率：預測第 1 名實際是否為冠軍 (placing == 1)
                if actual_placings[0] == 1:
                    hits += 1
            else:
                # 位置/上名率（精確計算）：計算預測前 K 名中有幾匹馬實際名次 <= k
                # 例如 k=3 時，計算 3 匹預測前三名中有幾匹實際跑進前三名
                hits += np.sum(actual_placings <= k) / k

        win_rate = hits / total_races if total_races > 0 else 0.0
        return float(win_rate)
    @staticmethod
    def mean_ndcg_score(
        df: pd.DataFrame,
        pred_score_col: str = "pred_score",
        target_relevance_col: str = "relevance",
        group_col: str = "race_id",
        k: int = 5
    ) -> float:
        """
        計算跨所有賽事的平均 NDCG@K 分數
        
        :param df: 包含預測分數與相關性標籤的 DataFrame
        :param pred_score_col: 模型預測得分欄位
        :param target_relevance_col: 相關性標籤欄位 (例如冠軍=3, 亞軍=2, 季軍=1, 其餘=0)
        :param group_col: 賽事 ID 欄位
        :param k: 計算 NDCG 的截斷名次 (預設 5)
        :return: 平均 NDCG 分數 (0.0 ~ 1.0)
        """
        if group_col not in df.columns or pred_score_col not in df.columns or target_relevance_col not in df.columns:
            raise ValueError("【錯誤】缺少計算 NDCG 所需的必要欄位！")

        ndcg_scores = []

        for _, group in df.groupby(group_col):
            if len(group) < 2:
                # 若賽事馬匹數量小於 2，無法有效計算排序，略過
                continue

            y_true = group[target_relevance_col].values.reshape(1, -1)
            y_pred = group[pred_score_col].values.reshape(1, -1)

            # 若真實標籤全部為 0（例如沒有馬跑入前三名或資料不全），跳過避免分母為 0
            if np.sum(y_true) == 0:
                continue

            try:
                score = ndcg_score(y_true, y_pred, k=k)
                ndcg_scores.append(score)
            except Exception as e:
                logger.warning(f"⚠️ 計算賽事 {group[group_col].iloc[0]} 的 NDCG 時發生異常: {e}")

        if not ndcg_scores:
            return 0.0

        return float(np.mean(ndcg_scores))
```

---

### File: `models\validation\__init__.py`

```py

```

---

### File: `models\validation\time_split.py`

```py
import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """
    時間序列賽事切分工具 (Time Series Splitter)
    專門用來將歷史賽事資料依據日期進行切分，確保不會發生未來資料洩漏 (Data Leakage)。
    """

    def __init__(self, date_col: str = "date", group_col: str = "race_id"):
        """
        :param date_col: 日期欄位名稱
        :param group_col: 賽事群組欄位名稱 (如 race_id)
        """
        self.date_col = date_col
        self.group_col = group_col

    def split_by_days(
        self, 
        df: pd.DataFrame, 
        val_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        以最後一天往前推指定天數作為驗證集，其餘為訓練集。
        
        :param df: 包含日期與賽事群組的完整 DataFrame
        :param val_days: 驗證集包含最近多少天的賽事
        :return: (train_df, val_df, train_groups, val_groups)
        """
        if self.date_col not in df.columns:
            raise ValueError(f"【錯誤】DataFrame 中找不到指定的日期欄位: '{self.date_col}'")
        if self.group_col not in df.columns:
            raise ValueError(f"【錯誤】DataFrame 中找不到指定的賽事群組欄位: '{self.group_col}'")

        # 確保日期格式為 datetime
        df_sorted = df.copy()
        df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])

        # 確保資料按照時間與賽事順序排列（Ranker 的硬性要求：同一場比賽的資料必須連續）
        df_sorted = df_sorted.sort_values(by=[self.date_col, self.group_col]).reset_index(drop=True)

        # 計算切分時間點
        max_date = df_sorted[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=val_days)

        logger.info(f"📅 資料集總日期範圍: {df_sorted[self.date_col].min().strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        logger.info(f"✂️ 切分點設定: 驗證集為最近 {val_days} 天 (大於 {split_date.strftime('%Y-%m-%d')})")

        # 劃分 Train 與 Val
        train_df = df_sorted[df_sorted[self.date_col] <= split_date].copy()
        val_df = df_sorted[df_sorted[self.date_col] > split_date].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("【錯誤】切分後的訓練集或驗證集為空！請檢查資料量或 val_days 設定。")

        # 計算 XGBRanker 所需的 groups (每場賽事的馬匹數量)
        # 必須確保 sort=False，且順序與 DataFrame 完全一致
        train_groups = train_df.groupby(self.group_col, sort=False).size().values
        val_groups = val_df.groupby(self.group_col, sort=False).size().values

        logger.info(f"📊 切分完成：訓練集樣本數 {len(train_df)} ({len(train_groups)} 場), 驗證集樣本數 {len(val_df)} ({len(val_groups)} 場)")

        return train_df, val_df, train_groups, val_groups
```

---

### File: `models\wrappers\__init__.py`

```py

```

---

### File: `models\wrappers\xgb_wrapper.py`

```py
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRanker

from models.base_model import BaseModel
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgb_ranker")
class XGBRankerWrapper(BaseModel):
    """基於 XGBoost Ranker 的賽馬排序模型封裝

    具備完整的例外處理與資料防禦機制。
    """

    def __init__(self, model_params: Optional[dict] = None):
        super().__init__(model_params)

        default_params = {
    "objective": "rank:ndcg",
    "max_depth": 5,
    "learning_rate": 0.02241986575232448,
    "n_estimators": 1100,
    "early_stopping_rounds": 137,
    "subsample": 0.7551375320147171,
    "colsample_bytree": 0.5807601058725049,
    "reg_alpha": 0.07583278924335168,
    "reg_lambda": 20.477793985681824,
    # 💡 建議搭配的通用硬體與重現性設定：
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",  # 若有 GPU 可改為 "hist" 並搭配 device="cuda"
}

        if self.model_params:
            default_params.update(self.model_params)

        # 清理潛在會觸發 C++ 報錯的相容性參數
        default_params.pop("eval_metric", None)
        default_params.pop("lambdarank_pair_method", None)

        self.model_params = default_params
        self.feature_dtypes = {}

        try:
            self.model = XGBRanker(**self.model_params)
        except Exception as e:
            logger.error(f"❌ 初始化 XGBRanker 失敗，請檢查參數設定: {self.model_params}")
            raise RuntimeError(f"XGBRanker 初始化異常: {e}") from e

    def _preprocess_features(
        self, df: pd.DataFrame, feature_cols: List[str], is_training: bool = True
    ) -> pd.DataFrame:
        """資料型態預處理與異常檢查"""
        if df is None or df.empty:
            raise ValueError("【錯誤】輸入的 DataFrame 為空或 None！")

        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"【錯誤】DataFrame 中找不到以下特徵欄位: {missing_cols}")

        X = df[feature_cols].copy()

        try:
            for col in feature_cols:
                if is_training:
                    # 1. 如果是 object 型態，先嘗試轉為數值 (無法轉的設為 NaN 或保留)
                    if X[col].dtype == "object":
                        converted = pd.to_numeric(X[col], errors="coerce")
                        # 如果轉完後全都是 NaN，說明它是真正的字串欄位 (如文字類別)，改轉為 category
                        if converted.isna().all() and not X[col].isna().all():
                            X[col] = X[col].astype("category")
                        else:
                            X[col] = converted

                    # 2. 若原本就是 category，保持 category
                    elif str(X[col].dtype) == "category":
                        X[col] = X[col].astype("category")

                    self.feature_dtypes[col] = X[col].dtype

                else:
                    # 預測階段：對齊訓練時的 dtype
                    target_dtype = self.feature_dtypes.get(col)
                    if target_dtype is not None:
                        if str(target_dtype) == "category":
                            categories = getattr(target_dtype, "categories", None)
                            X[col] = pd.Categorical(X[col], categories=categories)
                        else:
                            X[col] = pd.to_numeric(X[col], errors="coerce").astype(target_dtype)

        except Exception as e:
            logger.error(f"❌ 特徵型態預處理過程發生異常 (is_training={is_training}): {e}")
            raise TypeError(f"特徵預處理失敗: {e}") from e

        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        groups: np.ndarray = None,
        eval_set: Optional[Tuple[pd.DataFrame, List[str], str]] = None,
        eval_groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """訓練 XGBRanker 模型（含完整例外處理）"""
        # 1. 基礎輸入參數校驗
        if groups is None or len(groups) == 0:
            raise ValueError("【錯誤】XGBRanker 訓練必須提供非空的 groups 陣列（每場賽事的馬匹數量）！")

        if target_col not in train_df.columns:
            raise KeyError(f"【錯誤】訓練資料集中不存在目標標籤欄位 '{target_col}'！")

        if sum(groups) != len(train_df):
            raise ValueError(
                f"【錯誤】groups 的總和 ({sum(groups)}) 與訓練樣本總數 ({len(train_df)}) 不一致！"
            )

        self.feature_cols = feature_cols

        # 2. 特徵預處理
        try:
            X_train = self._preprocess_features(train_df, feature_cols, is_training=True)
            y_train = train_df[target_col]
        except Exception as e:
            logger.error(f"❌ 訓練集資料準備失敗: {e}")
            raise

        fit_params = {"group": groups}

        # 3. 驗證集與 Early Stopping 檢測
        if eval_set is not None:
            if eval_groups is None or len(eval_groups) == 0:
                raise ValueError("【錯誤】提供了 eval_set 時，必須同時提供非空的 eval_groups！")

            try:
                val_df, val_feature_cols, val_target_col = eval_set
                
                if sum(eval_groups) != len(val_df):
                    raise ValueError(
                        f"【錯誤】eval_groups 的總和 ({sum(eval_groups)}) 與驗證集樣本數 ({len(val_df)}) 不一致！"
                    )

                X_val = self._preprocess_features(val_df, val_feature_cols, is_training=False)
                y_val = val_df[val_target_col]

                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["eval_group"] = [eval_groups]

                if "early_stopping_rounds" not in self.model.get_params():
                    self.model.set_params(early_stopping_rounds=50)

            except Exception as e:
                logger.error(f"❌ 驗證集 (eval_set) 處理失敗: {e}")
                raise

            if "verbose" not in kwargs:
                kwargs["verbose"] = False
        else:
            if "early_stopping_rounds" in self.model.get_params():
                self.model.set_params(early_stopping_rounds=None)

        # 4. 執行 fit 並捕獲 XGBoost 底層 C++ / Runtime 異常
        kwargs.pop("eval_metric", None)  # 確保不透傳引發衝突的 metric

        logger.info(
            f"🚀 開始訓練 XGBRanker 模型，特徵數: {len(feature_cols)}, 訓練樣本數: {len(X_train)}"
        )

        try:
            self.model.fit(X_train, y_train, **fit_params, **kwargs)
            logger.info("✅ XGBRanker 模型訓練成功！")

        except xgb.core.XGBoostError as e:
            logger.error(f"❌ XGBoost 底層 C++ 引擎拋出錯誤: {e}")
            raise RuntimeError(f"XGBoost 訓練引擎崩潰: {e}") from e

        except MemoryError as e:
            logger.error("❌ 訓練過程記憶體溢出 (Out of Memory)！請嘗試減少 n_estimators 或 max_depth。")
            raise MemoryError("模型訓練記憶體不足") from e

        except Exception as e:
            logger.error(f"❌ 訓練過程中發生未預期的錯誤: {e}")
            raise RuntimeError(f"模型 fit 失敗: {e}") from e

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """預測賽事中各馬匹的排序得分（含例外處理）"""
        if self.model is None:
            raise RuntimeError("【錯誤】模型尚未訓練或載入，無法進行預測！")

        try:
            X = self._preprocess_features(df, self.feature_cols, is_training=False)
            scores = self.model.predict(X)

            if len(scores) != len(df):
                raise ValueError(f"【錯誤】預測結果數量 ({len(scores)}) 與輸入資料筆數 ({len(df)}) 不符！")

            return scores

        except KeyError as e:
            logger.error(f"❌ 推論失敗，缺失必要特徵欄位: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 模型推論 (predict) 過程發生異常: {e}")
            raise RuntimeError(f"模型預測失敗: {e}") from e
```

---

### File: `scraper\__init__.py`

```py

```

---

### File: `scraper\data_manager.py`

```py
import json
import pathlib
from typing import Any, Dict
import aiofiles
from config.settings import settings

class DataManager:
    def __init__(self, json_path: pathlib.Path = settings.raw_json_dir):
        self.json_path = json_path
        self.json_path.mkdir(parents=True, exist_ok=True)

    def _get_date_file_path(self, date_str: str) -> pathlib.Path:
        """格式化日期檔名：YYYY-MM-DD.json"""
        formatted_date = f"{date_str[:4]}-{date_str[5:7]}-{date_str[8:]}"
        return self.json_path / f"{formatted_date}.json"

    def check_file_exist(self, key_id: str, file_type: str = "horse") -> bool:
        """通用檢查檔案是否存在"""
        clean_id = key_id.strip().upper()

        if file_type == "horse":
            file_path = settings.raw_horses_json_dir / f"{clean_id}.json"
        elif file_type == "race":
            file_path = self._get_date_file_path(clean_id)

        return file_path.is_file()

    async def save_races_json(self, date_str: str, params: Dict[str, Any]) -> None:
        """非同步儲存單日賽果 JSON"""
        file_path = self._get_date_file_path(date_str)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)

    async def save_normal_json(self, file_name: str, params: Any) -> None:
        """非同步儲存一般 JSON (修正舊版路徑 Bug)"""
        file_path = self.json_path / f"{file_name}.json"
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            content = json.dumps(params, ensure_ascii=False, indent=4)
            await f.write(content)

```

---

### File: `scraper\hook.py`

```py
import asyncio
import random
from typing import Optional, Tuple, Any, Dict
import aiohttp
from selectolax.parser import HTMLParser

class Hook:
    """非同步網路請求模組，基於 aiohttp 與 selectolax (內部封裝馬會 URL 邏輯)"""

    def __init__(self):
        # 🌟 將 Base URL 與各種 Endpoint 封裝為私有變數，外部不再需要傳入 url
        self._domain = "https://racing.hkjc.com"
        self._result_endpoint = f"{self._domain}/zh-hk/local/information/localresults"
        self._sectional_endpoint = f"{self._domain}/zh-hk/local/information/displaysectionaltime"
        self._calendar_endpoint = f"{self._domain}/zh-hk/local/information/fixture" # 或對應的日曆 endpoint
        self._rating_endpoint = f"{self._domain}/racing/info/mcs/Chinese/Horses/clas/?&rf=http://racing.hkjc.com/zh-hk/local/information/latestonhorse?View=Horses/clas/&pageid=racing/local"
        self._horse_endpoint = f"{self._domain}/zh-hk/local/information/horse"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """進入 context manager 時建立 aiohttp session"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """離開 context manager 時自動關閉 session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _fetch(self, target_url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Tuple[HTMLParser, str]]:
        """
        私有核心非同步請求函式
        :param target_url: 目標請求的 URL 站點
        :param params: URL 查詢參數
        """
        if not self.session or self.session.closed:
            raise RuntimeError("🚨 Hook 必須在 `async with` 語境內使用！例如: `async with Hook() as hook:`")

        try:
            # 🌟 非同步隨機延遲，降低觸發 IP 封鎖的機率
            await asyncio.sleep(random.uniform(0.5, 1.8))

            async with self.session.get(target_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as response:
                response.raise_for_status()
                
                final_url = str(response.url)
                print(f"🔗 正在請求: {final_url}")

                # 🌟 非同步讀取 Raw Bytes 並直接喂給 selectolax
                content = await response.read()
                tree = HTMLParser(content)

                return tree, final_url

        except aiohttp.ClientError as e:
            print(f"❌ 網絡請求異常 [{target_url}]: {e}")
            return None
        except asyncio.TimeoutError:
            print(f"⏰ 請求超時 [{target_url}]")
            return None
        except Exception as e:
            print(f"💥 未知錯誤 [{target_url}]: {e}")
            return None

    # -------------------------------------------------------------
    # 對外暴露的介面 (由 Hook 內部控制要打哪一個 Endpoint)
    # -------------------------------------------------------------

    async def get_result_tree(self, no: str | int, race_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [基本賽果] 頁面"""
        actual_no = "1" if str(no) == "0" else str(no)
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        return await self._fetch(self._result_endpoint, params=params)

    async def get_sectional_tree(self, no: str | int, race_date: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [分段時間與走位] 頁面 (全新新增)"""
        actual_no = "1" if str(no) == "0" else str(no)
        params = {
            "racedate": race_date,
            "RaceNo": actual_no
        }
        return await self._fetch(self._sectional_endpoint, params=params)

    async def get_calendar_tree(self, year: int, month: int) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [賽事日曆] 頁面"""
        params = {
            "calyear": str(year),
            "calmonth": str(month).zfill(2)
        }
        return await self._fetch(self._calendar_endpoint, params=params)
    
    async def get_horse_tree(self, horse_id: str) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取 [賽事日曆] 頁面"""
        params = {
            "horseid": horse_id
        }
        return await self._fetch(self._horse_endpoint, params=params)

    async def get_rating_tree(self) -> Optional[Tuple[HTMLParser, str]]:
        """非同步獲取預設首頁"""
        return await self._fetch(self._rating_endpoint)
```

---

### File: `scraper\horse_pipeline.py`

```py
import asyncio
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from scraper.data_manager import DataManager
from scraper.hook import Hook
from scraper.parser.horse_parser import HorseProfileParser


class HorseScrapingPipeline:
    """馬匹詳細資料爬蟲調度管道 (Pipeline Layer)"""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.horse_db = DataManager(
            json_path=settings.raw_horses_json_dir
        )

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """深層 Exception 紀錄器：自動擷取引發 Error 的最深層檔案、函式名稱與行號"""
        exc_type, exc_value, exc_tb = sys.exc_info()

        # 追蹤 Traceback 到最深層的發源點
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        file_name = (
            last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown File"
        )
        func_name = (
            last_tb.tb_frame.f_code.co_name if last_tb else "Unknown Func"
        )
        line_no = last_tb.tb_lineno if last_tb else "Unknown Line"

        print(
            f"\n💥 [Horse Pipeline 崩潰] {context_msg}\n"
            f"   ├─ 類型: {exc_type.__name__ if exc_type else type(error).__name__}\n"
            f"   ├─ 訊息: {error}\n"
            f"   └─ 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )
        tb_summary = traceback.format_exception(exc_type, exc_value, exc_tb)
        print("   🔍 完整調用鏈追蹤 (Traceback):")
        for line in tb_summary[-3:]:  # 只印出最後 3 層最關鍵的堆疊
            print(f"      {line.strip()}")

    async def fetch_and_parse_horse(
        self, hook: Hook, horse_code: str
    ) -> Optional[Dict[str, Any]]:
        """單一馬匹資料請求與解析"""
        try:
            result = await hook.get_horse_tree(horse_code)

            if isinstance(result, Exception):
                self._log_exception(
                    f"[{horse_code}] 抓取馬匹 Profile Task 異常", result
                )
                return None

            if not result:
                print(
                    f"⚠️ [Horse Pipeline 警告] 無法取得馬匹 [{horse_code}] HTML 頁面"
                )
                return None

            tree, url = result
            parser = HorseProfileParser(tree, url)
            profile_data = parser.parse_horse_profile()

            if profile_data:
                profile_data["horse_code"] = horse_code
                return profile_data

            return None

        except Exception as e:
            self._log_exception(
                f"[{horse_code}] fetch_and_parse_horse 執行失敗", e
            )
            return None

    async def process_horse(self, hook: Hook, horse_code: str) -> None:
        """單一馬匹處理流程（含 Semaphore 與存檔檢查）"""
        async with self.semaphore:
            # 防重複爬取機制
            if self.horse_db.check_file_exist(horse_code):
                print(
                    f"⏩ [跳過] 馬匹代號 {horse_code} 本地 Profile 檔案已存在。"
                )
                return

            try:
                print(f"🐴 正在爬取馬匹資料: {horse_code}")
                profile_data = await self.fetch_and_parse_horse(
                    hook, horse_code
                )

                if profile_data:
                    await self.horse_db.save_normal_json(
                        horse_code, profile_data
                    )
                    print(f"✅ [成功] 馬匹 {horse_code} 資料已存檔。")
                else:
                    print(
                        f"⚠️ [Horse Pipeline 警告] 馬匹 {horse_code} 未能成功提取 valid profile。"
                    )

            except Exception as e:
                self._log_exception(
                    f"處理馬匹代號 [{horse_code}] 時發生嚴重錯誤", e
                )

    async def run(self, horse_codes: List[str]) -> None:
        """Pipeline 進入點：執行批量馬匹資料爬取

        :param horse_codes: 馬匹烙號/代號列表 (例: ['E123', 'G045', 'H112'])
        """
        try:
            print(f"📋 共收到 {len(horse_codes)} 個待處理馬匹代號。")

            if not horse_codes:
                print("❌ 沒有可執行的馬匹代號，程式結束。")
                return

            async with Hook() as hook:
                tasks = [
                    self.process_horse(hook, code) for code in horse_codes
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception(
                            "全域 process_horse 非同步任務發生錯誤", res
                        )

        except Exception as e:
            self._log_exception("HorsePipeline run 進入點發生嚴重崩潰", e)


if __name__ == "__main__":
    # 測試運行範例
    test_horse_codes = ["HK_2025_L397"]
    pipeline = HorseScrapingPipeline(max_concurrent=5)
    asyncio.run(pipeline.run(test_horse_codes))
```

---

### File: `scraper\race_pipeline.py`

```py
import asyncio
import sys
import traceback
from typing import List, Optional, Dict, Any, Tuple

from config.settings import settings
from scraper.hook import Hook
from scraper.parser.calander_parser import get_all_date_async
from scraper.parser.result_parser import ResultParser
from scraper.parser.sectional_parser import SectionalParser
from scraper.data_manager import DataManager


class RaceScrapingPipeline:
    """賽事爬蟲調度管道 (Pipeline Layer)"""

    def __init__(self, max_concurrent_days: int = 20):
        self.semaphore = asyncio.Semaphore(max_concurrent_days)
        self.races_db = DataManager(json_path=settings.raw_races_json_dir)
        self.sectional_db = DataManager(json_path=settings.raw_sectional_json_dir)

    def _log_exception(self, context_msg: str, error: Exception) -> None:
        """
        深層 Exception 紀錄器：自動擷取引發 Error 的最深層檔案、函式名稱與行號
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        
        # 追蹤 Traceback 到最深層的發源點
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "Unknown File"
        func_name = last_tb.tb_frame.f_code.co_name if last_tb else "Unknown Func"
        line_no = last_tb.tb_lineno if last_tb else "Unknown Line"

        print(
            f"\n💥 [Pipeline 崩潰] {context_msg}\n"
            f"   ├─ 類型: {exc_type.__name__ if exc_type else type(error).__name__}\n"
            f"   ├─ 訊息: {error}\n"
            f"   └─ 位置: {file_name} -> {func_name}() [Line {line_no}]"
        )
        # 印出精簡後的 Call Stack，方便追蹤最底層原因
        tb_summary = traceback.format_exception(exc_type, exc_value, exc_tb)
        print("   🔍 完整調用鏈追蹤 (Traceback):")
        for line in tb_summary[-3:]:  # 只印出最後 3 層最關鍵的堆疊
            print(f"      {line.strip()}")

    async def fetch_and_parse_race(
        self, hook: Hook, race_no: int, date_str: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        race_info: Optional[Dict[str, Any]] = None
        sectional_info: Optional[Dict[str, Any]] = None

        try:
            # 1. 並發請求基本賽果與分段時間頁面
            basic_task = hook.get_result_tree(race_no, date_str)
            sectional_task = hook.get_sectional_tree(
                race_no, f"{date_str[-2:]}/{date_str[5:7]}/{date_str[:4]}"
            )

            basic_result, sectional_result = await asyncio.gather(
                basic_task, sectional_task, return_exceptions=True
            )

            # 檢查並處理非同步任務中拋出的 Task 例外
            if isinstance(basic_result, Exception):
                self._log_exception(f"[{date_str} R{race_no}] 抓取基本賽果 Task 異常", basic_result)
            elif basic_result:
                basic_tree, basic_url = basic_result
                result_parser = ResultParser(basic_tree, basic_url)
                race_info = result_parser.parse_single_race(race_no)

            if isinstance(sectional_result, Exception):
                self._log_exception(f"[{date_str} R{race_no}] 抓取分段數據 Task 異常", sectional_result)
            elif sectional_result:
                sec_tree, sec_url = sectional_result
                sec_parser = SectionalParser(sec_tree, sec_url)
                sec_rows = sec_parser.parse_sectional_row()

                if sec_rows:
                    sectional_info = {
                        "race_no": race_no,
                        "sectional_data": sec_rows
                    }

            return race_info, sectional_info

        except Exception as e:
            self._log_exception(f"[{date_str} R{race_no}] fetch_and_parse_race 執行失敗", e)
            return None, None

    async def process_day(self, hook: Hook, date_str: str) -> None:
        async with self.semaphore:
            if self.races_db.check_file_exist(date_str) and self.sectional_db.check_file_exist(date_str, "race"):
                print(f"⏩ [跳過] 日期 {date_str} 本地賽果與分段檔案均已存在。")
                return

            try:
                init_result = await hook.get_result_tree("1", date_str)
                if not init_result:
                    print(f"⚠️ [Pipeline 警告] 無法取得 {date_str} 第一場數據 (可能是空頁面或網路請求失敗)")
                    return

                tree, url = init_result
                parser = ResultParser(tree, url)

                if parser.is_oversea():
                    print(f"🌍 [海外賽事] {date_str} 跳過處理。")
                    return

                total_races = parser.parse_race_length() or 0
                venue = (parser.parse_venue() or "")[:-1]

                print(f"🏇 正在爬取: {date_str} | 場地: {venue} | 共 {total_races} 場")

                if total_races == 0:
                    print(f"⚠️ [Pipeline 警告] {date_str} 判讀場次總數為 0，跳過該日處理。")
                    return

                race_tasks = [
                    self.fetch_and_parse_race(hook, i, date_str)
                    for i in range(1, total_races + 1)
                ]
                results = await asyncio.gather(*race_tasks, return_exceptions=True)

                races_data = []
                sectionals_data = []

                for idx, res in enumerate(results, 1):
                    if isinstance(res, Exception):
                        self._log_exception(f"[{date_str} R{idx}] 非同步子任務拋出未捕捉例外", res)
                    elif res:
                        r_info, s_info = res
                        if r_info is not None:
                            races_data.append(r_info)
                        if s_info is not None:
                            sectionals_data.append(s_info)

                race_payload = {
                    "date": date_str,
                    "venue": venue,
                    "races": races_data
                }

                sectional_payload = {
                    "date": date_str,
                    "venue": venue,
                    "sectionals": sectionals_data
                }

                await self.races_db.save_races_json(date_str, race_payload)
                await self.sectional_db.save_races_json(date_str, sectional_payload)

                print(f"✅ [成功] {date_str} 賽果與分段數據已分開存檔。")

            except Exception as e:
                self._log_exception(f"處理賽事日 [{date_str}] 時發生嚴重錯誤", e)

    async def run(self, years: List[int]) -> None:
        """Pipeline 進入點：執行批量日期下載"""
        try:
            target_dates = await get_all_date_async(*years)
            print(f"📅 共找到 {len(target_dates)} 個待處理賽事日期。")

            if not target_dates:
                print("❌ 沒有可執行的日期，程式結束。")
                return

            async with Hook() as hook:
                tasks = [self.process_day(hook, date_str) for date_str in target_dates]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        self._log_exception("全域 process_day 非同步任務發生錯誤", res)

        except Exception as e:
            self._log_exception("Pipeline run 進入點發生嚴重崩潰", e)
```

---

### File: `scraper\parser\__init__.py`

```py

```

---

### File: `scraper\parser\calander_parser.py`

```py
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
```

---

### File: `scraper\parser\horse_parser.py`

```py
import re
import sys
from typing import Dict, Optional
from selectolax.parser import HTMLParser


class HorseProfileParser:

    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def _log_exception(self, method_name: str, error: Exception) -> None:
        """統一例外 Log 輸出格式"""
        _, _, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
        print(f"💥 [ResultParser 錯誤] {method_name} (Line {line_no}): {error}")

    def parse_horse_profile(self) -> Optional[Dict[str, Optional[str]]]:
        horse_profile_params = {
            "origin_age": None,  # 出生地 / 馬齡 (例: 澳洲 / 5)
            "color_sex": None,  # 毛色 / 性別 (例: 栗 / 閹)
            "import_type": None,  # 進口類別 (例: 自購新馬)
            "season_stakes": None,  # 今季獎金*
            "total_stakes": None,  # 總獎金*
            "placing_records": None,  # 冠-亞-季-總出賽次數* (例: 1-7-4-27)
            "recent_10_races_count": None,  # 最近十個賽馬日出賽場數
            "current_location": None,  # 現在位置
            "location_arrival_date": None,  # 到達日期
            "import_date": None,  # 進口日期
            "trainer": None,  # 練馬師
            "owner": None,  # 馬主
            "current_rating": None,  # 現時評分
            "season_start_rating": None,  # 季初評分
            "sire": None,  # 父系
            "dam": None,  # 母系
            "damsire": None,  # 外祖父
        }

        # 精確對應清單 (確保不會誤觸「同父系馬」)
        field_keyword_map = {
            "出生地": "origin_age",
            "毛色": "color_sex",
            "進口類別": "import_type",
            "今季獎金": "season_stakes",
            "總獎金": "total_stakes",
            "冠-亞-季": "placing_records",
            "最近十個賽馬日": "recent_10_races_count",
            "練馬師": "trainer",
            "馬主": "owner",
            "現時評分": "current_rating",
            "季初評分": "season_start_rating",
            "外祖父": "damsire",  # 優先度高於「父系」
            "父系": "sire",
            "母系": "dam",
            "進口日期": "import_date",
        }

        try:
            tables = self.tree.css("table[class*='table_top_right']")
            if not tables:
                print(
                    f"⚠️ [parse_horse_profile] 找不到對應表格: {self.current_url}"
                )
                return horse_profile_params

            trs = [tr for table in tables for tr in table.css("tr")]

            for tr in trs:
                try:
                    tds = tr.css("td")
                    if not tds:
                        continue

                    # 雙欄佈局：每 3 個 td 為一組
                    for idx in range(0, len(tds), 3):
                        if idx + 2 >= len(tds):
                            break

                        label_text = tds[idx].text(strip=True)
                        val_text = tds[idx + 2].text(strip=True)

                        if not label_text:
                            continue

                        # 🛑 防護機制：完全跳過「同父系馬」選單欄位
                        if "同父系" in label_text:
                            continue

                        # 特殊處理：「現在位置 / 到達日期」
                        if "現在位置" in label_text and val_text:
                            match = re.search(
                                r"([^\s\(]+)(?:\s*\((.*?)\))?", val_text
                            )
                            if match:
                                horse_profile_params["current_location"] = (
                                    match.group(1)
                                )
                                if match.group(2):
                                    horse_profile_params[
                                        "location_arrival_date"
                                    ] = match.group(2)
                            else:
                                horse_profile_params["current_location"] = (
                                    val_text
                                )
                            continue

                        if not val_text:
                            continue

                        # 關鍵字匹配
                        for keyword, target_key in field_keyword_map.items():
                            if keyword in label_text:
                                horse_profile_params[target_key] = val_text
                                break

                except Exception as e:
                    self._log_exception("parse_horse_profile [Row Error]", e)

            return horse_profile_params

        except Exception as e:
            self._log_exception("parse_horse_profile (Global)", e)
            return horse_profile_params
```

---

### File: `scraper\parser\rating_parser.py`

```py
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
```

---

### File: `scraper\parser\result_parser.py`

```py
import re
import sys
from typing import Dict, List, Any, Optional
from selectolax.parser import HTMLParser


class ResultParser:
    def __init__(self, tree: HTMLParser, current_url: str = ""):
        self.tree = tree
        self.current_url = current_url

    def _log_exception(self, method_name: str, error: Exception) -> None:
        """統一例外 Log 輸出格式"""
        _, _, exc_tb = sys.exc_info()
        line_no = exc_tb.tb_lineno if exc_tb else "Unknown"
        print(f"💥 [ResultParser 錯誤] {method_name} (Line {line_no}): {error}")

    def parse_race_tab(self) -> Optional[Dict[str, Any]]:
        try:
            element = self.tree.css_first("div.race_tab")
            if not element:
                print("⚠️ [ResultParser 警告] parse_race_tab: 找不到 div.race_tab 元素")
                return None

            target_table = element.css_first("table")
            race_params = {
                "race_id": "",
                "basic_info": "",
                "track_condition": "",
                "track_info": "",
                "cumulative_finish_time": [],
                "sectional_finish_time": []
            }
            if not target_table:
                return race_params

            for row in target_table.css("tr"):
                cols = row.css("td")
                for idx, col in enumerate(cols):
                    text = col.text(strip=True)
                    if any(item in text for item in ["班", "關", "新"]):
                        race_params["basic_info"] = text
                    if "場地" in text and idx + 1 < len(cols):
                        race_params["track_condition"] = cols[idx + 1].text(strip=True)
                    if "賽道 :" in text and idx + 1 < len(cols):
                        race_params["track_info"] = cols[idx + 1].text(strip=True)
                    if "分段時間 :" in text:
                        siblings = cols[idx + 1:]
                        race_params["sectional_finish_time"] = [x.text(strip=True)[0:6] for x in siblings]
                    elif "時間 :" in text and "分段時間 :" not in text:
                        siblings = cols[idx + 1:]
                        race_params["cumulative_finish_time"] = [
                            re.sub(r"[()\[\]{}]", "", x.text(strip=True)) for x in siblings
                        ]
            return race_params

        except Exception as e:
            self._log_exception("parse_race_tab", e)
            return None

    def parse_results(self) -> Optional[List[Dict[str, Any]]]:
        try:
            element = self.tree.css_first('div[class*="performance"]')
            if element is None:
                element = self.tree.css_first('[class*="performance"]')

            if element is None:
                print("⚠️ [ResultParser 警告] parse_results: 未找到包含 performance 的元素")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            horses_params_list = []
            rows = target_table.css("tr")

            for row in rows[1:]:
                horse_params = {
                    "placing": None,          # 名次 (int / None)
                    "horse_no": None,         # 馬號/布號 (int)
                    "horse_name": "",         # 馬名 (str)
                    "jockey": "",             # 騎師 (str)
                    "trainer": "",            # 練馬師 (str)
                    "actual_weight": None,    # 實際負磅 (int / float)
                    "body_weight": None,      # 排位體重 (int / float)
                    "draw": None,             # 檔位 (int / None)
                    "margin": "",             # 與頭馬距離 / 勝負距離 (str)
                    "finish_time": "",        # 完成時間 (str)
                    "odds": None,              # 獨贏賠率 (float / None)
                    "horse_id": ""
                }
                key_list = list(horse_params.keys())
                cols = row.css("td")

                for i, col in enumerate(cols):
                    val = col.text(strip=True)
                    if i == 2:
                        a = col.css_first('a')
                        if a:
                            href = a.attributes.get('href')
                            horse_params["horse_id"] = href.split("=")[1]
                    if i != 9:
                        try:
                            val = float(val)
                        except ValueError:
                            pass

                        if i < 9:
                            horse_params[key_list[i]] = val
                        elif i > 9 and i <= len(key_list):
                            horse_params[key_list[i-1]] = val

                horses_params_list.append(horse_params)
            return horses_params_list

        except Exception as e:
            self._log_exception("parse_results", e)
            return None

    def parse_race_length(self) -> Optional[int]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("⚠️ [ResultParser 警告] parse_race_length: 找不到 div.top_races 元素")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            first_row = target_table.css_first("tr")
            if not first_row:
                return None

            all_td = first_row.css("td")
            empty_cnt = 0
            for td in all_td:
                # 檢查無文字且無子節點的空白排版格
                if not td.text(strip=True) and td.child is None:
                    empty_cnt += 1

            return len(all_td) - 2 - empty_cnt

        except Exception as e:
            self._log_exception("parse_race_length", e)
            return None

    def parse_venue(self) -> Optional[str]:
        try:
            element = self.tree.css_first("div.top_races")
            if not element:
                print("⚠️ [ResultParser 警告] parse_venue: 找不到 div.top_races 元素")
                return None

            target_table = element.css_first("table")
            if not target_table:
                return None

            first_row = target_table.css_first("tr")
            if not first_row:
                return None

            first_td = first_row.css_first("td")
            return first_td.text(strip=True) if first_td else ""

        except Exception as e:
            self._log_exception("parse_venue", e)
            return None

    def is_oversea(self) -> bool:
        try:
            if (
                not self.tree
                or "overseas" in self.current_url.lower()
                or self.tree.css_first("div#race_top_banner_container") is not None
                or self.tree.css_first("div.top_races") is None
            ):
                return True
            return False
        except Exception as e:
            self._log_exception("is_oversea", e)
            return True

    def parse_single_race(self, i: int) -> Dict[str, Any]:
        try:
            race_info = self.parse_race_tab() or {}
            race_performance = self.parse_results()
            race_info["race_id"] = i
            race_info["horses"] = race_performance
            return race_info
        except Exception as e:
            self._log_exception("parse_single_race", e)
            return {"race_id": i, "horses": None}
```

---

### File: `scraper\parser\sectional_parser.py`

```py
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
                    
```

---

