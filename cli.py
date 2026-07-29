import argparse
import asyncio
import gc
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

    def run_features_pipeline(self, chunk_race_days: int = 360):
        """分批次執行特徵工程，避免龐大數據導致 RAM 爆滿"""
        print(
            "⚙️ [Step 5] 開始執行：分批次量化特徵工程 Pipeline..."
        )
        if not self.check_db_has_races():
            return

        try:
            all_dates = self.db.get_all_race_dates()
            if not all_dates:
                print("⚠️ 未找到任何賽事日期記錄。")
                return

            total_days = len(all_dates)
            print(
                f"📅 歷史賽事總計 {total_days} 個賽事日 ({all_dates[0]} ~ {all_dates[-1]})"
            )

            date_chunks = [
                all_dates[i : i + chunk_race_days]
                for i in range(0, total_days, chunk_race_days)
            ]

            pipeline = FeaturesPipeline(key_cols=["race_id", "horse_id"])
            total_chunks = len(date_chunks)

            for idx, chunk in enumerate(date_chunks, start=1):
                start_date, end_date = chunk[0], chunk[-1]
                print(
                    f"🔄 [{idx}/{total_chunks}] 正在計算特徵 ({start_date} 至 {end_date})..."
                )

                raw_df = self.db.load_merged_race_data_by_dates(
                    start_date, end_date
                )
                if raw_df.empty:
                    continue

                feature_df = pipeline.run(df=raw_df)

                if_exists_mode = "replace" if idx == 1 else "append"
                self.db.save_feature_matrix(
                    df=feature_df,
                    table_name="feature_matrix",
                    if_exists=if_exists_mode,
                )

                print(
                    f"   └─ ✅ Chunk {idx} 已寫入 (包含 {len(feature_df)} 筆數據)"
                )

                del raw_df, feature_df
                gc.collect()

            print("✅ 所有批次特徵工程計算完成，已成功寫入資料庫！\n")

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

            # 調用 ModelPipeline 中的正確方法 run_train_pipeline
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

            # 調用 ModelPipeline 中的正確推論方法
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
            print("=" * 50)
            print("🏇  HKJC 賽馬數據工程與機器學習模型 CLI 工具")
            print("=" * 50)
            print("1. 執行賽果和分段時間爬蟲")
            print("2. 進行賽果和分段時間數據清洗")
            print("3. 執行馬匹資料爬蟲 (需要先有賽果資料庫)")
            print("4. 進行馬匹資料數據清洗")
            print("5. ⚙️  生成量化特徵矩陣 (Features Pipeline)")
            print("6. 🤖 訓練賽馬預測模型 (Model Pipeline)")
            print("7. 🔮 執行賽事勝率預測 (Inference)")
            print("8. ⚡ 執行一鍵全套 ETL + 特徵工程 + 模型訓練 (1 ➔ 6)")
            print("0. 退出系統")
            print("=" * 50)

            choice = input("請選擇要執行的功能 (0-8): ").strip()

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
            elif choice == "7":
                date_input = (
                    input(
                        "輸入預測日期 (YYYY-MM-DD，留空則預測最新賽事): "
                    ).strip()
                    or None
                )
                self.run_predictions(target_date=date_input)
            elif choice == "8":
                print("\n🔄 開始一鍵執行全套 Pipeline (從爬蟲到模型訓練)...")
                self.run_race_scraper()
                self.run_race_cleaner()
                if self.check_db_has_races():
                    self.run_horse_scraper()
                    self.run_horse_cleaner()
                    self.run_features_pipeline()
                    self.run_model_pipeline()
            elif choice == "0":
                print("👋 已退出 CLI 工具。")
                break
            else:
                print("❌ 無效選擇，請重新輸入！\n")


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

    if args.train_model:
        cli.run_model_pipeline(model_type=args.model_type)

    if args.predict:
        cli.run_predictions(target_date=args.start_date)


if __name__ == "__main__":
    main()