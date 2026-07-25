import os
import sys
import logging
import pandas as pd
from xgboost import XGBClassifier
from config.settings import settings 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PreTrainedBacktester:
    def __init__(self, model, target_col: str = "is_win"): # 這裡改為 "is_win"
        self.model = model
        self.target = target_col
        self.features = settings.latest_features
        self.df_results = None
        logging.info(f"🛠️ 勝率模型回測器初始化完成。目標欄位: {self.target}")
    def load_recent_test_data(self, start_date: str = "2025-07-01", end_date: str = "2026-08-01") -> pd.DataFrame:
        """
        直接使用 settings 中定義的特徵 Parquet 路徑載入 2025-26 數據
        """
        # 📌 自動從 settings 讀取 Parquet 路徑
        data_path = settings.features_parquet_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ settings 中指定的 features_parquet_path 檔案不存在：{data_path}")
            
        logging.info(f"💾 正在載入歷史大盤數據: {data_path} ...")
        df = pd.read_parquet(data_path)
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 切出 2026-07 的測試區間
        mask = (df['date'] >= start_date) & (df['date'] < end_date)
        df_test = df[mask].copy()
        
        # 清除特徵或 Target 含有 NaN 的無效列
        df_test = df_test.dropna(subset=self.features + [self.target])
        df_test = df_test.sort_values('date').reset_index(drop=True)
        
        logging.info(f"📊 測試集載入完成（區間: {start_date} ~ {end_date}）")
        logging.info(f"   └─ 總數據量: {len(df_test)} 筆 (共 {df_test['race_unique_id'].nunique()} 場賽事)")
        
        if len(df_test) == 0:
            raise ValueError(f"❌ 警告：切分後的測試集為空！請確認該 Parquet 是否包含 {start_date} 的數據。")
            
        return df_test
    def run_backtest(self, df_test: pd.DataFrame) -> pd.DataFrame:
        logging.info("🎯 開始對勝率模型進行推理 (Inference)...")
        
        X_test = df_test[self.features]
        df_res = df_test.copy()
        
        # 1. 預測勝率概率
        df_res['pred_prob'] = self.model.predict_proba(X_test)[:, 1]
        
        # 2. 排序
        df_res['pred_rank'] = df_res.groupby('race_unique_id')['pred_prob'].rank(ascending=False, method='first')
        
        # 3. 勝率模型專用指標計算
        total_races = df_res['race_unique_id'].nunique()
        
        # Top 1 獨贏命中率 (Win Strike Rate)
        top1_df = df_res[df_res['pred_rank'] == 1]
        win_hits = top1_df[self.target].sum() # 這裡是 is_win == 1 的數量
        win_rate = win_hits / total_races if total_races > 0 else 0
        
        # 覆蓋率 (Coverage): 模型預測機率高於一定閾值的場次比例 (選填，視需要加入)
        high_conf_hits = top1_df[top1_df['pred_prob'] > 0.3][self.target].sum()
        high_conf_count = len(top1_df[top1_df['pred_prob'] > 0.3])
        precision_high_conf = high_conf_hits / high_conf_count if high_conf_count > 0 else 0
        
        print("\n==================== 🏆 25-26 勝率模型回測報告 ====================")
        print(f"🔹 測試場次總數: {total_races} 場")
        print(f"🔹 獨贏命中率 (Win Rate): {win_rate:.2%}")
        print(f"🔹 高信心區間 (>30%) 獨贏精準度: {precision_high_conf:.2%} ({high_conf_hits}/{high_conf_count})")
        print("=====================================================================")
        
        self.df_results = df_res
        return df_res

    def save_predictions(self, filename: str = "oot_july_predictions.parquet"):
        """將預測結果儲存在與特徵 Parquet 相同的資料夾下"""
        if self.df_results is not None:
            # 📌 儲存在與大盤 Parquet 相同的目錄中
            output_path = settings.features_parquet_path.parent / filename
            self.df_results.to_parquet(output_path)
            logging.info(f"💾 預測明細已儲存至: {output_path}")
        else:
            logging.warning("⚠️ 沒有可儲存的預測結果。")