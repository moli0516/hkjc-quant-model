import os
import sys
import logging
import pandas as pd
import numpy as np
from config.settings import settings 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class PreTrainedBacktester:
    """
    專門用來評估「已訓練完成模型」的實戰回測器
    完全對接專案的 settings 系統，已全面升級支援 XGBRanker 與 舊Classifier
    """
    def __init__(self, model, target_col: str = "is_placed", features: list = None):
        """
        :param model: 已經訓練完成的模型實例 (支援 XGBClassifier 或 XGBRanker)
        :param target_col: 預測目標欄位，預設為 'is_placed'
        :param features: 特徵列表。如果為 None，則自動從 df 中偵測或回退至 settings
        """
        self.model = model
        self.target = target_col
        
        # 📌 關鍵修改：優先使用傳入的特徵清單，避免寫死 settings.latest_features
        if features is not None:
            self.features = features
        else:
            # 增加防禦性程式碼，防止 settings 裡沒有這個屬性時直接死掉
            self.features = getattr(settings, "latest_features", [])
            
        self.df_results = None
        logging.info(f"🛠️ 回測器初始化完成。對接特徵數: {len(self.features)} 個")

    def load_recent_test_data(self, start_date: str = "2025-07-01", end_date: str = "2026-08-01") -> pd.DataFrame:
        """
        直接使用 settings 中定義的特徵 Parquet 路徑載入測試數據
        """
        data_path = settings.features_parquet_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ settings 中指定的 features_parquet_path 檔案不存在：{data_path}")
            
        logging.info(f"💾 正在載入歷史大盤數據: {data_path} ...")
        df = pd.read_parquet(data_path)
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 切出測試區間
        mask = (df['date'] >= start_date) & (df['date'] < end_date)
        df_test = df[mask].copy()
        
        # 📌 關鍵修改：清除無效列時，如果 self.features 為空，則自動根據 df_test 現有的欄位
        check_cols = [c for c in self.features if c in df_test.columns] if self.features else []
        df_test = df_test.dropna(subset=check_cols + [self.target])
        
        # 按日期與場次嚴格排序（Ranker 機制的核心要求）
        df_test = df_test.sort_values(by=['date', 'race_unique_id']).reset_index(drop=True)
        
        logging.info(f"📊 測試集載入完成（區間: {start_date} ~ {end_date}）")
        logging.info(f"   └─ 總數據量: {len(df_test)} 筆 (共 {df_test['race_unique_id'].nunique()} 場賽事)")
        
        if len(df_test) == 0:
            raise ValueError(f"❌ 警告：切分後的測試集為空！")
            
        return df_test

    def run_backtest(self, df_test: pd.DataFrame) -> pd.DataFrame:
        """使用已訓練好的模型進行實戰命中率評估（自動相容多種模型接口與外部傳入得分）"""
        df_res = df_test.copy()
        
        # 🚀 【終極優化】：如果外部（例如 train.py 的搜尋管線）已經預測好了概率，直接跳過推理步驟！
        if 'pred_prob' in df_res.columns and 'pred_rank' in df_res.columns:
            logging.info("🔮 檢測到傳入的數據已包含 'pred_prob' 與 'pred_rank'，跳過重複推理，直接進行回測計算。")
        else:
            logging.info("🎯 開始對測試集進行模型推理 (Inference)...")
            X_test = df_res[self.features]
            
            # 🚀 【終極防呆防禦線】：檢查模型到底是 Classifier 還是 Ranker，自動分流！
            if hasattr(self.model, "predict_proba"):
                logging.info("🔮 檢測到二分類器，正在使用 predict_proba 預測絕對概率...")
                df_res['pred_prob'] = self.model.predict_proba(X_test)[:, 1]
            else:
                logging.info("🔮 檢測到排序模型 (XGBRanker)，正在執行場次內 Softmax 概率對齊...")
                # 1. 取得 Ranker 的 Margin Score
                raw_scores = self.model.predict(X_test)
                df_res['raw_score'] = raw_scores
                
                # 2. 定義場次內的 Softmax 歸一化（總期望值校準為 3.0）
                def race_level_softmax_to_probs(group):
                    exp_scores = np.exp(group['raw_score'] - np.max(group['raw_score']))
                    return (exp_scores / np.sum(exp_scores)) * 3

                # 3. 套用歸一化
                df_res['pred_prob'] = df_res.groupby('race_unique_id', sort=False).apply(
                    lambda x: pd.Series(race_level_softmax_to_probs(x), index=x.index)
                ).reset_index(level=0, drop=True)
            
            # 4. 在每場比賽中按預測機率重新進行嚴格排序（1 為最高機率）
            df_res['pred_rank'] = df_res.groupby('race_unique_id')['pred_prob'].rank(ascending=False, method='first')
        
        # 5. 指標計算
        total_races = df_res['race_unique_id'].nunique()
        
        # Top 1 獨膽上名率 (位置命中率)
        top1_df = df_res[df_res['pred_rank'] == 1]
        hit_top3_count = top1_df[self.target].sum()
        precision_at_1 = hit_top3_count / total_races if total_races > 0 else 0
        
        # Top 3 複式平均上名數
        top3_df = df_res[df_res['pred_rank'] <= 3]
        total_hits_in_top3 = top3_df[self.target].sum()
        avg_hits_per_race = total_hits_in_top3 / total_races if total_races > 0 else 0
        
        # 使用 logger 改寫原本的 print，使其符合你 train.py 的日誌配置
        logger = logging.getLogger("PreTrainedBacktester")
        logger.info(f"🔹 測試場次總數: {total_races} 場")
        logger.info(f"🔹 模型獨膽（Top 1）位置命中率: {precision_at_1:.2%}")
        logger.info(f"🔹 平均每場前三熱門上名數: {avg_hits_per_race:.2f} 匹 (最高 3.0)")
        
        self.df_results = df_res
        return df_res

    def save_predictions(self, filename: str = "oot_july_predictions.parquet"):
        """將預測結果儲存在與特徵 Parquet 相同的資料夾下"""
        if self.df_results is not None:
            output_path = settings.features_parquet_path.parent / filename
            self.df_results.to_parquet(output_path)
            logging.info(f"💾 預測明細已儲存至: {output_path}")
        else:
            logging.warning("⚠️ 沒有可儲存的預測結果。")