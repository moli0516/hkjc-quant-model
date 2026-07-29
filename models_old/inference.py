import xgboost as xgb
from config.settings import settings
import logging
import pandas as pd
import numpy as np

class Quant_inference_engine:
    def __init__(self, features, target, settings_obj = settings):
        self.settings = settings_obj
        self.features = features
        self.model = None
        self.target = target  # 💡 傳入的是「是否上名（1, 2, 3 名為 1，其餘為 0）」的欄位
        self.init_model()
    
    def init_model(self, custom_params=None):
        # 🚀 升級點 1：將 Classifier 換成更貼合賽馬情境的 XGBRanker (Learning to Rank)
        logging.info("⚙️ 正在初始化 XGBRanker 排序模型參數...")
        base_params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg",
            "max_depth": 4,
            "learning_rate": 0.01,
            "n_estimators": 1500,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "reg_alpha": 1,
            "reg_lambda": 20,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 40,
            # 💡 強迫模型專注優化前 3 名的排序
            "lambdarank_pair_method": "topk",
            "lambdarank_num_by_group": 3
        }
        
        # 如果有 Optuna 傳入的參數，自動覆蓋
        if custom_params:
            base_params.update(custom_params)
            
        self.model = xgb.XGBRanker(**base_params)
        logging.info("完成初始化 XGBRanker 模型參數")
    
    def train_model(self, df_hist: pd.DataFrame, val_day=30):
        # 確保目標值與特徵欄位無缺失值
        df_clean = df_hist.dropna(subset=self.features + [self.target]).copy()
        df_clean['date'] = pd.to_datetime(df_clean["date"])
        
        # 🚀 升級點 2：Ranker 要求同一個群組（場次）的數據必須連續。
        # 這裡嚴格按照時間與場次識別碼進行排序
        df_clean = df_clean.sort_values(by=["date", "race_unique_id"]).reset_index(drop=True)
        
        # 計算群組數以利 Log 觀察與劃分
        df_clean["qid"] = df_clean.groupby("race_unique_id", sort=False).ngroup()

        split_date = df_clean['date'].max() - pd.Timedelta(days=val_day)
        train_data = df_clean[df_clean['date'] <= split_date].copy()
        val_data = df_clean[(df_clean['date'] > split_date)].copy()
        
        X_train = train_data[self.features]
        y_train = train_data[self.target]
        
        # 🚀 升級點 3：計算訓練集與驗證集每場賽事各有多少匹馬 (XGBRanker 核心參數: group)
        train_groups = train_data.groupby("race_unique_id", sort=False).size().values
        val_groups = val_data.groupby("race_unique_id", sort=False).size().values
        
        logging.info(f"📊 資料集劃分完成：訓練集群組數 {len(train_groups)}，驗證集群組數 {len(val_groups)}")
        
        X_val = val_data[self.features]
        y_val = val_data[self.target]
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("❌ 訓練集或驗證集數據為空，請檢查 val_days 劃分或歷史數據量。")
            
        logging.info("🚀 正在訓練 XGBRanker 並開啟早停機制...")
        
        # 🚀 升級點 4：將 group 傳入 fit 函數
        self.model.fit(
            X = X_train,
            y = y_train,
            group = train_groups,
            eval_set = [(X_val, y_val)],
            eval_group = [val_groups],
            verbose = 25                         # 每 25 代輸出一次評估日誌
        )
        logging.info("✅ XGBRanker 訓練完成！")
        
        feature_importance = dict(zip(self.features, self.model.feature_importances_))
        return {
            "status": "success",
            "best_iteration": self.model.best_iteration,
            "best_score": self.model.best_score,
            "feature_importance": feature_importance 
        }
        
    def predict(self, X_race, race_ids):
        """
        🚀 升級點 5：預測函數重構
        X_race: 特徵矩陣
        race_ids: 對應每筆排位數據的 race_unique_id 數組/列表（用來做場次內 Softmax 歸一化）
        """
        if self.model is None:
            raise ValueError("❌ 模型尚未訓練，請先執行 train_model()。")
            
        logging.info("🔮 正在對排位進行群組相對得分預測...")
        
        # XGBRanker 的 predict 輸出的是相對的 Margin Score (有正有負)
        raw_scores = self.model.predict(X_race)
        
        # 建立臨時 DataFrame 以便利用 groupby 進行場次內的 Softmax 歸一化
        predict_df = pd.DataFrame({
            'race_unique_id': race_ids,
            'raw_score': raw_scores
        })
        
        def race_level_softmax_to_probs(group):
            # 為了數值穩定性，減去群組最大值 (防止指數爆炸)
            exp_scores = np.exp(group['raw_score'] - np.max(group['raw_score']))
            softmax_probs = exp_scores / np.sum(exp_scores)
            # 💡 一場賽事必定有 3 匹馬跑入位置。將概率總和歸一化調整為 3，
            # 這樣算出來的數值就直接代表該馬匹在該場比賽的『真實位置概率期望值』！
            return softmax_probs * 3

        logging.info("🧮 正在進行場次內群組 Softmax 概率歸一化（總期望值校準為 3.0）...")
        calibrated_probs = predict_df.groupby('race_unique_id', sort=False).apply(
            lambda x: pd.Series(race_level_softmax_to_probs(x), index=x.index)
        ).reset_index(level=0, drop=True)
        
        # 返回一個數值介於 0~1 之間、且跨場次完美對齊的位置概率數組
        return calibrated_probs.values
        
    def save_model(self, filepath: str):
        if self.model:
            self.model.save_model(filepath)
            logging.info(f"💾 模型已成功儲存至 {filepath}")
            
    def load_model(self, filepath: str):
        self.init_model()
        self.model.load_model(filepath)
        logging.info(f"📂 模型已成功自 {filepath} 載入")