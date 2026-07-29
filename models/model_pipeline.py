import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from database.db_manager import DBManager
from models.data_loader import RaceDataLoader
from models.validation.time_split import TimeSeriesSplitter
from models.registry import ModelRegistry
from models.metrics.ranking import RankingMetrics

import models.wrappers.xgb_wrapper

logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    賽馬機器學習統籌工作流 (Model Pipeline)
    負責將資料載入、時間切分、模型訓練、評估與推論串聯成標準化流程。
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
        feature_cols: Optional[list] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        執行完整的訓練與驗證 Pipeline
        """
        logger.info("🚀 開始執行訓練 Pipeline...")

        # 1. 載入並自動預處理數據 (透過 RaceDataLoader 的 load_dataset)
        df, default_feature_cols, _ = self.data_loader.load_dataset(include_odds=True)
        
        if df.empty:
            raise ValueError("【錯誤】訓練資料集為空，無法進行訓練！")

        # 🔧【修正重點】自動相容與對齊日期欄位 (防止 date 欄位缺失)
        # models/model_pipeline.py

# 1. 解析日期欄位
# models/model_pipeline.py

# 1. 解析日期：支援 YYYY/MM/DD 或 YYYY-MM-DD
        if "date" not in df.columns:
            if "race_date" in df.columns:
                df["date"] = df["race_date"]
            else:
                # 正則匹配：4位數字 + [/或-] + 2位數字 + [/或-] + 2位數字
                df["date"] = df["race_id"].astype(str).str.extract(r'(\d{4}[/-]\d{2}[/-]\d{2})')[0]

        # 2. 轉為標準 datetime 格式 (pandas 可以自動識別 2020/01/01)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 3. 清理與驗證
        initial_len = len(df)
        df = df.dropna(subset=["date"]).copy()
        cleaned_len = len(df)

        if cleaned_len == 0:
            raise ValueError("【錯誤】無法從 race_id 解析出任何有效日期！")

        logger.info(f"💡 成功解析 {cleaned_len} 筆賽事日期 (日期範圍: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')})")

            
        # 若使用者未自訂 feature_cols，則使用 DataLoader 自動提取的預設特徵
        if feature_cols is None:
            feature_cols = list(default_feature_cols)

        # 定義禁止傳入模型的「未來官子/識別符」欄位
        forbidden_cols = {
            'placing',
            'is_win',
            'is_top3',
            'relevance_score',
            'date',
            'race_date',
            'race_id',
            'horse_id',
        }

        # 🔧 修正重點：不論 feature_cols 是自動取得還是外部傳入，強制過濾禁用的欄位
        feature_cols = [col for col in feature_cols if col not in forbidden_cols]

        if not feature_cols:
            raise ValueError("【錯誤】經過禁用的欄位過濾後，有效特徵數為 0，無法進行訓練！")

        # 確保訓練資料中包含模型所需的 relevance_score 標籤 (若沒有則用 placing 反向衍生)
        if "relevance_score" not in df.columns:
            if "placing" in df.columns:
                # 範例：第1名=3分, 第2名=2分, 第3名=1分, 其餘=0分
                df["relevance_score"] = df["placing"].apply(lambda p: max(0, 4 - p) if p <= 3 else 0)
            else:
                raise KeyError("【錯誤】資料庫中缺乏 'placing' 欄位，無法計算 'relevance_score' 目標標籤！")

        # 確保訓練資料中包含模型所需的 relevance_score 標籤 (若沒有則用 placing 反向衍生)
        if "relevance_score" not in df.columns and "placing" in df.columns:
            # 範例：第1名=3分, 第2名=2分, 第3名=1分, 其餘=0分
            df["relevance_score"] = df["placing"].apply(lambda p: max(0, 4 - p) if p <= 3 else 0)

        logger.info(f"📊 總樣本數: {len(df)}, 有效特徵數: {len(feature_cols)}")

        # 2. 時間序列切分 (防止資料洩漏)
        train_df, val_df, train_groups, val_groups = self.splitter.split_by_days(df, val_days=val_days)

        # 3. 創建模型實例
        model = ModelRegistry.create(name=model_name, model_params=model_params)

        # 4. 執行模型訓練 (含驗證集與 Early Stopping)
        model.fit(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="relevance_score",
            groups=train_groups,
            eval_set=(val_df, feature_cols, "relevance_score"),
            eval_groups=val_groups
        )

        # 5. 模型預測與評估
        logger.info("📈 正在計算驗證集評估指標...")
        val_preds = model.predict(val_df)
        val_df_evaluated = val_df.copy()
        val_df_evaluated["pred_score"] = val_preds

        top1_win_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated, pred_score_col="pred_score", target_placing_col="placing", group_col="race_id", k=1
        )
        top3_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated, pred_score_col="pred_score", target_placing_col="placing", group_col="race_id", k=3
        )
        ndcg = RankingMetrics.mean_ndcg_score(
            val_df_evaluated, pred_score_col="pred_score", target_relevance_col="relevance_score", group_col="race_id", k=5
        )

        metrics = {
            "top1_win_rate": top1_win_rate,
            "top3_rate": top3_rate,
            "ndcg@5": ndcg
        }

        logger.info(f"🎯 驗證結果指標: {metrics}")
        return model, metrics

    def run_inference_pipeline(
        self,
        model: Any,
        inference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        執行推論 Pipeline：對給定的最新賽事特徵進行預測評分
        
        :param model: 已訓練好的模型實例
        :param inference_df: 包含特徵的最新賽事 DataFrame
        :return: 帶有預測得分與每場比賽排序的 DataFrame
        """
        logger.info("🔮 開始執行推論 (Inference) Pipeline...")
        
        preds = model.predict(inference_df)
        result_df = inference_df.copy()
        result_df["pred_score"] = preds

        # 依照賽事 (race_id) 內部對 pred_score 進行排名 (1 代表該場比賽預測第一名)
        result_df["pred_rank"] = result_df.groupby("race_id")["pred_score"].rank(ascending=False, method="min")
        
        logger.info("✅ 推論完成！")
        return result_df