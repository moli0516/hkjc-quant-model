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