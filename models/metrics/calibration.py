# models/metrics/calibration.py
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class OddsAwareCalibrator:
    """
    結合模型預測分數 (pred_score) 與市場隱含勝率 (implied_prob) 的機率校準器。
    使用二元 Logistic Regression 擬合模型優勢比 (Model Edge Ratio)。
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    def fit(self, pred_scores: np.ndarray, win_odds: np.ndarray, y_true: np.ndarray):
        """
        :param pred_scores: 模型的 raw predictions
        :param win_odds: 獨贏賠率
        :param y_true: 實際勝負 (1 或 0)
        """
        valid_mask = ~np.isnan(pred_scores) & ~np.isnan(win_odds) & (win_odds > 1.0)
        if np.sum(valid_mask) < 50:
            logger.warning("⚠️ 校準資料點過少，跳過擬合。")
            return self

        scores = pred_scores[valid_mask]
        odds = win_odds[valid_mask]
        y = y_true[valid_mask].astype(int)

        implied_p = 1.0 / odds
        # 建立特徵矩陣: [pred_score, implied_prob, log_odds]
        X = np.column_stack([scores, implied_p, np.log(odds)])

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("✅ Odds-Aware 機率校準器擬合完成！")
        except Exception as e:
            logger.error(f"❌ 機率校準擬合失敗: {e}")

        return self

    def predict_proba(self, pred_scores: np.ndarray, win_odds: np.ndarray) -> np.ndarray:
        """
        將 pred_score 與賠率轉換為對齊歷史勝率的概率
        """
        if not self.is_fitted:
            # Fallback: 使用隱含勝率
            odds = np.nan_to_num(win_odds, nan=99.0)
            return np.where(odds > 1.0, 1.0 / odds, 0.0)

        odds = np.nan_to_num(win_odds, nan=99.0)
        odds = np.maximum(odds, 1.01)
        implied_p = 1.0 / odds

        X = np.column_stack([np.nan_to_num(pred_scores, nan=-999.0), implied_p, np.log(odds)])
        probs = self.model.predict_proba(X)[:, 1]
        return probs