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