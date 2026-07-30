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