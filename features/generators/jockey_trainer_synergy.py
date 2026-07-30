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