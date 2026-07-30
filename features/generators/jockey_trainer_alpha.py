import numpy as np
import pandas as pd
import re
from features.utils import BayesianSmoother, LeakageGuard


class JockeyTrainerAlphaGenerator:
    """騎練動態 Alpha 特徵生成器 (Jockey & Trainer Dynamics Alpha Generator)"""

    EXECUTION_ORDER = 65

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    @staticmethod
    def _is_class_1_to_5(race_class_series: pd.Series) -> pd.Series:
        def check_valid(val):
            if pd.isna(val):
                return False
            s = str(val).upper().strip()
            if any(kw in s for kw in ["G1", "G2", "G3", "GROUP", "HKG"]):
                return False
            match = re.search(r"(\d+)", s)
            if match:
                c = float(match.group(1))
                return 1.0 <= c <= 5.0
            return False

        return race_class_series.apply(check_valid)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        required_cols = ["jockey", "trainer", "date", "placing"]
        if not all(col in df.columns for col in required_cols):
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "race_class" in work_df.columns:
            work_df["is_target_class"] = self._is_class_1_to_5(
                work_df["race_class"]
            )
        else:
            work_df["is_target_class"] = True

        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        work_df["jt_combo"] = (
            work_df["jockey"].astype(str)
            + "_"
            + work_df["trainer"].astype(str)
        )

        jt_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jt_combo"],
            target_col="is_win",
            prior_alpha=3.0,
            baseline_rate=0.08,
        )

        jt_top3_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jt_combo"],
            target_col="is_top3",
            prior_alpha=3.0,
            baseline_rate=0.24,
        )

        jockey_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["jockey"],
            target_col="is_win",
            prior_alpha=5.0,
            baseline_rate=0.08,
        )

        jt_win_alpha = jt_win_rate - jockey_win_rate

        prev_jockey = work_df.groupby("horse_id")["jockey"].shift(1)
        work_df["is_jockey_switched"] = (
            (work_df["jockey"] != prev_jockey) & (prev_jockey.notna())
        ).astype("float32")

        prev_jockey_win_rate = (
            work_df.groupby("horse_id")["jockey"]
            .shift(1)
            .map(jockey_win_rate)
        )
        work_df["jockey_upgrade_alpha"] = np.where(
            work_df["is_jockey_switched"] == 1.0,
            jockey_win_rate - prev_jockey_win_rate,
            0.0,
        )

        target_mask = work_df["is_target_class"]

        features["alpha_jt_combo_win_rate"] = np.where(
            target_mask, jt_win_rate, np.nan
        )
        features["alpha_jt_combo_top3_rate"] = np.where(
            target_mask, jt_top3_rate, np.nan
        )
        features["alpha_jt_synergy_alpha"] = np.where(
            target_mask, jt_win_alpha, np.nan
        )
        features["alpha_is_jockey_switched"] = np.where(
            target_mask, work_df["is_jockey_switched"], np.nan
        )
        features["alpha_jockey_upgrade_alpha"] = np.where(
            target_mask, work_df["jockey_upgrade_alpha"], np.nan
        )

        features = features.reindex(df.index)

        for col in features.columns:
            if col not in self.key_cols:
                features[col] = features[col].astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)

        return features