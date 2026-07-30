import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class TrackDistanceGenerator:
    """路程與場地歷史特徵生成器"""

    EXECUTION_ORDER = 80

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "date" if "date" in df.columns else ("race_date" if "race_date" in df.columns else None)
        if date_col is None or "placing" not in df.columns:
            features["horse_dist_win_rate"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", date_col]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")

        if "distance" in work_df.columns:
            dist_smooth = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols=["horse_id", "distance"], target_col="is_win", prior_alpha=2.0, baseline_rate=0.08
            )
            features["horse_dist_win_rate"] = dist_smooth.reindex(df.index).fillna(0.0).astype("float32")

        if "track" in work_df.columns or "track_type" in work_df.columns:
            t_col = "track" if "track" in work_df.columns else "track_type"
            track_smooth = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols=["horse_id", t_col], target_col="is_win", prior_alpha=2.0, baseline_rate=0.08
            )
            features["horse_track_win_rate"] = track_smooth.reindex(df.index).fillna(0.0).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features