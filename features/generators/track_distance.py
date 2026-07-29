import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, TrackEncoder


class TrackDistanceGenerator:

    """生成場地、途程、檔位與條件組合特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        course_clean = TrackEncoder.categorize_course_type(
            df.get("track_type", pd.Series(index=df.index))
        )
        features["course_type"] = course_clean

        if all(col in df.columns for col in ["venue", "track_type", "draw"]):
            features["track_draw_key"] = TrackEncoder.create_track_draw_combo(
                df["venue"], df["track_type"], df["draw"]
            )

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")

        if "distance" in work_df.columns:
            features["horse_dist_win_rate"] = (
                BayesianSmoother.calc_global_smooth_rate(
                    work_df,
                    group_cols=["horse_id", "distance"],
                    target_col="is_win",
                    prior_alpha=2.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        if "venue" in work_df.columns:
            features["horse_venue_win_rate"] = (
                BayesianSmoother.calc_global_smooth_rate(
                    work_df,
                    group_cols=["horse_id", "venue"],
                    target_col="is_win",
                    prior_alpha=2.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features