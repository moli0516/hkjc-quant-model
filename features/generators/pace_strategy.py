import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class PaceStrategyGenerator:

    """生成馬匹領放跑法特徵與同場 Race Pace 步速壓力。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "section_no" in df.columns and "position" in df.columns:
            sec1_df = df[df["section_no"] == 1].sort_values(
                ["horse_id", "date"]
            )

            avg_sec1_pos = BayesianSmoother.calc_rolling_stat(
                sec1_df,
                group_cols="horse_id",
                value_col="position",
                window_size=3,
                stat_type="mean",
            )
            features["horse_avg_sec1_pos_3"] = (
                avg_sec1_pos.reindex(df.index).fillna(8.0).astype("float32")
            )

            features["is_front_runner"] = (
                features["horse_avg_sec1_pos_3"] <= 3.0
            ).astype("float32")

            df_temp = df[self.key_cols].copy()
            df_temp["is_front"] = features["is_front_runner"]
            features["race_front_runner_count"] = (
                df_temp.groupby("race_id")["is_front"]
                .transform("sum")
                .astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features