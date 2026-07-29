import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, SpeedTimeCalculator


class SectionalSpeedGenerator:

    """生成分段時間、速段爆發與卡位能力特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "distance" in df.columns and "finish_time_sec" in df.columns:
            features["speed_mps_overall"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    df["distance"], df["finish_time_sec"]
                ).astype("float32")
            )

        if "sectional_time_last" in df.columns:
            features["speed_mps_last_sectional"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    pd.Series(400.0, index=df.index), df["sectional_time_last"]
                ).astype("float32")
            )

        if "pos_c1" in df.columns and "pos_c4" in df.columns:
            features["position_gain_c1_to_c4"] = (
                df["pos_c1"] - df["pos_c4"]
            ).astype("float32")

        work_df = df.sort_values(["horse_id", "date"]).copy()
        if "speed_mps_last_sectional" in features.columns:
            work_df["speed_mps_last_sectional"] = features[
                "speed_mps_last_sectional"
            ]
            features["horse_rolling_last_sec_speed_mean_3"] = (
                BayesianSmoother.calc_rolling_stat(
                    work_df,
                    group_cols="horse_id",
                    value_col="speed_mps_last_sectional",
                    window_size=3,
                    stat_type="mean",
                )
                .reindex(df.index)
                .astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features