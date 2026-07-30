import pandas as pd
import numpy as np
from features.utils import BayesianSmoother, LeakageGuard


class PaceStrategyGenerator:
    """跑法與賽事步速競爭特徵生成器"""

    EXECUTION_ORDER = 40

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        pos_series = None
        for col in ["pos_sec1", "running_position_avg", "position_1"]:
            if col in df.columns:
                pos_series = pd.to_numeric(df[col], errors="coerce")
                break

        if pos_series is None:
            features["is_front_runner"] = 0.0
            features["race_front_runner_count"] = 0.0
            features["horse_avg_sec1_pos_3"] = np.nan
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy() if "date" in df.columns else df.copy()
        work_df["_pos_clean"] = pos_series.reindex(work_df.index)
        
        rolling_pos = BayesianSmoother.calc_rolling_stat(
            work_df,
            group_cols="horse_id",
            value_col="_pos_clean",
            window_size=3,
            stat_type="mean"
        )
        features["horse_avg_sec1_pos_3"] = rolling_pos.reindex(df.index).astype("float32")

        is_front = (features["horse_avg_sec1_pos_3"].fillna(pos_series) <= 3.5).astype("float32")
        features["is_front_runner"] = is_front.reindex(df.index).astype("float32")

        df_temp = pd.DataFrame({"race_id": df["race_id"], "is_front": features["is_front_runner"]}, index=df.index)
        features["race_front_runner_count"] = (
            df_temp.groupby("race_id")["is_front"]
            .transform("sum")
            .astype("float32")
        )
        features["is_front_runner_race_front_runner_count_interaction"] = (features["is_front_runner"] * features["race_front_runner_count"]).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features