import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, SpeedTimeCalculator


class SectionalBurstGenerator:
    """末腳衝刺爆發力與速度比率特徵生成器"""

    EXECUTION_ORDER = 32

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 檢查是否有末段分段時間
        sec_cols = [
            c for c in ["sec1_time", "sec2_time", "sec3_time", "sec4_time", "sec5_time", "sec6_time"]
            if c in df.columns
        ]

        if not sec_cols or "finish_time_sec" not in df.columns or "distance" not in df.columns:
            features["burst_ratio_last_sec"] = 0.0
            features["horse_rolling_burst_ratio_3"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        # 全場平均速度 (m/s)
        overall_speed = SpeedTimeCalculator.calc_speed_mps(
            df["distance"], df["finish_time_sec"]
        )

        # 取最後一段 400m 時間
        last_sec_time = df[sec_cols].ffill(axis=1).iloc[:, -1]
        last_sec_speed = SpeedTimeCalculator.calc_speed_mps(
            pd.Series(400.0, index=df.index), last_sec_time
        )

        # 1. 爆發力指標 (末段速度 / 全場平均速度) -> >1.0 代表末段加速能力強
        burst_ratio = last_sec_speed / (overall_speed + 1e-6)
        features["burst_ratio_last_sec"] = burst_ratio.fillna(0.0).astype("float32")

        # 2. 歷史近 3 場的平均末腳爆發比率 (Rolling 滾動計算，嚴格防洩漏)
        if "date" in df.columns and "horse_id" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            work_df["_burst_ratio"] = features.loc[work_df.index, "burst_ratio_last_sec"]

            rolling_burst = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="_burst_ratio",
                window=3,
                stat="mean",
            )
            features["horse_rolling_burst_ratio_3"] = (
                rolling_burst.reindex(df.index).fillna(0.0).astype("float32")
            )
        else:
            features["horse_rolling_burst_ratio_3"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features