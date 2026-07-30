import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, RaceScaler


class SpeedFeatureGenerator:
    """標準化速度指數與分段衝刺特徵生成器"""

    EXECUTION_ORDER = 35

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 同場標準化完賽時間 Z-Score
        if "finish_time_sec" in df.columns and "race_id" in df.columns:
            temp_df = pd.DataFrame(
                {
                    "race_id": df["race_id"],
                    "_neg_finish_time": df["finish_time_sec"] * -1.0,
                },
                index=df.index,
            )
            features["finish_time_race_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_neg_finish_time"
            ).astype("float32")

        # 2. 同場末腳衝刺速度 Z-Score
        last_sec_val = None
        if "speed_mps_last_sectional" in df.columns:
            last_sec_val = df["speed_mps_last_sectional"]
        elif "sectional_time_last" in df.columns:
            last_sec_val = df["sectional_time_last"] * -1.0

        if last_sec_val is not None and "race_id" in df.columns:
            temp_df = pd.DataFrame(
                {"race_id": df["race_id"], "_last_sec_val": last_sec_val},
                index=df.index,
            )
            features["last_400m_speed_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_last_sec_val"
            ).astype("float32")

        # 3. 同場早段搶放體力消耗 Z-Score
        if "sec1_time" in df.columns and "race_id" in df.columns:
            sec1_num = pd.to_numeric(df["sec1_time"], errors="coerce")
            temp_df = pd.DataFrame(
                {"race_id": df["race_id"], "_neg_sec1": sec1_num * -1.0},
                index=df.index,
            )
            features["early_pace_expenditure_z"] = RaceScaler.race_z_score(
                temp_df, race_col="race_id", value_col="_neg_sec1"
            ).astype("float32")

        # 4. 馬匹歷史近 5 場 Speed Z-Score 滾動平均
        if "finish_time_race_z" in features.columns and "date" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            work_df["finish_time_race_z"] = features.loc[
                work_df.index, "finish_time_race_z"
            ]

            rolling_speed_z = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="finish_time_race_z",
                window=5,
                stat="mean",
            )
            features["horse_rolling_speed_z_mean_5"] = (
                rolling_speed_z.reindex(df.index).fillna(0.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features