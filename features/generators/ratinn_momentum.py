import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, RaceScaler


class RatingMomentumGenerator:
    """馬匹評分趨勢與同場評分優勢生成器 (純硬實力預測)"""

    EXECUTION_ORDER = 12

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "rating" not in df.columns:
            features["rating_diff_from_race_mean"] = 0.0
            features["rating_momentum_3"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        # 1. 同場賽事相對評分優勢 (與該場平均評分差額 & Z-Score)
        features["rating_diff_from_race_mean"] = RaceScaler.race_diff_from_mean(
            df, race_col="race_id", value_col="rating"
        ).astype("float32")

        features["rating_race_z"] = RaceScaler.race_z_score(
            df, race_col="race_id", value_col="rating"
        ).astype("float32")

        # 2. 評分動態上升/下降趨勢 (最近一次 rating vs 3場前 rating)
        if "date" in df.columns and "horse_id" in df.columns:
            work_df = df.sort_values(["horse_id", "date"]).copy()
            prev_rating_1 = work_df.groupby("horse_id")["rating"].shift(1)
            prev_rating_3 = work_df.groupby("horse_id")["rating"].shift(3)

            # 近 3 場評分變化量
            rating_momentum = prev_rating_1 - prev_rating_3
            features["rating_momentum_3"] = (
                rating_momentum.reindex(df.index).fillna(0.0).astype("float32")
            )
        else:
            features["rating_momentum_3"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features