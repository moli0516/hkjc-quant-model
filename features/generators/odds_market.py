import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class OddsMarketGenerator:

    """生成獨贏賠率、市場隱含勝率與熱門指標特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "win_odds" in df.columns:
            valid_odds = df["win_odds"].replace(0, np.nan)

            features["odds_implied_prob"] = (1.0 / valid_odds).astype(
                "float32"
            )
            features["is_market_favorite"] = (df["win_odds"] <= 3.0).astype(
                "float32"
            )

            odds_mean = df.groupby("race_id")["win_odds"].transform("mean")
            odds_std = df.groupby("race_id")["win_odds"].transform("std")
            features["odds_race_zscore"] = (
                (df["win_odds"] - odds_mean) / (odds_std + 1e-6)
            ).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features