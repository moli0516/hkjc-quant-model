import pandas as pd
from features.utils import RaceScaler, LeakageGuard


class ContextRelativeGenerator:

    """生成同場賽事內部相對優勢特徵 (Race-Level Relative Features)。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "win_odds" in df.columns:
            features["win_odds_race_z"] = RaceScaler.race_z_score(
                df, race_col="race_id", value_col="win_odds"
            ).astype("float32")
            features["win_odds_race_rank"] = RaceScaler.race_rank(
                df, race_col="race_id", value_col="win_odds", ascending=True
            ).astype("float32")

        if "actual_weight" in df.columns:
            features["weight_diff_from_race_mean"] = (
                RaceScaler.race_diff_from_mean(
                    df, race_col="race_id", value_col="actual_weight"
                ).astype("float32")
            )
            features["weight_race_z"] = RaceScaler.race_z_score(
                df, race_col="race_id", value_col="actual_weight"
            ).astype("float32")

        if "horse_rolling_win_rate_5" in df.columns:
            features["horse_win_rate_race_z"] = RaceScaler.race_z_score(
                df, race_col="race_id", value_col="horse_rolling_win_rate_5"
            ).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features