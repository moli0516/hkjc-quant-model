import pandas as pd
from features.utils import LeakageGuard


class InjuryRestGenerator:

    """生成參賽節奏、久休復出與抵港參賽間隔特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "date" in work_df.columns:
            work_df["race_dt"] = pd.to_datetime(work_df["date"])
            prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)

            days_diff = (work_df["race_dt"] - prev_dt).dt.days
            features["days_since_last_race"] = (
                days_diff.reindex(df.index).fillna(999.0).astype("float32")
            )

            features["is_layoff_60d"] = (
                features["days_since_last_race"] >= 60.0
            ).astype("float32")
            features["is_layoff_90d"] = (
                features["days_since_last_race"] >= 90.0
            ).astype("float32")

        if "import_date" in df.columns and "date" in df.columns:
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            race_dt = pd.to_datetime(df["date"])
            features["days_since_import"] = (
                (race_dt - import_dt).dt.days.fillna(999.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features