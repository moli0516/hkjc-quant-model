import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class HorseProfileGenerator:

    """生成馬匹靜態與背景屬性特徵 (Data Source: horses)。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "date" in df.columns and "birth_date" in df.columns:
            race_dt = pd.to_datetime(df["date"])
            birth_dt = pd.to_datetime(df["birth_date"])
            features["horse_age"] = (
                (race_dt - birth_dt).dt.days / 365.25
            ).astype("float32")
        elif "age" in df.columns:
            features["horse_age"] = df["age"].astype("float32")
        else:
            features["horse_age"] = np.nan

        if "brand_no" in df.columns:
            features["brand_prefix"] = (
                df["brand_no"].astype(str).str[0].str.upper()
            )
        else:
            features["brand_prefix"] = "UNKNOWN"

        categorical_cols = ["origin", "import_type", "sex", "color"]
        for col in categorical_cols:
            if col in df.columns:
                features[f"horse_{col}"] = df[col].fillna("UNKNOWN").astype(str)

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features