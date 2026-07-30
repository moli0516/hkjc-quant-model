import pandas as pd
import numpy as np
from features.utils import LeakageGuard


class RatingClassGenerator:
    """生成馬匹班次升降 (Class Change) 特徵。"""

    EXECUTION_ORDER = 10

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        
        if "date" not in df.columns or "horse_id" not in df.columns:
            features["class_change"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()

        # 班次變動 (Class Change)
        if "race_class" in work_df.columns:
            class_num = (
                work_df["race_class"]
                .astype(str)
                .str.extract(r"(\d+)", expand=False)
                .astype(float)
            )

            work_df["_class_num"] = class_num
            prev_class = work_df.groupby("horse_id")["_class_num"].shift(1)

            class_change = work_df["_class_num"] - prev_class
            features["class_change"] = class_change.reindex(df.index).fillna(0.0).astype("float32")
        else:
            features["class_change"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features