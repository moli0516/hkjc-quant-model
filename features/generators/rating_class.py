import pandas as pd
from features.utils import LeakageGuard


class RatingClassGenerator:

    """生成馬匹評分變動與班次升降 (Class Change) 特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "rating" in work_df.columns:
            prev_rating = work_df.groupby("horse_id")["rating"].shift(1)
            rating_change = work_df["rating"] - prev_rating
            features["rating_change"] = (
                rating_change.reindex(df.index).fillna(0.0).astype("float32")
            )

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
            features["class_change"] = (
                class_change.reindex(df.index).fillna(0.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features