import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class SynergyFitnessGenerator:

    """生成人馬默契與更換騎師特徵 (全面向量化優化)。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        work_df = df.sort_values(["horse_id", "date"]).copy()

        if "jockey" in work_df.columns:
            # 1. 是否更換騎師
            prev_jockey = work_df.groupby("horse_id")["jockey"].shift(1)
            is_changed = (work_df["jockey"] != prev_jockey).astype("float32")
            features["is_jockey_changed"] = (
                is_changed.reindex(df.index).fillna(0.0).astype("float32")
            )

            # 2. 人馬組合
            work_df["horse_jockey_pair"] = (
                work_df["horse_id"].astype(str)
                + "_"
                + work_df["jockey"].astype(str)
            )
            work_df["is_win"] = (work_df["placing"] == 1).astype("float32")

            # 🚀【核心效能優化】：用 cumcount() 替代 transform(lambda)
            # 歷史合作次數 (不包含當場，所以從 0 開始，剛好符合 shift 效果)
            pair_counts = work_df.groupby("horse_jockey_pair").cumcount()
            features["pair_ride_count"] = (
                pair_counts.reindex(df.index).astype("float32")
            )

            # 3. 貝氏平滑人馬勝率
            pair_win_rate = BayesianSmoother.calc_global_smooth_rate(
                work_df,
                group_cols="horse_jockey_pair",
                target_col="is_win",
                prior_alpha=2.0,
                baseline_rate=0.08,
            )
            features["pair_win_rate"] = (
                pair_win_rate.reindex(df.index).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features