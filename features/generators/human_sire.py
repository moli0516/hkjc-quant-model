import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class HumanSireGenerator:

    """生成騎師、練馬師、騎練組合與血統平滑特徵。"""

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        work_df = df.sort_values("date").copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        if "jockey" in work_df.columns:
            features["jockey_rolling_win_rate_50"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="jockey",
                    target_col="is_win",
                    window_size=50,
                    prior_alpha=10.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        if "trainer" in work_df.columns:
            features["trainer_rolling_win_rate_50"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="trainer",
                    target_col="is_win",
                    window_size=50,
                    prior_alpha=10.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        if "jockey" in work_df.columns and "trainer" in work_df.columns:
            work_df["jockey_trainer_combo"] = (
                work_df["jockey"].astype(str)
                + "_"
                + work_df["trainer"].astype(str)
            )
            features["jt_combo_win_rate_30"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="jockey_trainer_combo",
                    target_col="is_win",
                    window_size=30,
                    prior_alpha=5.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        if "sire" in work_df.columns:
            features["sire_global_win_rate"] = (
                BayesianSmoother.calc_global_smooth_rate(
                    work_df,
                    group_cols="sire",
                    target_col="is_win",
                    prior_alpha=15.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features