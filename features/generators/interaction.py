import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class InteractionGenerator:

    """生成交叉權重與高級交互特徵。"""

    EXECUTION_ORDER = 999  # 交叉特徵 Generator 必須排在最後面！

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 負重比率
        hw_col = next(
            (
                c
                for c in ["horse_weight", "declared_weight", "rank_weight"]
                if c in df.columns
            ),
            None,
        )
        if "actual_weight" in df.columns and hw_col:
            valid_hw = df[hw_col].replace(0, np.nan)
            features["weight_to_horse_body_ratio"] = (
                df["actual_weight"] / valid_hw
            ).astype("float32")

        # 2. 人馬勝率乘積
        h_win_col = next(
            (
                c
                for c in [
                    "horse_rolling_win_rate_5",
                    "h_smoothed_rolling_5_win_rate",
                ]
                if c in df.columns
            ),
            None,
        )
        j_win_col = next(
            (
                c
                for c in [
                    "jockey_rolling_win_rate_50",
                    "j_smoothed_rolling_30_win_rate",
                ]
                if c in df.columns
            ),
            None,
        )

        if h_win_col and j_win_col:
            features["horse_jockey_win_rate_interaction"] = (
                df[h_win_col] * df[j_win_col]
            ).astype("float32")

        # 3. 賠率落差與隱含勝率
        if "win_odds" in df.columns:
            implied_prob = 1.0 / df["win_odds"].replace(0, np.nan)
            features["win_odds_inv"] = implied_prob.astype("float32")

            if h_win_col:
                features["odds_vs_history_win_rate_gap"] = (
                    implied_prob - df[h_win_col]
                ).astype("float32")

        # 4. 檔位與速度 Z-Score 交互
        if "draw" in df.columns and "h_mean_speed_z_15" in df.columns:
            features["draw_speed_interaction"] = (
                df["draw"] * df["h_mean_speed_z_15"]
            ).astype("float32")

        # 5. 評分優勢 / 負重變化與體重交互
        rating_col = (
            "rating_vs_race_avg"
            if "rating_vs_race_avg" in df.columns
            else "rating"
        )
        if rating_col in df.columns and hw_col:
            features["rating_x_rank_weight"] = (
                df[rating_col] * df[hw_col]
            ).astype("float32")

        if "weight_delta" in df.columns and hw_col:
            features["delta_x_rank"] = (df["weight_delta"] * df[hw_col]).astype(
                "float32"
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features