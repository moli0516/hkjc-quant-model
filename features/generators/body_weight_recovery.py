import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class BodyWeightRecoveryGenerator:
    """馬匹體重變動與體能恢復特徵生成器 (完全不含賠率)"""

    EXECUTION_ORDER = 52

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 支援多種常見的馬匹體重欄位命名
        weight_col = next(
            (c for c in ["declared_weight", "horse_weight"] if c in df.columns),
            None,
        )

        if weight_col is None or "date" not in df.columns:
            features["horse_weight_vs_hist_mean"] = 0.0
            features["horse_weight_abs_change"] = 0.0
            features["is_heavy_workload_14d"] = 0.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["race_dt"] = pd.to_datetime(work_df["date"])

        # 1. 馬匹過去 3 場的平均體重 (與當前體重比較)
        hist_weight_mean = BayesianSmoother.calc_rolling_stat(
            work_df,
            group_cols="horse_id",
            value_col=weight_col,
            window=3,
            stat="mean",
        )
        prev_weight = work_df.groupby("horse_id")[weight_col].shift(1)

        features["horse_weight_vs_hist_mean"] = (
            (df[weight_col] - hist_weight_mean)
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )
        
        # 2. 上場與本場體重的絕對變化量 (過胖或減過頭皆影響勝率)
        features["horse_weight_abs_change"] = (
            (df[weight_col] - prev_weight)
            .abs()
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )

        # 3. 14 天內連續出賽的高強度密集賽程標記
        prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)
        days_rest = (work_df["race_dt"] - prev_dt).dt.days
        features["is_heavy_workload_14d"] = (
            (days_rest <= 14).astype("float32").reindex(df.index).fillna(0.0)
        )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features