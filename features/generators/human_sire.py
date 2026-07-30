import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class HumanSireGenerator:
    """騎師/練馬師/種馬 (Sire) 歷史績效特徵生成器（已嚴格實施 Shift 隔離防範 Leakage）"""

    EXECUTION_ORDER = 60

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "date" if "date" in df.columns else ("race_date" if "race_date" in df.columns else None)
        if date_col:
            work_df = df.sort_values([date_col, "race_id", "horse_id"]).copy()
        else:
            work_df = df.copy()

        # 1. 建立當場 is_win 標籤
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32") if "placing" in work_df.columns else 0.0

        # 2. 🔒 關鍵防洩漏：必須對各群組進行 shift(1)，確保不包含當場比賽結果
        if "jockey" in work_df.columns:
            work_df["shifted_jockey_is_win"] = work_df.groupby("jockey")["is_win"].shift(1)
            jockey_rolling_win = BayesianSmoother.calc_rolling_smooth_rate(
                work_df, group_cols="jockey", target_col="shifted_jockey_is_win", window=10, prior_alpha=5.0, baseline_rate=0.08
            )
            features["jockey_rolling_win_rate_10"] = jockey_rolling_win.reindex(df.index).astype("float32")

        if "trainer" in work_df.columns:
            work_df["shifted_trainer_is_win"] = work_df.groupby("trainer")["is_win"].shift(1)
            trainer_rolling_win = BayesianSmoother.calc_rolling_smooth_rate(
                work_df, group_cols="trainer", target_col="shifted_trainer_is_win", window=20, prior_alpha=10.0, baseline_rate=0.08
            )
            features["trainer_rolling_win_rate_20"] = trainer_rolling_win.reindex(df.index).astype("float32")

        if "sire" in work_df.columns:
            # 種馬通常為全域統計，但為防止洩漏當場，亦可依時間累積計算
            work_df["shifted_sire_is_win"] = work_df.groupby("sire")["is_win"].shift(1)
            sire_smooth_win = BayesianSmoother.calc_global_smooth_rate(
                work_df, group_cols="sire", target_col="shifted_sire_is_win", prior_alpha=15.0, baseline_rate=0.08
            )
            features["sire_global_win_rate"] = sire_smooth_win.reindex(df.index).astype("float32")

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features