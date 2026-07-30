import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard


class HorseRollingGenerator:

    """計算馬匹歷史賽事紀錄之滾動 (Rolling) 統計特徵（已嚴格防堵 Data Leakage）。"""

    EXECUTION_ORDER = 50

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 確保嚴格按馬匹與時間排序
        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        # 2. 關鍵防洩漏步驟：先對基礎欄位做 .shift(1)，將當場比賽結果隔離
        work_df["shifted_placing"] = work_df.groupby("horse_id")["placing"].shift(1)
        work_df["shifted_is_win"] = work_df.groupby("horse_id")["is_win"].shift(1)
        work_df["shifted_is_top3"] = work_df.groupby("horse_id")["is_top3"].shift(1)
        
        if "actual_weight" in work_df.columns:
            work_df["shifted_actual_weight"] = work_df.groupby("horse_id")["actual_weight"].shift(1)

        windows = [3, 5, 10]
        for w in windows:
            # 使用 shift 後的欄位計算 Rolling，徹底阻斷數據洩漏
            features[f"horse_rolling_pos_mean_{w}"] = (
                BayesianSmoother.calc_rolling_stat(
                    work_df,
                    group_cols="horse_id",
                    value_col="shifted_placing",
                    window_size=w,
                    stat_type="mean",
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_pos_std_{w}"] = (
                BayesianSmoother.calc_rolling_stat(
                    work_df,
                    group_cols="horse_id",
                    value_col="shifted_placing",
                    window_size=w,
                    stat_type="std",
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_win_rate_{w}"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="horse_id",
                    target_col="shifted_is_win",
                    window_size=w,
                    prior_alpha=3.0,
                    baseline_rate=0.08,
                )
                .reindex(df.index)
                .astype("float32")
            )

            features[f"horse_rolling_top3_rate_{w}"] = (
                BayesianSmoother.calc_rolling_smooth_rate(
                    work_df,
                    group_cols="horse_id",
                    target_col="shifted_is_top3",
                    window_size=w,
                    prior_alpha=3.0,
                    baseline_rate=0.24,
                )
                .reindex(df.index)
                .astype("float32")
            )

            if "actual_weight" in work_df.columns:
                features[f"horse_rolling_weight_mean_{w}"] = (
                    BayesianSmoother.calc_rolling_stat(
                        work_df,
                        group_cols="horse_id",
                        value_col="shifted_actual_weight",
                        window_size=w,
                        stat_type="mean",
                    )
                    .reindex(df.index)
                    .astype("float32")
                )

        if "date" in work_df.columns:
            work_df["race_dt"] = pd.to_datetime(work_df["date"])
            prev_dt = work_df.groupby("horse_id")["race_dt"].shift(1)
            days = (work_df["race_dt"] - prev_dt).dt.days.fillna(999)
            features["days_since_last_race"] = (
                days.reindex(df.index).astype("float32")
            )

        if "horse_weight" in work_df.columns:
            prev_horse_w = work_df.groupby("horse_id")["horse_weight"].shift(1)
            w_change = work_df["horse_weight"] - prev_horse_w
            features["horse_weight_change"] = (
                w_change.reindex(df.index).fillna(0.0).astype("float32")
            )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features