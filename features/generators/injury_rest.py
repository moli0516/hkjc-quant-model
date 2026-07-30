import pandas as pd
from features.utils import LeakageGuard


class InjuryRestGenerator:
    """生成參賽節奏、久休復出與抵港參賽間隔特徵。"""

    EXECUTION_ORDER = 110

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()
        
        date_col = "race_date" if "race_date" in df.columns else ("date" if "date" in df.columns else None)
        
        if date_col is None:
            features["days_since_last_race"] = 999.0
            features["is_layoff_60d"] = 0.0
            features["is_layoff_90d"] = 0.0
            features["days_since_import"] = 999.0
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", date_col]).copy()

        # 1. 計算距離上場賽事天數 (嚴格按時間)
        work_df["race_dt"] = pd.to_datetime(work_df[date_col])
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

        # 2. 🔒 安全計算抵港天數（嚴格濾除未來時間泄露）
        if "import_date" in df.columns:
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            race_dt = pd.to_datetime(df[date_col], errors="coerce")
            
            diff_days = (race_dt - import_dt).dt.days
            # 若抵港日晚於比賽日 (diff_days < 0)，視為未到港數據，填為 999.0
            valid_diff = diff_days.where(diff_days >= 0, 999.0)
            
            features["days_since_import"] = (
                valid_diff.reindex(df.index).fillna(999.0).astype("float32")
            )
        else:
            features["days_since_import"] = 999.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features