import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class HorseProfileGenerator:
    """生成馬匹服役資歷與烙印年資特徵 (Data Source: horses / race_results)"""

    EXECUTION_ORDER = 20

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        date_col = "race_date" if "race_date" in df.columns else ("date" if "date" in df.columns else None)

        # 🔒 安全計算：馬匹在「該場賽事當下」在香港服役的實際年數
        if date_col and "import_date" in df.columns:
            race_dt = pd.to_datetime(df[date_col], errors="coerce")
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            
            # 計算比賽當天距離抵港日期的天數（若抵港日在比賽日之後則會被遮蔽為 NaN）
            days_in_hk = (race_dt - import_dt).dt.days
            
            # 僅保留比賽當下已抵港的合法紀錄 (>=0 天)
            valid_days = days_in_hk.where(days_in_hk >= 0, np.nan)
            features["est_years_in_hk"] = (valid_days / 365.25).clip(lower=0.0, upper=10.0).fillna(0.0).astype("float32")
        else:
            features["est_years_in_hk"] = 0.0

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features