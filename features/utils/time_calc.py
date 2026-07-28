from typing import Optional
import pandas as pd


class SpeedTimeCalculator:
    """專門處理賽事時間、段速與標準化速度計算。"""

    @staticmethod
    def calc_speed_mps(
        distance_col: pd.Series,
        time_sec_col: pd.Series,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """計算平均每秒跑多少米 (Meters Per Second, m/s)。"""
        valid_time = time_sec_col.replace(0, pd.NA)
        speed = distance_col / valid_time
        return speed.rename(
            feature_name if feature_name else "speed_meters_per_sec"
        )

    @staticmethod
    def normalize_time_by_distance(
        time_sec_col: pd.Series,
        distance_col: pd.Series,
        target_dist: float = 1200.0,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """將不同路程的時間按比例折算至標準路程秒數（預設折算為 1200 米）。"""
        valid_dist = distance_col.replace(0, pd.NA)
        normalized_time = (time_sec_col / valid_dist) * target_dist
        return normalized_time.rename(
            feature_name
            if feature_name
            else f"norm_time_{int(target_dist)}m"
        )