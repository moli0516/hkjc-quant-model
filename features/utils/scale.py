import numpy as np
import pandas as pd


class RaceScaler:
    """專門處理同場賽事（Race-Level）特徵相對化與標準化."""

    @staticmethod
    def race_z_score(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        feature_name: str = None,
    ) -> pd.Series:
        """計算欄位在同場賽事中的 Z-Score (X - Mean) / Std."""
        grp = df.groupby(race_col)[value_col]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, 1e-6).fillna(1e-6)

        z_score = (df[value_col] - mean) / std
        return z_score.rename(
            feature_name if feature_name else f"{value_col}_race_z"
        )

    @staticmethod
    def race_diff_from_mean(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        feature_name: str = None,
    ) -> pd.Series:
        """計算欄位與同場平均值的差額 (X - Mean)."""
        mean = df.groupby(race_col)[value_col].transform("mean")
        diff = df[value_col] - mean
        return diff.rename(
            feature_name if feature_name else f"{value_col}_diff_mean"
        )

    @staticmethod
    def race_rank(
        df: pd.DataFrame,
        race_col: str,
        value_col: str,
        ascending: bool = False,
        feature_name: str = None,
    ) -> pd.Series:
        """計算數值在同場賽事中的相對排名 (例如：賠率第幾高、負重第幾重)."""
        rank = df.groupby(race_col)[value_col].rank(
            ascending=ascending, method="min"
        )
        return rank.rename(
            feature_name if feature_name else f"{value_col}_race_rank"
        )