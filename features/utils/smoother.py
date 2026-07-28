from typing import Optional, Union
import numpy as np
import pandas as pd


class BayesianSmoother:
    """通用貝氏平滑與滾動視窗特徵生成模板類別。

    包含強制的 `.shift(1)`，確保完全避免 Data Leakage。
    """

    @staticmethod
    def calc_global_smooth_rate(
        df: pd.DataFrame,
        group_cols: Union[str, list[str]],
        target_col: str,
        prior_alpha: float,
        baseline_rate: float,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """計算全域累積（Expanding Window）的貝氏平滑率。"""
        grp = df.groupby(group_cols)[target_col]

        attempts = (
            grp.transform(
                lambda x: x.shift(1).expanding(min_periods=1).count()
            )
            .fillna(0)
            .astype(float)
        )
        successes = (
            grp.transform(lambda x: x.shift(1).expanding(min_periods=1).sum())
            .fillna(0)
            .astype(float)
        )

        smooth_rate = (successes + prior_alpha * baseline_rate) / (
            attempts + prior_alpha
        )
        return smooth_rate.rename(
            feature_name
            if feature_name
            else f"{target_col}_global_smooth_a{prior_alpha}"
        )

    @staticmethod
    def calc_rolling_smooth_rate(
        df: pd.DataFrame,
        group_cols: Union[str, list[str]],
        target_col: str,
        window_size: int,
        prior_alpha: float,
        baseline_rate: float,
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """計算近 N 場（Rolling Window）的貝氏平滑率。"""
        grp = df.groupby(group_cols)[target_col]

        attempts = (
            grp.transform(
                lambda x: x.shift(1)
                .rolling(window=window_size, min_periods=1)
                .count()
            )
            .fillna(0)
            .astype(float)
        )
        successes = (
            grp.transform(
                lambda x: x.shift(1)
                .rolling(window=window_size, min_periods=1)
                .sum()
            )
            .fillna(0)
            .astype(float)
        )

        smooth_rate = (successes + prior_alpha * baseline_rate) / (
            attempts + prior_alpha
        )
        return smooth_rate.rename(
            feature_name
            if feature_name
            else f"{target_col}_rolling_{window_size}_smooth_a{prior_alpha}"
        )

    @staticmethod
    def calc_rolling_stat(
        df: pd.DataFrame,
        group_cols: Union[str, list[str]],
        value_col: str,
        window_size: int,
        stat_type: str = "mean",
        feature_name: Optional[str] = None,
    ) -> pd.Series:
        """通用數值型欄位（如速度、負重、名次、勝負距離）的滾動統計。"""
        grp = df.groupby(group_cols)[value_col]
        shifted_grp = grp.transform(lambda x: x.shift(1))

        if stat_type == "mean":
            res = shifted_grp.transform(
                lambda x: x.rolling(window=window_size, min_periods=1).mean()
            )
        elif stat_type == "std":
            res = (
                shifted_grp.transform(
                    lambda x: x.rolling(
                        window=window_size, min_periods=2
                    ).std()
                )
                .fillna(0)
                .astype(float)
            )
        elif stat_type == "max":
            res = shifted_grp.transform(
                lambda x: x.rolling(window=window_size, min_periods=1).max()
            )
        elif stat_type == "min":
            res = shifted_grp.transform(
                lambda x: x.rolling(window=window_size, min_periods=1).min()
            )
        else:
            raise ValueError(f"❌ 不支援的統計類型: {stat_type}")

        return res.rename(
            feature_name
            if feature_name
            else f"{value_col}_rolling_{window_size}_{stat_type}"
        )