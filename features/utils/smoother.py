# features/utils/smoother.py (加強參數相容與安全防護修正版)

import numpy as np
import pandas as pd


class BayesianSmoother:
    """貝氏平滑器與滾動統計工具庫 (防 Data Leakage、安全向量化與高相容性版)."""

    @staticmethod
    def calc_global_smooth_rate(
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        prior_alpha: float = 10.0,
        baseline_rate: float = 0.08,
    ) -> pd.Series:
        """計算歷史擴展窗口 (Expanding) 貝氏平滑率 (嚴格排除當場賽事)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 防洩漏：先使用原生 groupby.shift(1) 排除當場數據，避免使用 apply 導致 empty concat 崩潰
        shifted_target = df.groupby(group_cols)[target_col].shift(1)

        # 進行累積擴展窗口計算
        cum_sum = shifted_target.groupby([df[c] for c in group_cols]).cumsum()
        cum_count = shifted_target.groupby([df[c] for c in group_cols]).cumcount()

        smoothed_rate = (cum_sum.fillna(0.0) + prior_alpha * baseline_rate) / (
            cum_count.fillna(0.0) + prior_alpha
        )
        return smoothed_rate.fillna(baseline_rate).reindex(df.index)

    @staticmethod
    def calc_rolling_stat(
        df: pd.DataFrame,
        group_cols: str | list[str],
        value_col: str,
        window_size: int = 5,
        stat_type: str = "mean",
        min_periods: int = 1,
        window: int = None,  # 🔒 相容性修復：允許外部傳入 window 參數
        **kwargs,
    ) -> pd.Series:
        """計算動態滾動統計量 (嚴格排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # 若外部使用了 window 參數，自動覆蓋 window_size
        effective_window = window if window is not None else window_size

        if df.empty or value_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        # 🔒 1. 使用向量化的 groupby.shift(1) 排除當場數據
        shifted_series = df.groupby(group_cols)[value_col].shift(1)

        # 🔒 2. 對 shift 後的 Series 進行分組滾動計算
        grouped_shifted = shifted_series.groupby(
            [df[c] for c in group_cols]
            if len(group_cols) > 1
            else df[group_cols[0]]
        )

        if stat_type == "mean":
            res = grouped_shifted.rolling(
                window=effective_window, min_periods=min_periods
            ).mean()
        elif stat_type == "std":
            res = (
                grouped_shifted.rolling(
                    window=effective_window, min_periods=min_periods
                )
                .std()
                .fillna(0.0)
            )
        else:
            raise ValueError(f"不支援的 stat_type: {stat_type}")

        # 🔒 3. 重置 MultiIndex 並精確對齊原始 df.index
        if isinstance(res.index, pd.MultiIndex):
            res = res.reset_index(level=list(range(len(group_cols))), drop=True)

        return res.reindex(df.index)

    @staticmethod
    def calc_rolling_smooth_rate(
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        window_size: int = 5,
        prior_alpha: float = 5.0,
        baseline_rate: float = 0.08,
        window: int = None,  # 🔒 相容性修復：允許外部傳入 window 參數
        **kwargs,
    ) -> pd.Series:
        """計算滾動貝氏平滑勝率 (嚴格排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        effective_window = window if window is not None else window_size

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 向量化 Shift(1)
        shifted_target = df.groupby(group_cols)[target_col].shift(1)
        grouped_shifted = shifted_target.groupby(
            [df[c] for c in group_cols]
            if len(group_cols) > 1
            else df[group_cols[0]]
        )

        roll_sum = grouped_shifted.rolling(
            window=effective_window, min_periods=1
        ).sum()
        roll_count = grouped_shifted.rolling(
            window=effective_window, min_periods=1
        ).count()

        if isinstance(roll_sum.index, pd.MultiIndex):
            roll_sum = roll_sum.reset_index(
                level=list(range(len(group_cols))), drop=True
            )
            roll_count = roll_count.reset_index(
                level=list(range(len(group_cols))), drop=True
            )

        roll_sum = roll_sum.reindex(df.index).fillna(0.0)
        roll_count = roll_count.reindex(df.index).fillna(0.0)

        smoothed_rate = (roll_sum + prior_alpha * baseline_rate) / (
            roll_count + prior_alpha
        )
        return smoothed_rate.fillna(baseline_rate)