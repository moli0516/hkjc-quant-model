# features/utils/smoother.py (完整防 Data Leakage 時間加固修正版)

from typing import List, Tuple, Union
import numpy as np
import pandas as pd


class BayesianSmoother:
    """貝氏平滑器與滾動統計工具庫 (防 Data Leakage、自動時間排序、安全向量化與高相容性版)."""

    @staticmethod
    def _prepare_time_sorted_df(
        df: pd.DataFrame, group_cols: list[str]
    ) -> pd.DataFrame:
        """內部私有工具：識別日期欄位並強制按時間排序，確保 shift(1) 隔離當場資料嚴格有效。"""
        work_df = df.copy()

        # 自動識別日期欄位
        date_col = None
        for col in ["date", "race_date", "datetime"]:
            if col in work_df.columns:
                date_col = col
                break

        # 若找到日期欄位，強制轉換為 datetime 並依照 [group_cols + date_col] 排序
        if date_col is not None:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
            work_df = work_df.sort_values(by=group_cols + [date_col])

        return work_df

    @staticmethod
    def calc_global_smooth_rate(
        df: pd.DataFrame,
        group_cols: str | list[str],
        target_col: str,
        prior_alpha: float = 10.0,
        baseline_rate: float = 0.08,
    ) -> pd.Series:
        """計算歷史擴展窗口 (Expanding) 貝氏平滑率 (嚴格按時間排序，排除當場賽事)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 1. 內部強制時間排序，防止外部未排序導致 shift(1) 洩漏
        work_df = BayesianSmoother._prepare_time_sorted_df(df, group_cols)

        # 🔒 2. 在時間排序後的資料上執行 groupby.shift(1) 排除當場數據
        shifted_target = work_df.groupby(group_cols)[target_col].shift(1)

        # 🔒 3. 進行累積擴展窗口計算
        grouped_shifted = shifted_target.groupby(
            [work_df[c] for c in group_cols]
            if len(group_cols) > 1
            else work_df[group_cols[0]]
        )

        cum_sum = grouped_shifted.cumsum().fillna(0.0)
        cum_count = grouped_shifted.cumcount().fillna(0.0)

        # 🔒 4. 計算貝氏平滑率
        smoothed_rate = (cum_sum + prior_alpha * baseline_rate) / (
            cum_count + prior_alpha
        )

        # 🔒 5. 精確對齊回原始 df.index
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
        """計算動態滾動統計量 (嚴格按時間排序，排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        # 若外部使用了 window 參數，自動覆蓋 window_size
        effective_window = window if window is not None else window_size

        if df.empty or value_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        # 🔒 1. 內部強制時間排序，徹底解決傳入未排序 df 的問題
        work_df = BayesianSmoother._prepare_time_sorted_df(df, group_cols)

        # 🔒 2. 在排序後的資料上使用向量化的 groupby.shift(1) 排除當場數據
        shifted_series = work_df.groupby(group_cols)[value_col].shift(1)

        # 🔒 3. 對 shift 後的 Series 進行分組滾動計算
        grouped_shifted = shifted_series.groupby(
            [work_df[c] for c in group_cols]
            if len(group_cols) > 1
            else work_df[group_cols[0]]
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

        # 🔒 4. 重置 MultiIndex 並精確對齊原始 df.index
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
        """計算滾動貝氏平滑勝率 (嚴格按時間排序，排除當場比賽數據，相容 window / window_size 傳參)."""
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        effective_window = window if window is not None else window_size

        if df.empty or target_col not in df.columns:
            return pd.Series(baseline_rate, index=df.index)

        # 🔒 1. 內部強制時間排序，確保 Shift(1) 排除時間上的當場比賽
        work_df = BayesianSmoother._prepare_time_sorted_df(df, group_cols)

        # 🔒 2. 向量化 Shift(1)
        shifted_target = work_df.groupby(group_cols)[target_col].shift(1)
        grouped_shifted = shifted_target.groupby(
            [work_df[c] for c in group_cols]
            if len(group_cols) > 1
            else work_df[group_cols[0]]
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

        # 🔒 3. 精確對齊原始 df.index
        roll_sum = roll_sum.reindex(df.index).fillna(0.0)
        roll_count = roll_count.reindex(df.index).fillna(0.0)

        smoothed_rate = (roll_sum + prior_alpha * baseline_rate) / (
            roll_count + prior_alpha
        )
        return smoothed_rate.fillna(baseline_rate)