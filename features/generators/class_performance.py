import numpy as np
import pandas as pd
import re
from features.utils import BayesianSmoother, LeakageGuard


class ClassPerformanceGenerator:
    """馬匹班次表現與升降班適應力特徵生成器 (僅限 Class 1-5)"""

    EXECUTION_ORDER = 15

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    @staticmethod
    def _parse_class_num(race_class_series: pd.Series) -> pd.Series:
        """只解析 Class 1 到 5，其他班次 (Group/Griffin/特殊賽事) 一律回傳 np.nan"""

        def parse_val(val):
            if pd.isna(val):
                return np.nan
            s = str(val).upper().strip()

            # 抓取班次數字 (例如: "CLASS 3" -> 3)
            match = re.search(r"(\d+)", s)
            if match:
                class_num = float(match.group(1))
                # 嚴格限制在 1 至 5 班之間
                if 1.0 <= class_num <= 5.0:
                    return class_num

            # 非 1-5 班賽事 (如 Group 1, Griffin 等) 返回 NaN
            return np.nan

        return race_class_series.apply(parse_val)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        if "race_class" not in df.columns:
            LeakageGuard.validate_feature_dataframe(features, self.key_cols)
            return features

        work_df = df.sort_values(["horse_id", "date"]).copy()
        work_df["class_num"] = self._parse_class_num(work_df["race_class"])
        work_df["is_win"] = (work_df["placing"] == 1).astype("float32")
        work_df["is_top3"] = (work_df["placing"] <= 3).astype("float32")

        # 計算同班歷史勝率與上名率 (非 1-5 班的紀錄會自動忽略)
        class_win_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["horse_id", "class_num"],
            target_col="is_win",
            prior_alpha=2.0,
            baseline_rate=0.08,
        )
        features["horse_class_win_rate"] = class_win_rate.reindex(
            df.index
        ).astype("float32")

        class_top3_rate = BayesianSmoother.calc_global_smooth_rate(
            work_df,
            group_cols=["horse_id", "class_num"],
            target_col="is_top3",
            prior_alpha=2.0,
            baseline_rate=0.24,
        )
        features["horse_class_top3_rate"] = class_top3_rate.reindex(
            df.index
        ).astype("float32")

        # 升降班動態標籤計算
        prev_class = work_df.groupby("horse_id")["class_num"].shift(1)
        work_df["class_diff"] = work_df["class_num"] - prev_class

        # 只有兩場賽事都在 1-5 班內時，才會計算升降班
        work_df["is_class_up"] = (work_df["class_diff"] < 0).astype("float32")
        work_df["is_class_down"] = (work_df["class_diff"] > 0).astype("float32")

        features["is_class_up"] = (
            work_df["is_class_up"].reindex(df.index).fillna(0.0).astype("float32")
        )
        features["is_class_down"] = (
            work_df["is_class_down"]
            .reindex(df.index)
            .fillna(0.0)
            .astype("float32")
        )

        # 歷史升降班次數統計
        shifted_up = work_df.groupby("horse_id")["is_class_up"].shift(1)
        shifted_down = work_df.groupby("horse_id")["is_class_down"].shift(1)

        cum_class_up_count = (
            shifted_up.groupby(work_df["horse_id"]).cumsum().fillna(0.0)
        )
        cum_class_down_count = (
            shifted_down.groupby(work_df["horse_id"]).cumsum().fillna(0.0)
        )

        features["horse_hist_class_up_count"] = (
            cum_class_up_count.reindex(df.index).astype("float32")
        )
        features["horse_hist_class_down_count"] = (
            cum_class_down_count.reindex(df.index).astype("float32")
        )

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features