"""排序評估指標：model vs market。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RankingMetrics:
    def __init__(
        self,
        race_col: str = "race_id",
        label_col: str = "placing",
    ):
        self.race_col = race_col
        self.label_col = label_col

    def _race_level_hit(
        self,
        df: pd.DataFrame,
        rank_col: str,
        max_rank: int,
        max_label: int,
    ) -> float:
        """按場計算命中率，再對場次取平均。"""
        if df.empty:
            return float("nan")

        work = df.dropna(subset=[rank_col, self.label_col]).copy()
        if work.empty:
            return float("nan")

        work["_hit"] = (
            (work[rank_col] <= max_rank) & (work[self.label_col] <= max_label)
        ).astype(float)

        # 一場內只要有符合條件的馬即算該場命中（top1: rank==1 & placing==1）
        if max_rank == 1 and max_label == 1:
            per_race = work.groupby(self.race_col)["_hit"].max()
        else:
            # top3_rate: 模型前3名中有多少比例實際入三甲（按馬平均較直觀時可改）
            # 這裡採「場級」：該場 model_rank<=3 的馬中，placing<=3 的比例，再對場平均
            sub = work[work[rank_col] <= max_rank]
            if sub.empty:
                return float("nan")
            per_race = sub.groupby(self.race_col)["_hit"].mean()

        return float(per_race.mean())

    def top1_win_rate(
        self,
        df: pd.DataFrame,
        rank_col: str = "model_rank",
    ) -> float:
        work = df.dropna(subset=[rank_col, self.label_col]).copy()
        if work.empty:
            return float("nan")

        # 每場取 rank==1 的那匹，看是否 placing==1
        top1 = work[work[rank_col] == 1]
        if top1.empty:
            return float("nan")

        # 同一場若有多筆 rank==1（不應發生），取 first
        top1 = top1.groupby(self.race_col, as_index=False).first()
        return float((top1[self.label_col] == 1).mean())

    def top3_rate(
        self,
        df: pd.DataFrame,
        rank_col: str = "model_rank",
    ) -> float:
        """模型 rank<=3 的馬，實際 placing<=3 的比例（馬級平均）。"""
        work = df.dropna(subset=[rank_col, self.label_col]).copy()
        if work.empty:
            return float("nan")

        sub = work[work[rank_col] <= 3]
        if sub.empty:
            return float("nan")
        return float((sub[self.label_col] <= 3).mean())

    def compare(self, df: pd.DataFrame) -> dict:
        return {
            "model_top1": self.top1_win_rate(df, "model_rank"),
            "market_top1": self.top1_win_rate(df, "market_rank"),
            "model_top3": self.top3_rate(df, "model_rank"),
            "market_top3": self.top3_rate(df, "market_rank"),
            "n_races": int(df[self.race_col].nunique()) if not df.empty else 0,
            "n_runners": int(len(df)),
        }