"""市場基準：以獨贏賠率產生同場排名。"""

from __future__ import annotations

import pandas as pd


class MarketBaseline:
    """用 win_odds 產生 market_rank（1 = 最熱門）。"""

    def __init__(
        self,
        race_col: str = "race_id",
        odds_col: str = "win_odds",
        rank_col: str = "market_rank",
    ):
        self.race_col = race_col
        self.odds_col = odds_col
        self.rank_col = rank_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.odds_col not in df.columns:
            raise KeyError(f"缺少欄位: {self.odds_col}")
        if self.race_col not in df.columns:
            raise KeyError(f"缺少欄位: {self.race_col}")

        out = df.copy()
        valid = out[self.odds_col].notna() & (out[self.odds_col] > 0)

        out[self.rank_col] = pd.NA
        out.loc[valid, self.rank_col] = (
            out.loc[valid]
            .groupby(self.race_col)[self.odds_col]
            .rank(ascending=True, method="first")
        )
        out[self.rank_col] = out[self.rank_col].astype("Float64")
        return out