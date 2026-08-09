"""簡單下注規則與資金曲線統計。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BetEvaluator:
    def __init__(
        self,
        overlay_threshold: float = 1.15,
        stake: float = 1.0,
        race_col: str = "race_id",
        odds_col: str = "win_odds",
        label_col: str = "placing",
        score_col: str = "model_score",
        rank_col: str = "model_rank",
    ):
        self.overlay_threshold = overlay_threshold
        self.stake = stake
        self.race_col = race_col
        self.odds_col = odds_col
        self.label_col = label_col
        self.score_col = score_col
        self.rank_col = rank_col

    def _softmax_by_race(self, df: pd.DataFrame) -> pd.Series:
        """同場對 model_score 做 softmax，得到 model_prob。"""
        def _sm(s: pd.Series) -> pd.Series:
            x = s.astype(float).values
            x = x - np.nanmax(x)
            ex = np.exp(x)
            ex = np.where(np.isfinite(ex), ex, 0.0)
            denom = ex.sum()
            if denom <= 0:
                return pd.Series(np.ones(len(s)) / max(len(s), 1), index=s.index)
            return pd.Series(ex / denom, index=s.index)

        return df.groupby(self.race_col)[self.score_col].transform(_sm)

    def rule_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """每場固定下 model_rank == 1。"""
        work = df.dropna(subset=[self.rank_col, self.odds_col, self.label_col]).copy()
        work = work[work[self.odds_col] > 0]
        bets = work[work[self.rank_col] == 1].copy()
        bets["stake"] = self.stake
        # HKJC 獨贏：賠率為「贏時每注收回的倍數（含本金）」常見定義；
        # 若你的 win_odds 是淨賠率，請改為 profit = stake * odds
        bets["profit"] = np.where(
            bets[self.label_col] == 1,
            bets["stake"] * bets[self.odds_col] - bets["stake"],
            -bets["stake"],
        )
        bets["rule"] = "A_top1"
        return bets

    def rule_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Overlay 價值下注：
        model_prob > market_prob * threshold 且 model_rank <= 3。
        """
        work = df.dropna(
            subset=[self.rank_col, self.odds_col, self.label_col, self.score_col]
        ).copy()
        work = work[work[self.odds_col] > 0]

        work["model_prob"] = self._softmax_by_race(work)
        # 市場隱含機率（同場正規化）
        work["raw_market_prob"] = 1.0 / work[self.odds_col]
        work["market_prob"] = work.groupby(self.race_col)["raw_market_prob"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else s
        )

        mask = (
            (work[self.rank_col] <= 3)
            & (work["model_prob"] > work["market_prob"] * self.overlay_threshold)
        )
        bets = work.loc[mask].copy()
        bets["stake"] = self.stake
        bets["profit"] = np.where(
            bets[self.label_col] == 1,
            bets["stake"] * bets[self.odds_col] - bets["stake"],
            -bets["stake"],
        )
        bets["rule"] = "B_overlay"
        return bets
    
    def rule_c_same_pick(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        規則 C：僅當模型第一 == 市場大熱（同馬）+ 強試閘時才下。
        條件：model_rank==1 且 market_rank==1
        """
        work = df.dropna(
            subset=[self.rank_col, "market_rank", "is_strong_trial", self.odds_col, self.label_col]
        ).copy()
        work = work[work[self.odds_col] > 0]

        bets = work[
            (work[self.rank_col] == 1) & (work["market_rank"] == 1) & (work["is_strong_trial"] == 1)
        ].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()

        bets["stake"] = self.stake
        bets["profit"] = np.where(
            bets[self.label_col] == 1,
            bets["stake"] * bets[self.odds_col] - bets["stake"],
            -bets["stake"],
        )
        bets["rule"] = "C_same_pick"
        return bets
    
    def rule_c_compare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        規則 C-對照：僅當模型第一 == 市場大熱（同馬）+ 弱試閘時才下。
        條件：model_rank==1 且 market_rank==1
        """
        work = df.dropna(
            subset=[self.rank_col, "market_rank", "is_strong_trial", self.odds_col, self.label_col]
        ).copy()
        work = work[work[self.odds_col] > 0]

        bets = work[
            (work[self.rank_col] == 1) & (work["market_rank"] == 1) & (work["is_strong_trial"] == 0)
        ].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()

        bets["stake"] = self.stake
        bets["profit"] = np.where(
            bets[self.label_col] == 1,
            bets["stake"] * bets[self.odds_col] - bets["stake"],
            -bets["stake"],
        )
        bets["rule"] = "C_same_pick"
        return bets


    def rule_d_model1_market_top2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        規則 D：模型第一，且該馬市場排名 <= 2 才下。
        條件：model_rank==1 且 market_rank<=2
        """
        work = df.dropna(
            subset=[self.rank_col, "market_rank", self.odds_col, self.label_col]
        ).copy()
        work = work[work[self.odds_col] > 0]

        bets = work[
            (work[self.rank_col] == 1) & (work["market_rank"] <= 2)
        ].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()

        bets["stake"] = self.stake
        bets["profit"] = np.where(
            bets[self.label_col] == 1,
            bets["stake"] * bets[self.odds_col] - bets["stake"],
            -bets["stake"],
        )
        bets["rule"] = "D_model1_mkt_top2"
        return bets

    def summarize(self, bets: pd.DataFrame) -> dict:
        if bets is None or bets.empty:
            return {
                "n_bets": 0,
                "n_wins": 0,
                "hit_rate": float("nan"),
                "total_stake": 0.0,
                "total_profit": 0.0,
                "roi": float("nan"),
                "max_drawdown": 0.0,
                "avg_odds": float("nan"),
                "equity_curve": pd.Series(dtype=float),
            }

        bets = bets.sort_values(
            by=[c for c in ["race_date", "date", self.race_col] if c in bets.columns]
        ).copy()

        total_stake = float(bets["stake"].sum())
        total_profit = float(bets["profit"].sum())
        n_wins = int((bets[self.label_col] == 1).sum())
        equity = bets["profit"].cumsum()
        peak = equity.cummax()
        drawdown = equity - peak
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0

        return {
            "n_bets": int(len(bets)),
            "n_wins": n_wins,
            "hit_rate": float(n_wins / len(bets)) if len(bets) else float("nan"),
            "total_stake": total_stake,
            "total_profit": total_profit,
            "roi": float(total_profit / total_stake) if total_stake > 0 else float("nan"),
            "max_drawdown": max_dd,
            "avg_odds": float(bets[self.odds_col].mean()),
            "equity_curve": equity,
        }