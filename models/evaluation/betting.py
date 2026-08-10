"""下注結算與規則執行（規則本體在 rules/）。"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from models.evaluation.rules.base import BettingRule
from models.evaluation.rules.registry import (
    RuleRegistry,
    default_registry,
    default_report_ids,
)


class BetEvaluator:
    """固定注碼結算 + 透過 RuleRegistry 執行規則。

    相容舊 API：rule_a / rule_b / rule_c_same_pick / ...
    新 API：run_rule / run_many
    """

    def __init__(
        self,
        overlay_threshold: float = 1.15,
        stake: float = 1.0,
        race_col: str = "race_id",
        odds_col: str = "win_odds",
        label_col: str = "placing",
        score_col: str = "model_score",
        rank_col: str = "model_rank",
        registry: Optional[RuleRegistry] = None,
    ):
        self.overlay_threshold = overlay_threshold
        self.stake = stake
        self.race_col = race_col
        self.odds_col = odds_col
        self.label_col = label_col
        self.score_col = score_col
        self.rank_col = rank_col
        self.registry = registry or default_registry()

    def _ctx(self) -> dict:
        return {
            "overlay_threshold": self.overlay_threshold,
            "race_col": self.race_col,
            "odds_col": self.odds_col,
            "label_col": self.label_col,
            "score_col": self.score_col,
            "rank_col": self.rank_col,
            "stake": self.stake,
        }

    def _settle(self, bets: pd.DataFrame, rule_id: str = "") -> pd.DataFrame:
        if bets is None or bets.empty:
            return pd.DataFrame()

        out = bets.groupby(self.race_col, as_index=False).first().copy()
        out["stake"] = self.stake
        out["profit"] = np.where(
            out[self.label_col] == 1,
            out["stake"] * out[self.odds_col] - out["stake"],
            -out["stake"],
        )
        out["rule_id"] = rule_id
        out["rule"] = rule_id  # 相容舊欄位名
        return out

    # ------------------------------------------------------------------
    # 新 API
    # ------------------------------------------------------------------
    def run_rule(
        self, df: pd.DataFrame, rule: BettingRule | str
    ) -> pd.DataFrame:
        if isinstance(rule, str):
            rule = self.registry.get(rule)
        if not rule.is_available(df):
            return pd.DataFrame()
        selected = rule.select(df, ctx=self._ctx())
        if selected is None or selected.empty:
            return pd.DataFrame()
        return self._settle(selected, rule_id=rule.rule_id)

    def run_many(
        self,
        df: pd.DataFrame,
        rule_ids: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """執行多條規則並 summarize。回傳 {rule_id: summary_dict}。"""
        ids = rule_ids if rule_ids is not None else default_report_ids()
        out: dict[str, dict] = {}
        for rid in ids:
            if not self.registry.has(rid):
                continue
            bets = self.run_rule(df, rid)
            summary = self.summarize(bets)
            summary["rule_id"] = rid
            if self.registry.has(rid):
                summary["rule_name"] = self.registry.get(rid).name
            out[rid] = summary
            
        return out

    def run_many_bets(
        self,
        df: pd.DataFrame,
        rule_ids: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        ids = rule_ids if rule_ids is not None else default_report_ids()
        return {
            rid: self.run_rule(df, rid)
            for rid in ids
            if self.registry.has(rid)
        }

    # ------------------------------------------------------------------
    # 統計
    # ------------------------------------------------------------------
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

        sort_cols = [
            c for c in ["race_date", "date", self.race_col] if c in bets.columns
        ]
        bets = bets.sort_values(by=sort_cols).copy() if sort_cols else bets.copy()

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