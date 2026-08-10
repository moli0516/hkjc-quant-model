"""具體規則定義（代號穩定，不覆蓋歷史語義）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.evaluation.rules.base import BettingRule

DEFAULT_REPORT_RULE_IDS = ["M0", "A0", "B0", "C1", "C3", "C4", "D0", "E0"]
def _base_filter(
    df: pd.DataFrame,
    rank_col: str = "model_rank",
    odds_col: str = "win_odds",
    label_col: str = "placing",
    extra: list[str] | None = None,
) -> pd.DataFrame:
    cols = [rank_col, odds_col, label_col] + (extra or [])
    cols = [c for c in cols if c in df.columns or c in (extra or [])]
    # 必備
    need = [odds_col, label_col]
    work = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    work = work[work[odds_col] > 0]
    return work


class RuleM0(BettingRule):
    """大熱 baseline：market_rank == 1"""

    rule_id = "M0"
    name = "market_top1"
    description = "market_rank == 1"
    required_cols = ("market_rank", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df)
        if "market_rank" not in work.columns:
            return work.iloc[0:0]
        return work[work["market_rank"] == 1].copy()


class RuleA0(BettingRule):
    """模型第一：model_rank == 1"""

    rule_id = "A0"
    name = "model_top1"
    description = "model_rank == 1"
    required_cols = ("model_rank", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        ctx = ctx or {}
        rank_col = ctx.get("rank_col", "model_rank")
        work = _base_filter(df, rank_col=rank_col)
        if rank_col not in work.columns:
            return work.iloc[0:0]
        return work[work[rank_col] == 1].copy()


class RuleB0(BettingRule):
    """Overlay：model_prob > market_prob * threshold 且 model_rank <= 3"""

    rule_id = "B0"
    name = "overlay_value"
    description = "model_prob > market_prob * threshold & model_rank <= 3"
    required_cols = ("model_rank", "model_score", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        ctx = ctx or {}
        race_col = ctx.get("race_col", "race_id")
        score_col = ctx.get("score_col", "model_score")
        rank_col = ctx.get("rank_col", "model_rank")
        odds_col = ctx.get("odds_col", "win_odds")
        threshold = float(ctx.get("overlay_threshold", 1.15))

        work = df.dropna(
            subset=[rank_col, odds_col, "placing", score_col]
        ).copy()
        work = work[work[odds_col] > 0]
        if work.empty:
            return work

        def _sm(s: pd.Series) -> pd.Series:
            x = s.astype(float).values
            x = x - np.nanmax(x)
            ex = np.exp(x)
            ex = np.where(np.isfinite(ex), ex, 0.0)
            denom = ex.sum()
            if denom <= 0:
                return pd.Series(np.ones(len(s)) / max(len(s), 1), index=s.index)
            return pd.Series(ex / denom, index=s.index)

        work["model_prob"] = work.groupby(race_col)[score_col].transform(_sm)
        work["raw_market_prob"] = 1.0 / work[odds_col]
        work["market_prob"] = work.groupby(race_col)["raw_market_prob"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else s
        )
        mask = (work["model_prob"] > work["market_prob"] * threshold) & (
            work[rank_col] <= 3
        )
        return work.loc[mask].copy()


class RuleC0(BettingRule):
    """同選大熱：model_rank == 1 且 market_rank == 1"""

    rule_id = "C0"
    name = "same_pick"
    description = "model_rank == 1 & market_rank == 1"
    required_cols = ("model_rank", "market_rank", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank"])
        if "market_rank" not in work.columns or "model_rank" not in work.columns:
            return work.iloc[0:0]
        return work[
            (work["model_rank"] == 1) & (work["market_rank"] == 1)
        ].copy()


class RuleC1(BettingRule):
    """同選 + 強試閘（目前相對最佳用法）"""

    rule_id = "C1"
    name = "same_pick_strong_trial"
    description = "model_rank == 1 & market_rank == 1 & is_strong_trial == 1"
    required_cols = (
        "model_rank",
        "market_rank",
        "is_strong_trial",
        "win_odds",
        "placing",
    )

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank", "is_strong_trial"])
        need = ["model_rank", "market_rank", "is_strong_trial"]
        if any(c not in work.columns for c in need):
            return work.iloc[0:0]
        work = work.dropna(subset=["is_strong_trial"])
        return work[
            (work["model_rank"] == 1)
            & (work["market_rank"] == 1)
            & (work["is_strong_trial"] == 1)
        ].copy()


class RuleC2(BettingRule):
    """同選 + 弱試閘（C1 對照）"""

    rule_id = "C2"
    name = "same_pick_weak_trial"
    description = "model_rank == 1 & market_rank == 1 & is_strong_trial == 0"
    required_cols = (
        "model_rank",
        "market_rank",
        "is_strong_trial",
        "win_odds",
        "placing",
    )

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank", "is_strong_trial"])
        need = ["model_rank", "market_rank", "is_strong_trial"]
        if any(c not in work.columns for c in need):
            return work.iloc[0:0]
        work = work.dropna(subset=["is_strong_trial"])
        return work[
            (work["model_rank"] == 1)
            & (work["market_rank"] == 1)
            & (work["is_strong_trial"] == 0)
        ].copy()

class RuleC3(BettingRule):
    """同選 + 新鮮試閘"""

    rule_id = "C3"
    name = "same_pick_fresh_trial"
    description = "model_rank == 1 & market_rank == 1 & is_fresh_trial_7_28d == 1"
    required_cols = (
        "model_rank",
        "market_rank",
        "is_fresh_trial_7_28d",
        "win_odds",
        "placing",
    )

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank", "is_fresh_trial_7_28d"])
        need = ["model_rank", "market_rank", "is_fresh_trial_7_28d"]
        if any(c not in work.columns for c in need):
            return work.iloc[0:0]
        work = work.dropna(subset=["is_fresh_trial_7_28d"])
        return work[
            (work["model_rank"] == 1)
            & (work["market_rank"] == 1)
            & (work["is_fresh_trial_7_28d"] == 1)
        ].copy()


class RuleC4(BettingRule):
    """同選 + 非新鮮試閘（C3 對照）"""

    rule_id = "C4"
    name = "same_pick_not_fresh_trial"
    description = "model_rank == 1 & market_rank == 1 & is_fresh_trial_7_28d == 0"
    required_cols = (
        "model_rank",
        "market_rank",
        "is_fresh_trial_7_28d",
        "win_odds",
        "placing",
    )

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank", "is_fresh_trial_7_28d"])
        need = ["model_rank", "market_rank", "is_fresh_trial_7_28d"]
        if any(c not in work.columns for c in need):
            return work.iloc[0:0]
        work = work.dropna(subset=["is_fresh_trial_7_28d"])
        return work[
            (work["model_rank"] == 1)
            & (work["market_rank"] == 1)
            & (work["is_fresh_trial_7_28d"] == 0)
        ].copy()

class RuleD0(BettingRule):
    """模型第一且市場前二"""

    rule_id = "D0"
    name = "model_top1_market_top2"
    description = "model_rank == 1 & market_rank <= 2"
    required_cols = ("model_rank", "market_rank", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank"])
        if "market_rank" not in work.columns or "model_rank" not in work.columns:
            return work.iloc[0:0]
        return work[
            (work["model_rank"] == 1) & (work["market_rank"] <= 2)
        ].copy()


class RuleD1(BettingRule):
    """模型第一且市場前三（可選實驗）"""

    rule_id = "D1"
    name = "model_top1_market_top3"
    description = "model_rank == 1 & market_rank <= 3"
    required_cols = ("model_rank", "market_rank", "win_odds", "placing")

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank"])
        if "market_rank" not in work.columns or "model_rank" not in work.columns:
            return work.iloc[0:0]
        return work[
            (work["model_rank"] == 1) & (work["market_rank"] <= 3)
        ].copy()
        
class RuleE0(BettingRule):
    """同選 + 強試閘 + 新鮮試閘（C1 ^ C3）"""

    rule_id = "E0"
    name = "same_pick_strong_trial_fresh_trail"
    description = "model_rank == 1 & market_rank == 1 & is_strong_trial == 1 & is_fresh_trial_7_28d == 1"
    required_cols = (
        "model_rank",
        "market_rank",
        "is_strong_trial",
        "is_fresh_trial_7_28d",
        "win_odds",
        "placing",
    )

    def select(self, df: pd.DataFrame, ctx: dict | None = None) -> pd.DataFrame:
        work = _base_filter(df, extra=["market_rank", "is_strong_trial", "is_fresh_trial_7_28d"])
        need = ["model_rank", "market_rank", "is_strong_trial", "is_fresh_trial_7_28d"]
        if any(c not in work.columns for c in need):
            return work.iloc[0:0]
        work = work.dropna(subset=["is_strong_trial", "is_fresh_trial_7_28d"])
        return work[
            (work["model_rank"] == 1)
            & (work["market_rank"] == 1)
            & (work["is_strong_trial"] == 1)
            & (work["is_fresh_trial_7_28d"] == 1)
        ].copy()


# 預設註冊用的實例工廠
def all_builtin_rules() -> list[BettingRule]:
    return [
        RuleM0(),
        RuleA0(),
        RuleB0(),
        RuleC0(),
        RuleC1(),
        RuleC2(),
        RuleC3(),
        RuleC4(),
        RuleD0(),
        RuleD1(),
        RuleE0()
    ]
