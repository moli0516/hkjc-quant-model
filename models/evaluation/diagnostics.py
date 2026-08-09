"""Walk-forward 診斷物件：大熱 baseline、賠率畫像、分層、試閘 residual。"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class WalkForwardDiagnostics:
    """對 Walk-forward 的 predictions 做結構化診斷。

    用法::

        diag = WalkForwardDiagnostics(stake=1.0)
        report = diag.run(predictions_df)          # 一鍵 1–4
        # 或分步：
        # diag.rule_market_top1(df)
        # diag.model_top1_odds_profile(df)
        # diag.stratified_model_vs_market(df)
        # diag.trial_residual_multi(df)
    """

    DEFAULT_ODDS_BINS = [0, 3, 5, 8, 15, 30, np.inf]
    DEFAULT_ODDS_LABELS = ["<3", "3-5", "5-8", "8-15", "15-30", ">=30"]

    DEFAULT_TRIAL_FLAGS = [
        "is_strong_trial",
        "is_close_trial",
        "is_fresh_trial_7_28d",
        "last_trial_pass_flag",
    ]

    def __init__(
        self,
        stake: float = 1.0,
        race_col: str = "race_id",
        horse_col: str = "horse_id",
        model_rank_col: str = "model_rank",
        market_rank_col: str = "market_rank",
        odds_col: str = "win_odds",
        label_col: str = "placing",
        date_col: str = "date",
        odds_bins: Optional[list] = None,
        odds_labels: Optional[list] = None,
        trial_flag_cols: Optional[list[str]] = None,
        residual_fav_odds_low: float = 2.0,
        residual_fav_odds_high: float = 5.0,
    ):
        self.stake = stake
        self.race_col = race_col
        self.horse_col = horse_col
        self.model_rank_col = model_rank_col
        self.market_rank_col = market_rank_col
        self.odds_col = odds_col
        self.label_col = label_col
        self.date_col = date_col
        self.odds_bins = odds_bins or list(self.DEFAULT_ODDS_BINS)
        self.odds_labels = odds_labels or list(self.DEFAULT_ODDS_LABELS)
        self.trial_flag_cols = trial_flag_cols or list(self.DEFAULT_TRIAL_FLAGS)
        self.residual_fav_odds_low = residual_fav_odds_low
        self.residual_fav_odds_high = residual_fav_odds_high

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    def _ensure_cols(self, df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"predictions 缺少欄位: {missing}")

    def _settle_fixed_stake(self, bets: pd.DataFrame) -> pd.DataFrame:
        out = bets.copy()
        out["stake"] = self.stake
        won = out[self.label_col] == 1
        out["profit"] = np.where(
            won,
            out["stake"] * out[self.odds_col] - out["stake"],
            -out["stake"],
        )
        return out

    def summarize_bets(self, bets: pd.DataFrame) -> dict:
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
                "median_odds": float("nan"),
            }

        sort_cols = [c for c in [self.date_col, self.race_col] if c in bets.columns]
        bets = bets.sort_values(by=sort_cols).copy() if sort_cols else bets.copy()

        total_stake = float(bets["stake"].sum())
        total_profit = float(bets["profit"].sum())
        n_wins = int((bets[self.label_col] == 1).sum())
        equity = bets["profit"].cumsum()
        max_dd = float((equity - equity.cummax()).min()) if len(equity) else 0.0

        return {
            "n_bets": int(len(bets)),
            "n_wins": n_wins,
            "hit_rate": float(n_wins / len(bets)),
            "total_stake": total_stake,
            "total_profit": total_profit,
            "roi": float(total_profit / total_stake) if total_stake else float("nan"),
            "max_drawdown": max_dd,
            "avg_odds": float(bets[self.odds_col].mean()),
            "median_odds": float(bets[self.odds_col].median()),
        }

    def _pick_rank1(
        self, df: pd.DataFrame, rank_col: str, rule_name: str
    ) -> pd.DataFrame:
        self._ensure_cols(
            df, [self.race_col, rank_col, self.odds_col, self.label_col]
        )
        work = df.dropna(subset=[rank_col, self.odds_col, self.label_col]).copy()
        work = work[work[self.odds_col] > 0]
        bets = work[work[rank_col] == 1].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()
        bets = self._settle_fixed_stake(bets)
        bets["rule"] = rule_name
        return bets

    # ------------------------------------------------------------------
    # 1) 大熱 / 模型第一
    # ------------------------------------------------------------------
    def rule_market_top1(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pick_rank1(df, self.market_rank_col, "M_market_top1")

    def rule_model_top1(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pick_rank1(df, self.model_rank_col, "A_model_top1")

    # ------------------------------------------------------------------
    # 2) 模型第一賠率畫像
    # ------------------------------------------------------------------
    def model_top1_odds_profile(
        self, df: pd.DataFrame
    ) -> tuple[dict, pd.DataFrame]:
        self._ensure_cols(
            df,
            [self.race_col, self.model_rank_col, self.odds_col, self.label_col],
        )
        top1 = df.dropna(
            subset=[self.model_rank_col, self.odds_col, self.label_col]
        ).copy()
        top1 = top1[
            (top1[self.model_rank_col] == 1) & (top1[self.odds_col] > 0)
        ]
        top1 = top1.groupby(self.race_col, as_index=False).first()

        summary = {
            "n": int(len(top1)),
            "win_rate": float((top1[self.label_col] == 1).mean())
            if len(top1)
            else float("nan"),
            "avg_odds": float(top1[self.odds_col].mean())
            if len(top1)
            else float("nan"),
            "median_odds": float(top1[self.odds_col].median())
            if len(top1)
            else float("nan"),
            "p25_odds": float(top1[self.odds_col].quantile(0.25))
            if len(top1)
            else float("nan"),
            "p75_odds": float(top1[self.odds_col].quantile(0.75))
            if len(top1)
            else float("nan"),
        }

        top1 = top1.copy()
        top1["odds_bin"] = pd.cut(
            top1[self.odds_col],
            bins=self.odds_bins,
            labels=self.odds_labels,
            right=False,
        )

        rows = []
        for b, g in top1.groupby("odds_bin", observed=False):
            n = len(g)
            wins = int((g[self.label_col] == 1).sum())
            avg_odds = float(g[self.odds_col].mean()) if n else float("nan")
            profit = float(
                (
                    (g[self.label_col] == 1) * (g[self.odds_col] - 1.0)
                    + (g[self.label_col] != 1) * (-1.0)
                ).sum()
            )
            rows.append(
                {
                    "odds_bin": str(b),
                    "n": n,
                    "wins": wins,
                    "win_rate": float(wins / n) if n else float("nan"),
                    "avg_odds": avg_odds,
                    "implied_fair_approx": float(1.0 / avg_odds)
                    if n and avg_odds > 0
                    else float("nan"),
                    "roi_flat1": float(profit / n) if n else float("nan"),
                }
            )
        return summary, pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3) 分層 model vs market
    # ------------------------------------------------------------------
    def race_level_hits(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_cols(
            df,
            [
                self.race_col,
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
            ],
        )
        work = df.dropna(
            subset=[
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
            ]
        ).copy()
        work = work[work[self.odds_col] > 0]

        model_top = (
            work[work[self.model_rank_col] == 1]
            .groupby(self.race_col, as_index=False)
            .first()[[self.race_col, self.label_col, self.odds_col]]
            .rename(
                columns={
                    self.label_col: "model_placing",
                    self.odds_col: "model_pick_odds",
                }
            )
        )
        market_top = (
            work[work[self.market_rank_col] == 1]
            .groupby(self.race_col, as_index=False)
            .first()[[self.race_col, self.label_col, self.odds_col]]
            .rename(
                columns={
                    self.label_col: "market_placing",
                    self.odds_col: "market_fav_odds",
                }
            )
        )

        races = model_top.merge(market_top, on=self.race_col, how="inner")
        races["model_hit"] = (races["model_placing"] == 1).astype(float)
        races["market_hit"] = (races["market_placing"] == 1).astype(float)

        if self.horse_col in work.columns:
            agree_map = {}
            for rid, g in work.groupby(self.race_col):
                m = set(g.loc[g[self.model_rank_col] == 1, self.horse_col])
                k = set(g.loc[g[self.market_rank_col] == 1, self.horse_col])
                agree_map[rid] = bool(m & k)
            races["agree"] = races[self.race_col].map(agree_map)
        else:
            races["agree"] = np.nan

        return races

    def stratified_model_vs_market(self, df: pd.DataFrame) -> dict:
        races = self.race_level_hits(df)
        races["fav_odds_bin"] = pd.cut(
            races["market_fav_odds"],
            bins=self.odds_bins,
            labels=self.odds_labels,
            right=False,
        )

        def _agg(g: pd.DataFrame) -> pd.Series:
            n = len(g)
            return pd.Series(
                {
                    "n_races": n,
                    "model_top1": float(g["model_hit"].mean()) if n else float("nan"),
                    "market_top1": float(g["market_hit"].mean())
                    if n
                    else float("nan"),
                    "gap_model_minus_market": float(
                        g["model_hit"].mean() - g["market_hit"].mean()
                    )
                    if n
                    else float("nan"),
                    "avg_fav_odds": float(g["market_fav_odds"].mean())
                    if n
                    else float("nan"),
                    "avg_model_pick_odds": float(g["model_pick_odds"].mean())
                    if n
                    else float("nan"),
                }
            )

        by_fav = (
            races.groupby("fav_odds_bin", observed=False)
            .apply(_agg, include_groups=False)
            .reset_index()
        )

        if races["agree"].notna().any():
            races = races.copy()
            races["agree_flag"] = races["agree"].map(
                {True: "same_pick", False: "disagree"}
            )
            by_agree = (
                races.groupby("agree_flag", observed=False)
                .apply(_agg, include_groups=False)
                .reset_index()
            )
        else:
            by_agree = pd.DataFrame()

        overall = _agg(races).to_frame().T
        overall.insert(0, "slice", "ALL")

        return {
            "overall": overall,
            "by_fav_odds": by_fav,
            "by_agree": by_agree,
            "races": races,
        }

    # ------------------------------------------------------------------
    # 4) 試閘 residual
    # ------------------------------------------------------------------
    def trial_residual_in_odds_band(
        self,
        df: pd.DataFrame,
        trial_flag_col: str,
        fav_odds_low: Optional[float] = None,
        fav_odds_high: Optional[float] = None,
    ) -> pd.DataFrame:
        low = self.residual_fav_odds_low if fav_odds_low is None else fav_odds_low
        high = self.residual_fav_odds_high if fav_odds_high is None else fav_odds_high

        self._ensure_cols(
            df,
            [
                self.race_col,
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
                trial_flag_col,
            ],
        )

        races = self.race_level_hits(df)
        band_ids = races.loc[
            (races["market_fav_odds"] >= low) & (races["market_fav_odds"] < high),
            self.race_col,
        ]
        sub = df[df[self.race_col].isin(band_ids)].copy()
        picks = sub[sub[self.model_rank_col] == 1].dropna(
            subset=[trial_flag_col, self.label_col]
        )
        picks = picks.groupby(self.race_col, as_index=False).first()

        rows = []
        for flag_val, g in picks.groupby(trial_flag_col):
            n = len(g)
            wins = int((g[self.label_col] == 1).sum())
            rows.append(
                {
                    "fav_odds_band": f"[{low},{high})",
                    "trial_flag": trial_flag_col,
                    "flag_value": flag_val,
                    "n": n,
                    "wins": wins,
                    "win_rate": float(wins / n) if n else float("nan"),
                    "avg_odds": float(g[self.odds_col].mean()) if n else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def trial_residual_multi(
        self,
        df: pd.DataFrame,
        flag_cols: Optional[list[str]] = None,
        fav_odds_low: Optional[float] = None,
        fav_odds_high: Optional[float] = None,
    ) -> pd.DataFrame:
        cols = flag_cols or self.trial_flag_cols
        parts = []
        for col in cols:
            if col not in df.columns:
                continue
            parts.append(
                self.trial_residual_in_odds_band(
                    df,
                    trial_flag_col=col,
                    fav_odds_low=fav_odds_low,
                    fav_odds_high=fav_odds_high,
                )
            )
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # 一鍵入口
    # ------------------------------------------------------------------
    def run(self, predictions: pd.DataFrame, print_report: bool = True) -> dict:
        self._ensure_cols(
            predictions,
            [
                self.race_col,
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
            ],
        )

        sum_m = self.summarize_bets(self.rule_market_top1(predictions))
        sum_a = self.summarize_bets(self.rule_model_top1(predictions))
        odds_summary, odds_bins = self.model_top1_odds_profile(predictions)
        strata = self.stratified_model_vs_market(predictions)
        residual = self.trial_residual_multi(predictions)

        report = {
            "rule_M_market": sum_m,
            "rule_A_model": sum_a,
            "model_top1_odds_summary": odds_summary,
            "model_top1_odds_bins": odds_bins,
            "strata": strata,
            "trial_residual": residual,
        }

        if print_report:
            self.print_report(report)
        return report

    def print_report(self, report: dict) -> None:
        def _fmt(d: dict) -> None:
            for k, v in d.items():
                if isinstance(v, float):
                    print(f"   ├─ {k}: {v:.6f}")
                else:
                    print(f"   ├─ {k}: {v}")

        print("\n" + "=" * 60)
        print("📌 [1] 大熱 baseline (M) vs 模型第一 (A)")
        print("=" * 60)
        print("▶ 規則 M — market_rank==1")
        _fmt(report["rule_M_market"])
        print("▶ 規則 A — model_rank==1")
        _fmt(report["rule_A_model"])

        print("\n" + "=" * 60)
        print("📌 [2] 模型第一的賠率畫像")
        print("=" * 60)
        _fmt(report["model_top1_odds_summary"])
        print("\n分箱：")
        print(report["model_top1_odds_bins"].to_string(index=False))

        print("\n" + "=" * 60)
        print("📌 [3] 分層 model vs market（依大熱賠率）")
        print("=" * 60)
        print(report["strata"]["by_fav_odds"].to_string(index=False))
        by_agree = report["strata"].get("by_agree")
        if by_agree is not None and len(by_agree):
            print("\n依是否與大熱同選：")
            print(by_agree.to_string(index=False))

        print("\n" + "=" * 60)
        print(
            f"📌 [4] 試閘 residual（大熱賠率 "
            f"{self.residual_fav_odds_low}–{self.residual_fav_odds_high}）"
        )
        print("=" * 60)
        tr = report["trial_residual"]
        if tr is None or tr.empty:
            print("（無可用試閘旗標或樣本為空）")
        else:
            print(tr.to_string(index=False))
        print("=" * 60 + "\n")