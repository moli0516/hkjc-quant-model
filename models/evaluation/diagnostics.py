"""Walk-forward 診斷物件：大熱 baseline、賠率畫像、分層、試閘 residual、fold 穩定性。"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


class WalkForwardDiagnostics:
    """對 Walk-forward 的 predictions 做結構化診斷。

    用法::

        diag = WalkForwardDiagnostics(stake=1.0)
        report = diag.run(predictions_df)          # 一鍵 1–4
        diag.evaluate_h1_stability(predictions_df)  # C1 vs A0
        diag.evaluate_c3_stability(predictions_df)  # C3 vs A0
        diag.evaluate_e0_stability(predictions_df)  # E0 vs A0
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

    def _same_pick_base(self, df: pd.DataFrame) -> pd.DataFrame:
        """model_rank==1 且 market_rank==1 的列（未 groupby）。"""
        need = [
            self.race_col,
            self.model_rank_col,
            self.market_rank_col,
            self.odds_col,
            self.label_col,
        ]
        self._ensure_cols(df, need)
        work = df.dropna(
            subset=[
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
            ]
        ).copy()
        work = work[work[self.odds_col] > 0]
        mask = (work[self.model_rank_col] == 1) & (work[self.market_rank_col] == 1)
        return work.loc[mask].copy()

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
    # 注單建構：A / M / C1 / C3 / E0
    # ------------------------------------------------------------------
    def _bets_rule_a(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pick_rank1(df, self.model_rank_col, "A0")

    def _bets_rule_m(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._pick_rank1(df, self.market_rank_col, "M0")

    def _bets_rule_c1(
        self,
        df: pd.DataFrame,
        strong_trial_col: str = "is_strong_trial",
    ) -> pd.DataFrame:
        """C1: same_pick + is_strong_trial == 1"""
        self._ensure_cols(df, [strong_trial_col])
        work = self._same_pick_base(df)
        work = work.dropna(subset=[strong_trial_col])
        bets = work[work[strong_trial_col] == 1].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()
        bets = self._settle_fixed_stake(bets)
        bets["rule"] = "C1"
        return bets

    def _bets_rule_c3(
        self,
        df: pd.DataFrame,
        fresh_col: str = "is_fresh_trial_7_28d",
    ) -> pd.DataFrame:
        """C3: same_pick + is_fresh_trial_7_28d == 1"""
        self._ensure_cols(df, [fresh_col])
        work = self._same_pick_base(df)
        work = work.dropna(subset=[fresh_col])
        bets = work[work[fresh_col] == 1].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()
        bets = self._settle_fixed_stake(bets)
        bets["rule"] = "C3"
        return bets

    def _bets_rule_e0(
        self,
        df: pd.DataFrame,
        strong_trial_col: str = "is_strong_trial",
        fresh_col: str = "is_fresh_trial_7_28d",
    ) -> pd.DataFrame:
        """E0: same_pick + strong + fresh"""
        self._ensure_cols(df, [strong_trial_col, fresh_col])
        work = self._same_pick_base(df)
        work = work.dropna(subset=[strong_trial_col, fresh_col])
        bets = work[
            (work[strong_trial_col] == 1) & (work[fresh_col] == 1)
        ].copy()
        bets = bets.groupby(self.race_col, as_index=False).first()
        bets = self._settle_fixed_stake(bets)
        bets["rule"] = "E0"
        return bets

    # 相容舊名稱
    def _bets_rule_c(
        self,
        df: pd.DataFrame,
        require_strong_trial: bool = True,
        strong_trial_col: str = "is_strong_trial",
    ) -> pd.DataFrame:
        if require_strong_trial:
            return self._bets_rule_c1(df, strong_trial_col=strong_trial_col)
        work = self._same_pick_base(df)
        bets = work.groupby(self.race_col, as_index=False).first()
        bets = self._settle_fixed_stake(bets)
        bets["rule"] = "C0"
        return bets

    def _summarize_bets_by_fold(
        self,
        bets: pd.DataFrame,
        fold_col: str = "fold_id",
    ) -> pd.DataFrame:
        if bets is None or bets.empty:
            return pd.DataFrame(
                columns=[
                    fold_col,
                    "rule",
                    "n_bets",
                    "n_wins",
                    "hit_rate",
                    "roi",
                    "avg_odds",
                ]
            )
        if fold_col not in bets.columns:
            raise KeyError(
                f"注單缺少 {fold_col}。請確認 Walk-forward 的 predictions 含 fold_id。"
            )

        rows = []
        rule = bets["rule"].iloc[0] if "rule" in bets.columns else "unknown"
        for fold, g in bets.groupby(fold_col, sort=True):
            n = len(g)
            wins = int((g[self.label_col] == 1).sum())
            stake_sum = float(g["stake"].sum())
            profit_sum = float(g["profit"].sum())
            rows.append(
                {
                    fold_col: fold,
                    "rule": rule,
                    "n_bets": n,
                    "n_wins": wins,
                    "hit_rate": float(wins / n) if n else float("nan"),
                    "roi": float(profit_sum / stake_sum) if stake_sum else float("nan"),
                    "avg_odds": float(g[self.odds_col].mean()) if n else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def _attach_fold(
        self,
        bets: pd.DataFrame,
        predictions: pd.DataFrame,
        fold_col: str = "fold_id",
    ) -> pd.DataFrame:
        if bets is None or bets.empty:
            return bets

        if fold_col in bets.columns and bets[fold_col].notna().all():
            return bets

        if fold_col not in predictions.columns:
            raise KeyError(
                f"predictions 缺少 {fold_col}，無法做跨 fold 分析。"
            )

        race_fold = (
            predictions[[self.race_col, fold_col]]
            .dropna(subset=[self.race_col, fold_col])
            .drop_duplicates(subset=[self.race_col], keep="first")
        )
        out = bets.drop(columns=[fold_col], errors="ignore")
        out = out.merge(race_fold, on=self.race_col, how="left")
        if fold_col not in out.columns:
            raise KeyError(f"merge 後仍無 {fold_col}，請檢查 race_id。")
        n_miss = int(out[fold_col].isna().sum())
        if n_miss:
            raise ValueError(
                f"有 {n_miss} 注無法對到 {fold_col}，請檢查 predictions。"
            )
        return out

    # ------------------------------------------------------------------
    # 通用 fold 穩定性：treatment vs control（+ 可選 market）
    # ------------------------------------------------------------------
    def evaluate_fold_stability(
        self,
        predictions: pd.DataFrame,
        bets_treatment: pd.DataFrame,
        bets_control: pd.DataFrame,
        treatment_id: str,
        control_id: str,
        bets_market: Optional[pd.DataFrame] = None,
        fold_col: str = "fold_id",
        min_n_control: int = 50,
        min_n_treatment: int = 20,
        min_frac_folds: float = 2.0 / 3.0,
        print_report: bool = True,
        title: Optional[str] = None,
    ) -> dict:
        """
        多數有效 fold 上 treatment.hit > control.hit，
        且 overall ROI_treatment >= ROI_control → decision=support。
        """
        self._ensure_cols(
            predictions,
            [
                self.race_col,
                self.model_rank_col,
                self.market_rank_col,
                self.odds_col,
                self.label_col,
                fold_col,
            ],
        )

        if bets_market is None:
            bets_market = self._bets_rule_m(predictions)

        bets_t = self._attach_fold(bets_treatment, predictions, fold_col)
        bets_c = self._attach_fold(bets_control, predictions, fold_col)
        bets_m = self._attach_fold(bets_market, predictions, fold_col)

        by_t = self._summarize_bets_by_fold(bets_t, fold_col=fold_col)
        by_c = self._summarize_bets_by_fold(bets_c, fold_col=fold_col)
        by_m = self._summarize_bets_by_fold(bets_m, fold_col=fold_col)

        t_prefix, c_prefix = "T", "C"
        wide = (
            by_c[[fold_col, "n_bets", "hit_rate", "roi"]]
            .rename(
                columns={
                    "n_bets": f"n_{c_prefix}",
                    "hit_rate": f"hit_{c_prefix}",
                    "roi": f"roi_{c_prefix}",
                }
            )
            .merge(
                by_t[[fold_col, "n_bets", "hit_rate", "roi"]].rename(
                    columns={
                        "n_bets": f"n_{t_prefix}",
                        "hit_rate": f"hit_{t_prefix}",
                        "roi": f"roi_{t_prefix}",
                    }
                ),
                on=fold_col,
                how="outer",
            )
            .merge(
                by_m[[fold_col, "n_bets", "hit_rate", "roi"]].rename(
                    columns={
                        "n_bets": "n_M",
                        "hit_rate": "hit_M",
                        "roi": "roi_M",
                    }
                ),
                on=fold_col,
                how="outer",
            )
            .sort_values(fold_col)
            .reset_index(drop=True)
        )

        # 相容 H1 舊欄名：A=control, treatment=C 風格別名
        wide = wide.rename(
            columns={
                f"n_{c_prefix}": "n_control",
                f"hit_{c_prefix}": "hit_control",
                f"roi_{c_prefix}": "roi_control",
                f"n_{t_prefix}": "n_treatment",
                f"hit_{t_prefix}": "hit_treatment",
                f"roi_{t_prefix}": "roi_treatment",
            }
        )
        # 亦提供 H1 風格別名（control=A, treatment 當 C）
        wide["n_A"] = wide["n_control"]
        wide["hit_A"] = wide["hit_control"]
        wide["roi_A"] = wide["roi_control"]
        wide["n_C"] = wide["n_treatment"]
        wide["hit_C"] = wide["hit_treatment"]
        wide["roi_C"] = wide["roi_treatment"]

        wide["delta_hit_T_minus_C"] = wide["hit_treatment"] - wide["hit_control"]
        wide["delta_roi_T_minus_C"] = wide["roi_treatment"] - wide["roi_control"]
        wide["delta_hit_C_minus_A"] = wide["delta_hit_T_minus_C"]
        wide["t_beats_c_hit"] = wide["delta_hit_T_minus_C"] > 0
        wide["c_beats_a_hit"] = wide["t_beats_c_hit"]

        valid = wide[
            (wide["n_control"].fillna(0) >= min_n_control)
            & (wide["n_treatment"].fillna(0) >= min_n_treatment)
        ].copy()
        n_valid = int(len(valid))
        frac_hit = float(valid["t_beats_c_hit"].mean()) if n_valid > 0 else float("nan")

        overall_t = self.summarize_bets(bets_t)
        overall_c = self.summarize_bets(bets_c)
        overall_m = self.summarize_bets(bets_m)

        roi_ok = overall_t["roi"] >= overall_c["roi"]
        if n_valid == 0:
            decision = "evidence_insufficient"
        elif frac_hit >= min_frac_folds and roi_ok:
            decision = "support"
        elif frac_hit >= min_frac_folds and not roi_ok:
            decision = "weak_support_hit_only"
        else:
            decision = "not_supported"

        result = {
            "decision": decision,
            "treatment_id": treatment_id,
            "control_id": control_id,
            "n_valid_folds": n_valid,
            "frac_folds_hit_t_gt_c": frac_hit,
            "frac_folds_hit_c_gt_a": frac_hit,  # H1 相容
            "min_frac_folds": min_frac_folds,
            "min_n_control": min_n_control,
            "min_n_treatment": min_n_treatment,
            "min_n_a": min_n_control,
            "min_n_c": min_n_treatment,
            "overall_treatment": overall_t,
            "overall_control": overall_c,
            "overall_market": overall_m,
            "overall_A": overall_c,
            "overall_C": overall_t,
            "overall_M": overall_m,
            "by_fold": wide,
            "by_fold_valid": valid,
        }

        if print_report:
            self._print_fold_stability_report(
                result, title=title or f"{treatment_id} vs {control_id}"
            )
        return result

    def _print_fold_stability_report(
        self, result: dict, title: str = ""
    ) -> None:
        t_id = result.get("treatment_id", "T")
        c_id = result.get("control_id", "C")
        print("\n" + "=" * 60)
        print(f"📌 Fold 穩定性：{title or (t_id + ' vs ' + c_id)}")
        print("=" * 60)
        print(f"   ├─ decision: {result['decision']}")
        print(f"   ├─ treatment / control: {t_id} / {c_id}")
        print(f"   ├─ n_valid_folds: {result['n_valid_folds']}")
        print(
            f"   ├─ frac_folds (hit_T > hit_C): "
            f"{result['frac_folds_hit_t_gt_c']}"
        )
        print(
            f"   ├─ threshold: >= {result['min_frac_folds']:.3f} "
            f"(min_n_control={result['min_n_control']}, "
            f"min_n_treatment={result['min_n_treatment']})"
        )
        print(
            f"   ├─ overall hit control/treatment/M: "
            f"{result['overall_control']['hit_rate']:.4f} / "
            f"{result['overall_treatment']['hit_rate']:.4f} / "
            f"{result['overall_market']['hit_rate']:.4f}"
        )
        print(
            f"   └─ overall ROI control/treatment/M: "
            f"{result['overall_control']['roi']:.4f} / "
            f"{result['overall_treatment']['roi']:.4f} / "
            f"{result['overall_market']['roi']:.4f}"
        )
        print("\n按 fold：")
        cols = [
            c
            for c in [
                "fold_id",
                "n_A",
                "n_C",
                "n_M",
                "hit_A",
                "hit_C",
                "hit_M",
                "roi_A",
                "roi_C",
                "delta_hit_C_minus_A",
                "c_beats_a_hit",
            ]
            if c in result["by_fold"].columns
        ]
        print(result["by_fold"][cols].to_string(index=False))
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 便捷入口：H1 / C3 / E0
    # ------------------------------------------------------------------
    def evaluate_h1_stability(
        self,
        predictions: pd.DataFrame,
        fold_col: str = "fold_id",
        require_strong_trial: bool = True,
        strong_trial_col: str = "is_strong_trial",
        min_n_a: int = 50,
        min_n_c: int = 20,
        min_frac_folds: float = 2.0 / 3.0,
        print_report: bool = True,
    ) -> dict:
        """H1：C1（同選+強試閘）vs A0。"""
        bets_a = self._bets_rule_a(predictions)
        if require_strong_trial:
            bets_t = self._bets_rule_c1(
                predictions, strong_trial_col=strong_trial_col
            )
            t_id = "C1"
        else:
            bets_t = self._bets_rule_c(
                predictions, require_strong_trial=False
            )
            t_id = "C0"
        result = self.evaluate_fold_stability(
            predictions,
            bets_treatment=bets_t,
            bets_control=bets_a,
            treatment_id=t_id,
            control_id="A0",
            fold_col=fold_col,
            min_n_control=min_n_a,
            min_n_treatment=min_n_c,
            min_frac_folds=min_frac_folds,
            print_report=print_report,
            title=f"H1 {t_id} vs A0",
        )
        result["require_strong_trial"] = require_strong_trial
        return result

    def evaluate_c3_stability(
        self,
        predictions: pd.DataFrame,
        fold_col: str = "fold_id",
        fresh_col: str = "is_fresh_trial_7_28d",
        control: str = "A0",
        min_n_control: int = 50,
        min_n_treatment: int = 20,
        min_frac_folds: float = 2.0 / 3.0,
        print_report: bool = True,
    ) -> dict:
        """C3（同選+fresh）vs control（預設 A0；可傳 C1）。"""
        bets_t = self._bets_rule_c3(predictions, fresh_col=fresh_col)
        if control == "C1":
            bets_c = self._bets_rule_c1(predictions)
            c_id = "C1"
        else:
            bets_c = self._bets_rule_a(predictions)
            c_id = "A0"
        return self.evaluate_fold_stability(
            predictions,
            bets_treatment=bets_t,
            bets_control=bets_c,
            treatment_id="C3",
            control_id=c_id,
            fold_col=fold_col,
            min_n_control=min_n_control,
            min_n_treatment=min_n_treatment,
            min_frac_folds=min_frac_folds,
            print_report=print_report,
            title=f"C3 vs {c_id}（fresh trial）",
        )

    def evaluate_e0_stability(
        self,
        predictions: pd.DataFrame,
        fold_col: str = "fold_id",
        strong_trial_col: str = "is_strong_trial",
        fresh_col: str = "is_fresh_trial_7_28d",
        control: str = "A0",
        min_n_control: int = 50,
        min_n_treatment: int = 15,
        min_frac_folds: float = 2.0 / 3.0,
        print_report: bool = True,
    ) -> dict:
        """E0（同選+strong+fresh）vs control（預設 A0；可傳 C1）。

        預設 min_n_treatment=15（E0 注較少）。
        """
        bets_t = self._bets_rule_e0(
            predictions,
            strong_trial_col=strong_trial_col,
            fresh_col=fresh_col,
        )
        if control == "C1":
            bets_c = self._bets_rule_c1(
                predictions, strong_trial_col=strong_trial_col
            )
            c_id = "C1"
        else:
            bets_c = self._bets_rule_a(predictions)
            c_id = "A0"
        return self.evaluate_fold_stability(
            predictions,
            bets_treatment=bets_t,
            bets_control=bets_c,
            treatment_id="E0",
            control_id=c_id,
            fold_col=fold_col,
            min_n_control=min_n_control,
            min_n_treatment=min_n_treatment,
            min_frac_folds=min_frac_folds,
            print_report=print_report,
            title=f"E0 vs {c_id}（strong∧fresh）",
        )

    def evaluate_all_trial_fold_stability(
        self,
        predictions: pd.DataFrame,
        fold_col: str = "fold_id",
        print_report: bool = True,
    ) -> dict:
        """一次跑 H1(C1)、C3、E0 對 A0 的 fold 穩定性。"""
        return {
            "h1_c1_vs_a0": self.evaluate_h1_stability(
                predictions, fold_col=fold_col, print_report=print_report
            ),
            "c3_vs_a0": self.evaluate_c3_stability(
                predictions, fold_col=fold_col, print_report=print_report
            ),
            "e0_vs_a0": self.evaluate_e0_stability(
                predictions, fold_col=fold_col, print_report=print_report
            ),
            "c3_vs_c1": self.evaluate_c3_stability(
                predictions,
                fold_col=fold_col,
                control="C1",
                print_report=print_report,
            ),
            "e0_vs_c1": self.evaluate_e0_stability(
                predictions,
                fold_col=fold_col,
                control="C1",
                print_report=print_report,
            ),
        }

    # 相容舊 print 名稱
    def _print_h1_report(self, result: dict) -> None:
        self._print_fold_stability_report(
            result,
            title=f"H1 {result.get('treatment_id', 'C')} vs "
            f"{result.get('control_id', 'A')}",
        )

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