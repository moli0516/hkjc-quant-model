"""
Walk-forward 評估主控。
時間正向多折：訓練 → 預測 → 市場對照 → 下注模擬。
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable, Optional

import pandas as pd

from models.evaluation.baselines import MarketBaseline
from models.evaluation.betting import BetEvaluator
from models.evaluation.metrics_ext import RankingMetrics

logger = logging.getLogger(__name__)


class WalkForwardEvaluator:
    def __init__(
        self,
        feature_cols: list[str],
        model_params: dict,
        min_train_days: int = 730,
        step_days: int = 30,
        date_col: str = "date",
        race_col: str = "race_id",
        label_col: str = "placing",
        odds_col: str = "win_odds",
        overlay_threshold: float = 1.15,
        model_factory: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        feature_cols : 訓練用特徵（不得含賠率）
        model_params : 傳入 model_factory 的超參數
        min_train_days : 最小訓練天數（預設約 24 個月）
        step_days : 每個測試折長度
        date_col : 日期欄位名（'date' 或 'race_date'）
        model_factory : callable(params) -> model
            model 需實作:
              fit(df, feature_cols) 或 fit(X, y, group=...)
              predict(df[feature_cols]) -> scores
            若為 None，會嘗試用 registry 建立 xgb_ranker。
        """
        self.feature_cols = list(feature_cols)
        self.model_params = dict(model_params or {})
        self.min_train_days = min_train_days
        self.step_days = step_days
        self.date_col = date_col
        self.race_col = race_col
        self.label_col = label_col
        self.odds_col = odds_col
        self.model_factory = model_factory

        self.baseline = MarketBaseline(
            race_col=race_col, odds_col=odds_col, rank_col="market_rank"
        )
        self.metrics = RankingMetrics(race_col=race_col, label_col=label_col)
        self.bettor = BetEvaluator(
            overlay_threshold=overlay_threshold,
            race_col=race_col,
            odds_col=odds_col,
            label_col=label_col,
        )

    # ------------------------------------------------------------------
    # Fold 生成
    # ------------------------------------------------------------------
    def generate_folds(self, df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        if self.date_col not in df.columns:
            raise KeyError(f"缺少日期欄位: {self.date_col}")

        dates = pd.to_datetime(df[self.date_col], errors="coerce")
        if dates.isna().all():
            raise ValueError(f"{self.date_col} 無法解析為日期")

        min_date = dates.min().normalize()
        max_date = dates.max().normalize()
        train_end = min_date + timedelta(days=self.min_train_days)

        folds = []
        while train_end < max_date:
            test_start = train_end
            test_end = min(train_end + timedelta(days=self.step_days), max_date)
            if test_start >= test_end:
                break
            folds.append((train_end, test_start, test_end))
            train_end = test_end

        logger.info("Walk-forward folds: %d (min_train=%dd, step=%dd)",
                    len(folds), self.min_train_days, self.step_days)
        return folds

    # ------------------------------------------------------------------
    # 模型建立
    # ------------------------------------------------------------------
    def _build_model(self):
        if self.model_factory is not None:
            return self.model_factory(self.model_params)

        # 後備：嘗試專案 registry
        try:
            from models.registry import ModelRegistry
            reg = ModelRegistry()
            return reg.create("xgb_ranker", self.model_params)
        except Exception as e:
            raise RuntimeError(
                "未提供 model_factory，且無法從 registry 建立 xgb_ranker。 "
                f"原始錯誤: {e}"
            ) from e

    def _fit_model(self, model, train: pd.DataFrame):
        """對齊 XGBRankerWrapper.fit(train_df, feature_cols, target_col, groups, ...)"""
        if "relevance_score" not in train.columns and self.label_col in train.columns:
            train = train.copy()
            train["relevance_score"] = train[self.label_col].apply(
                lambda p: max(0, 4 - p) if pd.notna(p) and p <= 3 else 0
            )

        groups = train.groupby(self.race_col).size().values
        return model.fit(
            train_df=train,
            feature_cols=self.feature_cols,
            target_col="relevance_score",
            groups=groups,
        )

    def _predict_model(self, model, test: pd.DataFrame):
        """對齊 wrapper.predict(df) 或 predict(X)"""
        if hasattr(model, "predict"):
            try:
                return model.predict(test)
            except TypeError:
                return model.predict(test[self.feature_cols])
        raise AttributeError("model 沒有 predict 方法")

    # ------------------------------------------------------------------
    # 單折
    # ------------------------------------------------------------------
    def _train_predict_fold(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> pd.DataFrame:
        model = self._build_model()
        self._fit_model(model, train)

        out = test.copy()
        scores = self._predict_model(model, out)
        out["model_score"] = scores
        out["model_rank"] = (
            out.groupby(self.race_col)["model_score"]
            .rank(ascending=False, method="first")
        )
        return out

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col]).sort_values(
            [self.date_col, self.race_col]
        )

        # 基本欄位檢查
        required = [self.race_col, self.label_col, self.odds_col] + self.feature_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"資料缺少欄位: {missing}")

        # 確保訓練特徵不含賠率
        if self.odds_col in self.feature_cols:
            raise ValueError(f"feature_cols 不應包含賠率欄位 {self.odds_col}")

        folds = self.generate_folds(df)
        chunks: list[pd.DataFrame] = []

        for i, (train_end, test_start, test_end) in enumerate(folds, 1):
            train = df[df[self.date_col] <= train_end]
            test = df[
                (df[self.date_col] > test_start) & (df[self.date_col] <= test_end)
            ]
            if test.empty or train.empty:
                continue

            logger.info(
                "Fold %d/%d | train<=%s (%d) | test (%s, %s] (%d)",
                i, len(folds),
                train_end.date(), len(train),
                test_start.date(), test_end.date(), len(test),
            )

            pred = self._train_predict_fold(train, test)
            pred = self.baseline.transform(pred)
            pred["fold_id"] = i
            pred["train_end"] = train_end
            pred["test_end"] = test_end
            chunks.append(pred)

        if not chunks:
            logger.warning("沒有任何有效 fold，回傳空 DataFrame")
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    def evaluate(self, df: pd.DataFrame) -> dict:
        preds = self.run(df)
        print(preds.columns.tolist())
        if preds.empty:
            return {
                "ranking": {},
                "rule_a": self.bettor.summarize(pd.DataFrame()),
                "rule_b": self.bettor.summarize(pd.DataFrame()),
                "rule_c": self.bettor.summarize(pd.DataFrame()),
                "rule_c_1": self.bettor.summarize(pd.DataFrame()),
                "rule_d": self.bettor.summarize(pd.DataFrame()),
                "predictions": preds,
            }

        ranking = self.metrics.compare(preds)
        bets_a = self.bettor.rule_a(preds)
        bets_b = self.bettor.rule_b(preds)
        bets_c = self.bettor.rule_c_same_pick(preds)
        bets_c_1 = self.bettor.rule_c_compare(preds)
        bets_d = self.bettor.rule_d_model1_market_top2(preds)

        report = {
            "ranking": ranking,
            "rule_a": self.bettor.summarize(bets_a),
            "rule_b": self.bettor.summarize(bets_b),
            "rule_c": self.bettor.summarize(bets_c),
            "rule_c_1": self.bettor.summarize(bets_c_1),
            "rule_d": self.bettor.summarize(bets_d),
            "predictions": preds,
        }

        logger.info(
            "WF 完成 | model_top1=%.4f market_top1=%.4f | "
            "rule_a ROI=%.4f (%d bets) | rule_b ROI=%.4f (%d bets)",
            ranking.get("model_top1", float("nan")),
            ranking.get("market_top1", float("nan")),
            report["rule_a"].get("roi", float("nan")),
            report["rule_a"].get("n_bets", 0),
            report["rule_b"].get("roi", float("nan")),
            report["rule_b"].get("n_bets", 0),
            
        )
        return report