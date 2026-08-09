import logging
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from database.db_manager import DBManager
from models.data_loader import RaceDataLoader
from models.metrics.ranking import RankingMetrics
from models.metrics.finance import FinanceMetrics  # 👈 [新增] 匯入財務指標模組
from models.registry import ModelRegistry

import models.wrappers.xgb_wrapper
from config.settings import settings
from models.validation.time_split import TimeSeriesSplitter

logger = logging.getLogger(__name__)


class ModelPipeline:
    """賽馬機器學習統籌工作流 (Model Pipeline)

    負責將資料載入、時間切分、模型訓練、超參數尋優、評估與推論串聯成標準化流程。
    """

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self.data_loader = RaceDataLoader(db_manager)
        self.splitter = TimeSeriesSplitter(date_col="date", group_col="race_id")

    def run_train_pipeline(
        self,
        model_name: str = "xgb_ranker",
        model_params: Optional[Dict[str, Any]] = None,
        val_days: int = 30,
        feature_cols: Optional[list] = None,
        use_kelly: bool = True,
        kelly_fraction: float = 0.5,
        initial_bankroll: float = 1000.0,
        max_stake_pct: float = 0.05,
    ) -> Tuple[Any, Dict[str, float]]:
        """執行完整的訓練與驗證 Pipeline（支援平注法或凱利公式動態注碼）"""
        logger.info("🚀 開始執行訓練 Pipeline...")

        # 1. 載入並自動預處理數據 (保留 include_odds=True 以確保驗證集有賠率進行財務回測)
        df, default_feature_cols, _ = self.data_loader.load_dataset(
            include_odds=True
        )

        if df.empty:
            raise ValueError("【錯誤】訓練資料集為空，無法進行訓練！")

        # 解析日期：支援 YYYY/MM/DD 或 YYYY-MM-DD
        if "date" not in df.columns:
            if "race_date" in df.columns:
                df["date"] = df["race_date"]
            else:
                df["date"] = df["race_id"].astype(str).str.extract(
                    r"(\d{4}[/-]\d{2}[/-]\d{2})"
                )[0]

        # 轉為標準 datetime 格式
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 清理與驗證
        cleaned_len = len(df.dropna(subset=["date"]))
        df = df.dropna(subset=["date"]).copy()

        if cleaned_len == 0:
            raise ValueError(
                "【錯誤】無法從 race_id 解析出任何有效日期！"
            )

        logger.info(
            f"💡 成功解析 {cleaned_len} 筆賽事日期 (日期範圍: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')})"
        )

        # 若使用者未自訂 feature_cols，則使用 DataLoader 自動提取的預設特徵
        if feature_cols is None:
            feature_cols = list(default_feature_cols)

        # 定義禁止傳入模型的「未來官子/當場結果/當場賠率」欄位 (防 Leakage)
        forbidden_cols = settings.banned_features

        # 雙重防洩漏保險：強制過濾禁用的欄位
        feature_cols = [
            col for col in feature_cols if col not in forbidden_cols
        ]

        if not feature_cols:
            raise ValueError(
                "【錯誤】經過禁用的欄位過濾後，有效特徵數為 0，無法進行訓練！"
            )

        # 確保訓練資料中包含模型所需的 relevance_score 標籤
        if "relevance_score" not in df.columns and "placing" in df.columns:
            df["relevance_score"] = df["placing"].apply(
                lambda p: max(0, 4 - p) if p <= 3 else 0
            )

        logger.info(
            f"📊 總樣本數: {len(df)}, 有效特徵數: {len(feature_cols)}"
        )

        # =========================================================================
        # 🚨 訓練前強制執行 Micro-Tracing 檢驗防範數據洩漏
        # =========================================================================
        logger.info("🔍 正在執行訓練前 Micro-Tracing 數據洩漏檢驗...")
        try:
            rolling_features_to_check = [c for c in feature_cols if any(kw in c.lower() for kw in ["rolling", "smooth", "rate"])]
            
            if rolling_features_to_check:
                _, is_passed = self.run_micro_tracing(
                    df=df,
                    rolling_cols=rolling_features_to_check,
                    min_races=5,
                    max_races=10
                )
                if not is_passed:
                    raise RuntimeError("【嚴重錯誤】Micro-Tracing 檢驗失敗！偵測到當場數據洩漏 (Data Leakage)，請檢查特徵工程中的 shift() 邏輯。訓練已強制中斷。")
                logger.info("✅ Micro-Tracing 檢驗通過，未發現當場數據洩漏！")
            else:
                logger.info("⚠️ 特徵清單中未偵測到需要檢驗的滾動特徵，跳過 Micro-Tracing。")
        except Exception as e:
            logger.error(f"Micro-Tracing 檢驗過程發生例外狀況: {e}")
            raise

        # 2. 時間序列切分 (防止資料洩漏)
        train_df, val_df, train_groups, val_groups = (
            self.splitter.split_by_days(df, val_days=val_days)
        )

        # 3. 創建模型實例
        model = ModelRegistry.create(
            name=model_name, model_params=model_params
        )

        # 4. 執行模型訓練
        model.fit(
            train_df=train_df,
            feature_cols=feature_cols,
            target_col="relevance_score",
            groups=train_groups,
            eval_set=(val_df, feature_cols, "relevance_score"),
            eval_groups=val_groups,
        )

        # =========================================================================
        # 📊 特徵重要性 (Feature Importance) 提取與日誌輸出
        # =========================================================================
        self._log_feature_importance(model, feature_cols)

        # 5. 模型預測與評估
        logger.info("📈 正在計算驗證集評估指標...")
        val_preds = model.predict(val_df)
        val_df_evaluated = val_df.copy()
        val_df_evaluated["pred_score"] = val_preds

        # 計算原本的排名指標
        top1_win_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_placing_col="placing",
            group_col="race_id",
            k=1,
        )
        top3_rate = RankingMetrics.top_k_win_rate(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_placing_col="placing",
            group_col="race_id",
            k=3,
        )
        ndcg = RankingMetrics.mean_ndcg_score(
            val_df_evaluated,
            pred_score_col="pred_score",
            target_relevance_col="relevance_score",
            group_col="race_id",
            k=5,
        )

        # =========================================================================
        # 💰 [更新] 計算財務指標 (支援平注法或凱利公式動態注碼)
        # =========================================================================
            
        finance_metrics = {}
        if "win_odds" in val_df_evaluated.columns and "placing" in val_df_evaluated.columns:
            fin_results = FinanceMetrics.calculate_betting_performance(
                df=val_df_evaluated,
                pred_score_col="pred_score",
                odds_col="win_odds",
                target_placing_col="placing",
                group_col="race_id",
                stake=10.0,
                kelly_fraction=1/6,
                max_stake_pct=0.03,
                temperature=2.5,
                min_ev=1.12,
                use_kelly=True
            )

            finance_metrics = {
                "roi": fin_results["roi"],
                "net_profit": fin_results["net_profit"],
                "total_bets": fin_results["total_bets"],
                "final_bankroll": fin_results["final_bankroll"],
            }
            # 如果啟用了凱利公式，也可以順便記錄最終資金池結餘
            if use_kelly and "final_bankroll" in fin_results:
                finance_metrics["final_bankroll"] = fin_results["final_bankroll"]
        else:
            logger.warning("⚠️ 驗證集中缺少 'win_odds' 或 'placing' 欄位，跳過財務回測。")

        # 合併排序指標與財務指標
        metrics = {
            "top1_win_rate": top1_win_rate,
            "top3_rate": top3_rate,
            "ndcg@5": ndcg,
            **finance_metrics
        }

        logger.info(f"🎯 驗證結果指標總結: {metrics}")
        return model, metrics

    def _get_default_search_space(self, model_name: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
        """針對不同模型提供預設的 Optuna 超參數尋優空間 (Search Space)
        
        修正重點：
        1. 強制移除非首馬導向的 'rank:pairwise'，統一使用 'rank:ndcg'
        2. 將 eval_metric 綁定為 ndcg@3 / ndcg@1，對齊賽馬頭馬預測
        """
        if model_name == "xgb_ranker":
            def xgb_ranker_space(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    # 🔒 強制鎖定 rank:ndcg，避免 pairwise 浪費容量在末段馬匹排序
                    "objective": "rank:ndcg",
                    "eval_metric": trial.suggest_categorical("eval_metric", ["ndcg@1", "ndcg@3", "ndcg@5"]),
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=100),
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 30, 100),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0),
                    "random_state": 42,
                    "tree_method": "hist",
                    "enable_categorical": True,
                }
            return xgb_ranker_space

        elif model_name == "lgb_ranker":
            def lgb_ranker_space(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    "objective": "lambdarank",
                    "metric": "ndcg",
                    "eval_at": [1, 3],
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "n_estimators": trial.suggest_int("n_estimators", 500, 1500, step=100),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    "random_state": 42,
                }
            return lgb_ranker_space

        else:
            raise ValueError(f"【錯誤】未定義該模型的超參數搜尋空間: {model_name}")

    def run_tune_pipeline(
        self,
        metric_name: str,
        model_name: str = "xgb_ranker",
        n_trials: int = 30,
        val_days: int = 30,
        
        direction: str = "maximize",
        feature_cols: Optional[list] = None,
        custom_param_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
        retrain_best: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """管線內自動超參數尋優 (Optuna Tuning)"""
        from models.hyperopt.optuna_tuner import OptunaTuner
        logger.info(f"🎯 開始執行管線自動尋優: [Model: {model_name}] [Target: {metric_name}] [Trials: {n_trials}]")

        param_fn = custom_param_fn or self._get_default_search_space(model_name)

        tuner = OptunaTuner(
            pipeline=self,
            model_name=model_name,
            val_days=val_days,
            metric_name=metric_name,
            direction=direction,
        )

        study = tuner.optimize(
            param_fn=param_fn,
            n_trials=n_trials,
            study_name=f"{model_name}_tune",
        )

        best_params = study.best_params
        logger.info(f"🏆 管線尋優完成！最佳指標值 [{metric_name}]: {study.best_value:.4f}")
        logger.info(f"💡 最佳參數組合: {best_params}")

        best_model = None
        if retrain_best:
            logger.info("🚀 正在使用最佳超參數重新訓練最終模型...")
            best_model, final_metrics = self.run_train_pipeline(
                model_name=model_name,
                model_params=best_params,
                val_days=val_days,
                feature_cols=feature_cols,
            )
            logger.info(f"✅ 最終模型重新訓練完畢，驗證集指標: {final_metrics}")

        return best_params, best_model

    def _log_feature_importance(self, model: Any, feature_cols: list, top_n: int = 20):
        """解析內部原生的模型物件並列印特徵重要性 (相容 Wrapper)"""
        try:
            raw_model = getattr(model, "model", model)

            importances = None
            if hasattr(raw_model, "feature_importances_"):
                importances = raw_model.feature_importances_
            elif hasattr(raw_model, "get_score"): 
                score_dict = raw_model.get_score(importance_type="gain")
                importances = [score_dict.get(f"f{i}", score_dict.get(col, 0.0)) for i, col in enumerate(feature_cols)]

            if importances is None or len(importances) != len(feature_cols):
                logger.warning("⚠️ 無法讀取該模型的特徵重要性 (Feature Importance)。")
                return

            fi_df = (
                pd.DataFrame({"feature": feature_cols, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )

            print("\n" + "=" * 60)
            print(f"🔥 [模型特徵權重排行榜]")
            print("=" * 60)
            for idx, row in fi_df.iterrows():
                print(f"  #{idx+1:02d} | {row['feature']:<35} | 權重: {row['importance']:.6f}")
            print("=" * 60 + "\n")

        except Exception as e:
            logger.warning(f"⚠️ 提取 Feature Importance 過程發生異常: {e}")

    def run_inference_pipeline(
        self, model: Any, inference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """執行推論 Pipeline：對給定的最新賽事特徵進行預測評分"""
        logger.info("🔮 開始執行推論 (Inference) Pipeline...")

        preds = model.predict(inference_df)
        result_df = inference_df.copy()
        result_df["pred_score"] = preds

        result_df["pred_rank"] = result_df.groupby("race_id")["pred_score"].rank(
            ascending=False, method="min"
        )

        logger.info("✅ 推論完成！")
        return result_df

    def run_micro_tracing(
        self,
        df: Optional[pd.DataFrame] = None,
        horse_id: Optional[str] = None,
        rolling_cols: Optional[list] = None,
        min_races: int = 5,
        max_races: int = 10,
    ) -> Tuple[pd.DataFrame, bool]:
        """對數據集執行單匹馬「微觀逐行印出」(Micro-Tracing) 檢驗，防止時序資料洩漏"""
        from models.validation.micro_tracing import HorseMicroTracer

        if df is None:
            df, _, _ = self.data_loader.load_dataset(include_odds=False)

        tracer = HorseMicroTracer(
            date_col="date",
            horse_id_col="horse_id",
            target_col="placing",
            race_id_col="race_id",
        )

        trace_df, is_passed, _ = tracer.trace_horse(
            df=df,
            horse_id=horse_id,
            rolling_cols=rolling_cols,
            min_races=min_races,
            max_races=max_races,
            verbose=True,
        )
        return trace_df, is_passed

    def run_walk_forward_evaluation(
        self,
        model_name: str = "xgb_ranker",
        model_params: Optional[Dict[str, Any]] = None,
        min_train_days: int = 730,
        step_days: int = 30,
        overlay_threshold: float = 1.15,
        feature_cols: Optional[list] = None,
        run_diagnosis: bool = True,
        diagnosis_stake: float = 1.0,
    ) -> dict:
        """Walk-forward 評估：多段時間正向重訓 + 模型 vs 市場 + 簡單下注 ROI。

        資料需含 date / win_odds / placing；訓練特徵會排除 banned_features 與賠率。
        run_diagnosis=True 時，對 predictions 再跑大熱 baseline / 賠率畫像 / 分層 / 試閘 residual。
        """
        from models.evaluation.walk_forward import WalkForwardEvaluator

        logger.info(
            "📈 開始 Walk-forward 評估 | model=%s | min_train_days=%d | step_days=%d | overlay=%.2f",
            model_name,
            min_train_days,
            step_days,
            overlay_threshold,
        )

        # 1. 載入評估用資料（必須含賠率，才能做市場 baseline 與 ROI）
        df, default_feature_cols, _ = self.data_loader.load_dataset(
            include_odds=True
        )

        if df.empty:
            raise ValueError("【錯誤】Walk-forward 資料集為空。")

        # 2. 統一日期欄位
        if "date" not in df.columns:
            if "race_date" in df.columns:
                df["date"] = df["race_date"]
            else:
                df["date"] = (
                    df["race_id"]
                    .astype(str)
                    .str.extract(r"(\d{4}[/-]\d{2}[/-]\d{2})")[0]
                )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        if df.empty:
            raise ValueError("【錯誤】無法解析任何有效 date，Walk-forward 中止。")

        # 3. 特徵欄位（排除禁用欄位與當場賠率）
        if feature_cols is None:
            feature_cols = list(default_feature_cols)

        forbidden_cols = set(getattr(settings, "banned_features", []) or [])
        forbidden_cols.update({"win_odds", "odds", "placing", "relevance_score"})

        feature_cols = [
            c for c in feature_cols if c not in forbidden_cols and c in df.columns
        ]
        if not feature_cols:
            raise ValueError("【錯誤】Walk-forward 有效特徵數為 0。")
        logger.info("Walk-forward 有效特徵數: %d", len(feature_cols))

        # 4. 標籤
        if "relevance_score" not in df.columns and "placing" in df.columns:
            df["relevance_score"] = df["placing"].apply(
                lambda p: max(0, 4 - p) if pd.notna(p) and p <= 3 else 0
            )

        # 5. 模型參數
        params = dict(model_params or {})

        # 6. model_factory
        def _factory(p: Dict[str, Any]):
            return ModelRegistry.create(name=model_name, model_params=p)

        evaluator = WalkForwardEvaluator(
            feature_cols=feature_cols,
            model_params=params,
            min_train_days=min_train_days,
            step_days=step_days,
            date_col="date",
            race_col="race_id",
            label_col="placing",
            odds_col="win_odds",
            overlay_threshold=overlay_threshold,
            model_factory=_factory,
        )

        report = evaluator.evaluate(df)
        logger.info("✅ Walk-forward 評估完成。")

        # 7. 診斷 1–4（可關）
        if run_diagnosis:
            preds = report.get("predictions") if isinstance(report, dict) else None
            if preds is not None and isinstance(preds, pd.DataFrame) and not preds.empty:
                report["diagnosis"] = self.run_walk_forward_diagnosis(
                    predictions=preds,
                    stake=diagnosis_stake,
                    print_report=True,
                )
            else:
                logger.warning("⚠️ 無 predictions，跳過 Walk-forward 診斷。")
                report["diagnosis"] = None

        return report

    def run_walk_forward_diagnosis(
        self,
        predictions: pd.DataFrame,
        stake: float = 1.0,
        print_report: bool = True,
    ) -> dict:
        """對 Walk-forward predictions 執行診斷（大熱 / 賠率畫像 / 分層 / 試閘 residual）。"""
        from models.evaluation.diagnostics import WalkForwardDiagnostics

        logger.info("🔍 開始 Walk-forward 診斷...")
        diag = WalkForwardDiagnostics(stake=stake)
        return diag.run(predictions, print_report=print_report)