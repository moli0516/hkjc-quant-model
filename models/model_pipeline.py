import logging
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from database.db_manager import DBManager
from models.data_loader import RaceDataLoader
from models.metrics.ranking import RankingMetrics
from models.registry import ModelRegistry


import models.wrappers.xgb_wrapper
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
    ) -> Tuple[Any, Dict[str, float]]:
        """執行完整的訓練與驗證 Pipeline"""
        logger.info("🚀 開始執行訓練 Pipeline...")

        # 1. 載入並自動預處理數據
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
        forbidden_cols = {
            # --- 識別符與時間 ---
            "race_id",
            "horse_id",
            "date",
            "race_date",
            # --- 當場賽後結果 (Target & Race Result Leakage) ---
            "placing",
            "is_win",
            "is_top3",
            "relevance_score",
            "finish_time_sec",
            "margin_len",
            "sectional_time_last",  # 當場末腳時間 (當場結果)
            "position_gain_first_to_last",  # 當場走位變化 (當場結果)
            "speed_mps_overall",  # 當場平均速度 (當場結果)
            "speed_mps_last_sectional",  # 當場末段速度 (當場結果)
            # --- 當場臨場賠率 (Market Leakage) ---
            "win_odds",
            "win_odds_inv",
            "odds_implied_prob",
            "is_market_favorite",
            "win_odds_race_rank",
            "win_odds_race_z",
            "odds_race_zscore",
            "odds_vs_history_win_rate_gap",
            "rating_x_rank_weight",
        }

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

        metrics = {
            "top1_win_rate": top1_win_rate,
            "top3_rate": top3_rate,
            "ndcg@5": ndcg,
        }

        logger.info(f"🎯 驗證結果指標: {metrics}")
        return model, metrics

    def _get_default_search_space(self, model_name: str) -> Callable[[optuna.Trial], Dict[str, Any]]:
        """針對不同模型提供預設的 Optuna 超參數尋優空間 (Search Space)"""
        
        if model_name == "xgb_ranker":
            def xgb_ranker_space(trial: optuna.Trial) -> Dict[str, Any]:
                return {
                    "objective": trial.suggest_categorical("objective", ["rank:pairwise", "rank:ndcg"]),
                    "eval_metric": "ndcg@5",
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 800, 1500, step=100),
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 150),
                    "subsample": trial.suggest_float("subsample", 0.5, 0.8),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 25.0),
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
        model_name: str = "xgb_ranker",
        n_trials: int = 30,
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
        feature_cols: Optional[list] = None,
        custom_param_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
        retrain_best: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        """管線內自動超參數尋優 (Optuna Tuning)

        :param model_name: 模型名稱 ('xgb_ranker', 'lgb_ranker' 等)
        :param n_trials: 搜尋試驗輪數
        :param val_days: 驗證集切分天數
        :param metric_name: 優化目標指標 ('top1_win_rate', 'ndcg@5', 'top3_rate')
        :param direction: 'maximize' 或 'minimize'
        :param feature_cols: (可選) 自訂特徵欄位清單
        :param custom_param_fn: (可選) 自訂 Optuna Search Space 函數
        :param retrain_best: 尋優結束後，是否使用最佳參數自動重新訓練最終模型
        :return: (best_params, best_model_instance)
        """
        from models.hyperopt.optuna_tuner import OptunaTuner
        logger.info(f"🎯 開始執行管線自動尋優: [Model: {model_name}] [Target: {metric_name}] [Trials: {n_trials}]")

        # 1. 取得 Search Space
        param_fn = custom_param_fn or self._get_default_search_space(model_name)

        # 2. 實例化超參數尋優器
        tuner = OptunaTuner(
            pipeline=self,
            model_name=model_name,
            val_days=val_days,
            metric_name=metric_name,
            direction=direction,
        )

        # 3. 執行 Optuna 尋優
        study = tuner.optimize(
            param_fn=param_fn,
            n_trials=n_trials,
            study_name=f"{model_name}_tune",
        )

        best_params = study.best_params
        logger.info(f"🏆 管線尋優完成！最佳指標值 [{metric_name}]: {study.best_value:.4f}")
        logger.info(f"💡 最佳參數組合: {best_params}")

        # 4. 選項：自動以最佳參數重新訓練最終模型
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
            # 1. 解開 Wrapper 取得底層的原生模型 (如 XGBRanker/LGBMRanker)
            raw_model = getattr(model, "model", model)

            # 2. 提取特徵重要性數值
            importances = None
            if hasattr(raw_model, "feature_importances_"):
                importances = raw_model.feature_importances_
            elif hasattr(raw_model, "get_score"):  # 原生 XGBoost Booster 結構
                score_dict = raw_model.get_score(importance_type="gain")
                importances = [score_dict.get(f"f{i}", score_dict.get(col, 0.0)) for i, col in enumerate(feature_cols)]

            if importances is None or len(importances) != len(feature_cols):
                logger.warning("⚠️ 無法讀取該模型的特徵重要性 (Feature Importance)。")
                return

            # 3. 組裝為 DataFrame 排序
            fi_df = (
                pd.DataFrame({"feature": feature_cols, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .reset_index(drop=True)
            )

            # 4. 列印高亮日誌
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

        # 依照賽事 (race_id) 內部對 pred_score 進行排名
        result_df["pred_rank"] = result_df.groupby("race_id")["pred_score"].rank(
            ascending=False, method="min"
        )

        logger.info("✅ 推論完成！")
        return result_df