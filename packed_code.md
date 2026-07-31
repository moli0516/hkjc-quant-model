# Project Codebase: models

## Directory Structure

```
models/
├── hyperopy
│   ├── __init__.py
│   └── optuna_tuner.py
├── metrics
│   ├── __init__.py
│   └── ranking.py
├── tests
│   ├── test_base_model.py
│   ├── test_data_loader.py
│   ├── test_pipeline.py
│   ├── test_ranking.py
│   ├── test_registry
│   ├── test_time_split.py
│   └── test_xgb_wrapper.py
├── validation
│   ├── __init__.py
│   └── time_split.py
├── wrappers
│   ├── __init__.py
│   └── xgb_wrapper.py
├── __init__.py
├── base_model.py
├── data_loader.py
├── model_pipeline.py
└── registry.py
```

---

## Source Code

### File: `__init__.py`

```py

```

---

### File: `base_model.py`

```py
import abc
import logging
from typing import List, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """
    所有機器學習模型的抽象基底類別 (Abstract Base Class)
    定義統一的介面規範，確保不同的演算法 (如 XGBRanker, LightGBMRanker)
    具有一致的訓練、預測與存檔行為。
    """

    def __init__(self, model_params: dict = None):
        """
        :param model_params: 模型的超參數字典 (Hyperparameters)
        """
        self.model_params = model_params or {}
        self.model = None
        self.feature_cols: List[str] = []

    @abc.abstractmethod
    def fit(
        self, 
        train_df: pd.DataFrame, 
        feature_cols: List[str], 
        target_col: str, 
        groups: np.ndarray = None,
        eval_set: Tuple[pd.DataFrame, List[str], str, np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        模型訓練介面
        
        :param train_df: 訓練集的 DataFrame
        :param feature_cols: 參與訓練的特徵欄位名稱清單
        :param target_col: 目標標籤欄位名稱 (例如 'placing' 或 'is_win')
        :param groups: 排序模型專用的賽事群組陣列 (XGBRanker / LightGBMRanker 必填)
        :param eval_set: 驗證集資料 (val_df, feature_cols, target_col, val_groups)
        """
        pass

    @abc.abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        模型預測介面 (回傳預測分數或機率)
        
        :param df: 包含特徵的 DataFrame
        :return: 1D numpy array 預測結果
        """
        pass

    def save(self, filepath: str) -> None:
        """
        將訓練好的模型序列化並儲存至硬碟
        """
        import joblib
        try:
            joblib.dump(self, filepath)
            logger.info(f"✅ 模型已成功儲存至: {filepath}")
        except Exception as e:
            logger.error(f"❌ 模型儲存失敗 ({filepath}): {e}")
            raise e

    @classmethod
    def load(cls, filepath: str) -> Any:
        """
        從硬碟載入已序列化的模型
        """
        import joblib
        try:
            model_instance = joblib.load(filepath)
            logger.info(f"✅ 模型已成功從 {filepath} 載入")
            return model_instance
        except Exception as e:
            logger.error(f"❌ 模型載入失敗 ({filepath}): {e}")
            raise e
```

---

### File: `data_loader.py`

```py
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from config.settings import settings

# 設定 Logging
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """賽馬訓練數據加載與預處理器 (Data Feed Provider) - 【完全不含賠率版本】

    職責：
    1. 從 DBManager 撈取 feature_matrix 與 race_results 整合數據。
    2. 自動由 placing 衍生二元標籤 (is_win, is_top3)（若資料庫中未包含）。
    3. 根據 config/settings.json 進行類別型態轉換 (astype('category'))。
    4. 嚴格清理並規範特徵型態，防止 XGBoost/LGBM 底層 C++ 引擎因 Category 索引含 float 而崩潰。
    5. 動態分離並提取 Feature、Target、ID 與 Evaluation 欄位，嚴格排除賽後洩漏欄位與所有賠率欄位。
    6. 確保資料按 race_id 排序，並計算 XGBRanker/LGBMRanker 所需的 group 陣列。
    7. 記憶體快取 (In-Memory Caching)：防止多次訓練或 Optuna Tuning 時重複載入與預處理。
    """

    def __init__(self, db_manager=None):
        """:param db_manager: DBManager 實例。若未傳入，將自動初始化新實例。"""
        if db_manager is None:
            from database.db_manager import DBManager

            self.db = DBManager()
        else:
            self.db = db_manager

        # 💡 快取儲存字典: Key 固定為 (include_odds: bool = False)，Value 為 (df, feature_cols, groups)
        self._cache = {}

    def clear_cache(self):
        """🧹 手動清空記憶體快取 (例如在重新執行 Step 5 特徵工程後使用)"""
        self._cache.clear()
        logger.info("🧹 已成功清空 RaceDataLoader 記憶體快取！")

    def get_feature_cols(
        self, df: pd.DataFrame, include_odds: bool = False
    ) -> List[str]:
        """嚴格特徵過濾：

        1. 無條件剔除「當場賽果特徵 (Post-race Data Leakage)」。
        2. 【全面無條件剔除】所有與賠率 (Odds) 及市場指標相關之特徵。
        """
        # 1. 基礎排除清單 (ID, Target, Evaluation)
        exclude_set = set(
            settings.id_cols + settings.target_cols + settings.eval_cols
        )

        # 2. 🚨【無條件絕對剔除】當場賽果數據 (Post-race Data Leakage)
        post_race_leakage_cols = [
            "placing",
            "finish_time_sec",
            "finish_time_race_z",  # 當場完賽時間 Z-Score
            "last_400m_speed_z",  # 當場末腳速度 Z-Score
            "early_pace_expenditure_z",  # 當場早段搶放 Z-Score
            "speed_mps_last_sectional",
            "sectional_time_last",
            "sec1_time",
            "sec2_time",
            "sec3_time",
            "sec4_time",
            "sec5_time",
            "sec6_time",
            "position",
            "plc",
            "margin_len",
        ]
        exclude_set.update(post_race_leakage_cols)

        # 3. 🛡️【完全無條件剔除】所有賠率與市場相關特徵 (Odds & Market Features)
        strict_odds_cols = [
            "win_odds",
            "win_odds_race_z",
            "win_odds_race_rank",
            "odds_implied_prob",
            "is_market_favorite",
            "odds_race_zscore",
            "win_odds_inv",
            "odds_vs_history_win_rate_gap",
            "odds_rank_in_race",
            "implied_prob_share",
        ]
        exclude_set.update(strict_odds_cols)

        # 額外掃描並無條件剔除欄位名稱中包含 'odds' 或 'market' 的動態欄位
        odds_features = [
            col
            for col in df.columns
            if "odds" in col.lower() or "market" in col.lower()
        ]
        exclude_set.update(odds_features)

        feature_cols = [c for c in df.columns if c not in exclude_set]
        return feature_cols

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """對原始 DataFrame 進行標籤衍生、形態轉換與排序等預處理步驟。

        :param df: 原始 merged DataFrame
        :return: 預處理後的 DataFrame
        """
        df = df.copy()

        # 1. 若原始資料僅有 placing，自動衍生二元分類目標標籤 (is_win, is_top3)
        if "placing" in df.columns:
            if "is_win" not in df.columns:
                df["is_win"] = (df["placing"] == 1).astype(int)
            if "is_top3" not in df.columns:
                df["is_top3"] = (df["placing"] <= 3).astype(int)

        # 2. 根據 settings 轉換類別型特徵，防禦 C++ 引擎的 Category float index 崩潰
        cat_cols = set(settings.categorical_cols)
        for col in df.columns:
            if col in cat_cols:
                # 先將 NaN 或混雜格式填補後轉字串，再轉 category，避免 category index 出現 float/NaN
                df[col] = (
                    df[col]
                    .astype(str)
                    .replace({"nan": "missing", "None": "missing", "<NA>": "missing"})
                )
                df[col] = df[col].astype("category")
            elif (
                col not in settings.id_cols
                and col not in settings.eval_cols
                and col not in settings.target_cols
            ):
                # 非 ID/Target/Category 的純數值特徵，統一轉為 float32，避免 float64 或 object 型態殘留
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    "float32"
                )

        # 3. 確保資料嚴格按 race_id 排序 (Ranking 演算法必備)
        if "race_id" in df.columns:
            df = df.sort_values("race_id").reset_index(drop=True)

        return df

    @staticmethod
    def prepare_ranking_groups(df: pd.DataFrame) -> np.ndarray:
        """計算每場賽事 (race_id) 的馬匹數量陣列，供 XGBRanker / LightGBMRanker 使用。

        :param df: 已按 race_id 排序的 DataFrame
        :return: 包含每場賽事出賽馬匹數量的 1D numpy array
        """
        if "race_id" not in df.columns:
            raise KeyError(
                "DataFrame 中找不到 'race_id' 欄位，無法計算 ranking groups！"
            )

        groups = df.groupby("race_id", sort=False).size().to_numpy()
        return groups

    def load_dataset(
        self, include_odds: bool = False, force_reload: bool = False
    ) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
        """[主入口 API] 發起數據載入、標籤衍生、類型轉換與特徵提取流程。

        註：`include_odds` 預設強制為 False，永遠排除賠率數據。

        :param include_odds: 保留參數介面，固定為 False
        :param force_reload: 若為 True，會無視快取並重新從資料庫載入與預處理
        :return: (processed_df, feature_cols, groups)
        """
        cache_key = False  # 強制快取 Key 為不含賠率狀態

        # 💡 1. 檢查記憶體快取：若已有快取且未要求強制重載，直接返回
        if not force_reload and cache_key in self._cache:
            cached_df, cached_features, cached_groups = self._cache[cache_key]
            logger.info(
                "⚡ [Cache Hit] 直接從記憶體載入數據集 (完全排除賠率特徵)！"
            )
            return cached_df.copy(), list(cached_features), cached_groups.copy()

        logger.info("📦 開始從 DBManager 載入特徵矩陣與賽果數據...")
        raw_df = self.db.load_feature_result()

        if raw_df is None or raw_df.empty:
            raise ValueError(
                "【錯誤】資料庫中的 feature_matrix 或 race_results 為空，請先執行特徵工程！"
            )

        logger.info(
            f"📊 原始數據載入完成，共 {len(raw_df)} 條記錄。開始進行無賠率預處理..."
        )

        # 2. 預處理 (自動補齊標籤、類別轉型、排序)
        df = self.process_dataframe(raw_df)

        # 3. 提取特徵欄位清單 (強制剔除賠率特徵)
        feature_cols = self.get_feature_cols(df, include_odds=False)

        # 4. 計算 Ranking Groups
        groups = self.prepare_ranking_groups(df)

        # 💡 5. 寫入記憶體快取
        self._cache[cache_key] = (df, feature_cols, groups)

        logger.info(
            f"✅ DataLoader 處理完畢並已建立快取："
            f"記錄數={len(df)}, 賽事場數={len(groups)}, 純基本面特徵數={len(feature_cols)} (完全排除賠率)"
        )

        return df.copy(), list(feature_cols), groups.copy()
```

---

### File: `model_pipeline.py`

```py
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
```

---

### File: `registry.py`

```py
import logging
from typing import Dict, Type, Any
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    模型工廠與註冊器 (Model Registry / Factory)
    用於動態註冊、管理與創建不同的機器學習模型包裝類別。
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        類別裝飾器：用於將模型類別註冊到工廠中
        
        用法範例:
            @ModelRegistry.register("xgb_ranker")
            class XGBRankerWrapper(BaseModel):
                ...
        """
        def decorator(subclass: Type[BaseModel]):
            if not issubclass(subclass, BaseModel):
                raise TypeError(f"【錯誤】被註冊的類別 '{subclass.__name__}' 必須繼承自 BaseModel！")
            
            if name in cls._registry:
                logger.warning(f"⚠️ 警告: 模型名稱 '{name}' 已存在於註冊表中，將會被覆蓋。")
                
            cls._registry[name] = subclass
            logger.info(f"📌 成功註冊模型: '{name}' -> {subclass.__name__}")
            return subclass
        return decorator

    @classmethod
    def create(cls, name: str, model_params: Any = None) -> BaseModel:
        """
        根據模型名稱動態創建模型實例
        
        :param name: 模型註冊名稱 (如 "xgb_ranker")
        :param model_params: 傳入模型的超參數字典或參數物件
        :return: 對應模型的實例 (BaseModel 的子類別)
        """
        if name not in cls._registry:
            available_models = list(cls._registry.keys())
            raise ValueError(f"【錯誤】找不到名為 '{name}' 的模型！現有可用的模型列表為: {available_models}")
        
        model_cls = cls._registry[name]
        logger.info(f"🔨 正在創建模型實例: '{name}' ({model_cls.__name__})")
        
        if model_params is not None:
            return model_cls(model_params=model_params)
        return model_cls()

    @classmethod
    def list_models(cls) -> list:
        """列出目前所有已註冊的模型名稱"""
        return list(cls._registry.keys())
```

---

### File: `hyperopy\__init__.py`

```py
from models.hyperopt.optuna_tuner import OptunaTuner

__all__ = ["OptunaTuner"]
```

---

### File: `hyperopy\optuna_tuner.py`

```py
import logging
from typing import Callable, Dict, Any, Optional
import optuna

from models.model_pipeline import ModelPipeline

logger = logging.getLogger(__name__)


class OptunaTuner:
    """
    Optuna 自動超參數尋優器 (Hyperparameter Tuner)
    封裝對 ModelPipeline 的調用，防範資料洩漏並集中管理搜尋實驗。
    """

    def __init__(
        self,
        pipeline: ModelPipeline,
        model_name: str = "xgb_ranker",
        val_days: int = 30,
        metric_name: str = "top1_win_rate",
        direction: str = "maximize",
    ):
        """
        :param pipeline: 已初始化的 ModelPipeline 實例
        :param model_name: 在 ModelRegistry 註冊的模型名稱 (例如 'xgb_ranker')
        :param val_days: 驗證集切分天數
        :param metric_name: 評估指標名稱 ('top1_win_rate', 'ndcg@5', 'top3_rate')
        :param direction: 優化方向 ('maximize' 或 'minimize')
        """
        self.pipeline = pipeline
        self.model_name = model_name
        self.val_days = val_days
        self.metric_name = metric_name
        self.direction = direction

    def _create_objective(self, param_fn: Callable[[optuna.Trial], Dict[str, Any]]) -> Callable[[optuna.Trial], float]:
        """建立內部使用的 Objective 函數"""

        def objective(trial: optuna.Trial) -> float:
            # 1. 透過外部傳入的 param_fn 生成該輪 Trial 的超參數組合
            model_params = param_fn(trial)

            try:
                # 2. 調用 Pipeline 進行標準訓練與驗證 (自動處理 TimeSplit)
                _, metrics = self.pipeline.run_train_pipeline(
                    model_name=self.model_name,
                    model_params=model_params,
                    val_days=self.val_days,
                )

                # 3. 提取指定的評估指標
                score = metrics.get(self.metric_name, 0.0)
                return float(score)

            except Exception as e:
                # 防禦機制：若極端參數導致崩潰，給予低分並跳過
                logger.warning(f"⚠️ Trial #{trial.number} 執行異常: {e}")
                return 0.0 if self.direction == "maximize" else 999.0

        return objective

    def optimize(
        self,
        param_fn: Callable[[optuna.Trial], Dict[str, Any]],
        n_trials: int = 30,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> optuna.Study:
        """
        執行自動調參流程
        
        :param param_fn: 接受 trial 並回傳 model_params 字典的函數
        :param n_trials: 試驗輪數
        :param timeout: 最大搜尋時間限制 (秒)
        :param study_name: 實驗名稱
        :return: 完成後的 Optuna Study 物件
        """
        # 隱藏 Optuna 過多的預設資訊
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study_name = study_name or f"{self.model_name}_optimization"
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            pruner=optuna.pruners.MedianPruner(),
        )

        logger.info(
            f"🚀 開始執行 Optuna 自動超參數尋優 (模型: {self.model_name}, 輪數: {n_trials}, 優化指標: {self.metric_name})..."
        )

        objective_fn = self._create_objective(param_fn)
        study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        logger.info(f"🏆 尋優完成！最佳指標 [{self.metric_name}]: {study.best_value:.4f}")
        return study
```

---

### File: `metrics\__init__.py`

```py

```

---

### File: `metrics\ranking.py`

```py
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class RankingMetrics:
    """
    賽馬排序與預測能力評估指標計算器
    """

    @staticmethod
    @staticmethod
    def top_k_win_rate(
        df: pd.DataFrame, 
        pred_score_col: str = "pred_score", 
        target_placing_col: str = "placing", 
        group_col: str = "race_id",
        k: int = 1
    ) -> float:
        if group_col not in df.columns or pred_score_col not in df.columns or target_placing_col not in df.columns:
            raise ValueError(f"【錯誤】缺少必要欄位，請檢查是否包含 '{group_col}', '{pred_score_col}', '{target_placing_col}'")

        hits = 0
        total_races = 0

        for _, group in df.groupby(group_col):
            if len(group) == 0:
                continue
            
            total_races += 1
            # 取預測分數最高的前 K 匹馬
            top_k_preds = group.sort_values(by=pred_score_col, ascending=False).head(k)
            actual_placings = top_k_preds[target_placing_col].values
            
            if k == 1:
                # 獨贏/冠中率：預測第 1 名實際是否為冠軍 (placing == 1)
                if actual_placings[0] == 1:
                    hits += 1
            else:
                # 位置/上名率（精確計算）：計算預測前 K 名中有幾匹馬實際名次 <= k
                # 例如 k=3 時，計算 3 匹預測前三名中有幾匹實際跑進前三名
                hits += np.sum(actual_placings <= k) / k

        win_rate = hits / total_races if total_races > 0 else 0.0
        return float(win_rate)
    @staticmethod
    def mean_ndcg_score(
        df: pd.DataFrame,
        pred_score_col: str = "pred_score",
        target_relevance_col: str = "relevance",
        group_col: str = "race_id",
        k: int = 5
    ) -> float:
        """
        計算跨所有賽事的平均 NDCG@K 分數
        
        :param df: 包含預測分數與相關性標籤的 DataFrame
        :param pred_score_col: 模型預測得分欄位
        :param target_relevance_col: 相關性標籤欄位 (例如冠軍=3, 亞軍=2, 季軍=1, 其餘=0)
        :param group_col: 賽事 ID 欄位
        :param k: 計算 NDCG 的截斷名次 (預設 5)
        :return: 平均 NDCG 分數 (0.0 ~ 1.0)
        """
        if group_col not in df.columns or pred_score_col not in df.columns or target_relevance_col not in df.columns:
            raise ValueError("【錯誤】缺少計算 NDCG 所需的必要欄位！")

        ndcg_scores = []

        for _, group in df.groupby(group_col):
            if len(group) < 2:
                # 若賽事馬匹數量小於 2，無法有效計算排序，略過
                continue

            y_true = group[target_relevance_col].values.reshape(1, -1)
            y_pred = group[pred_score_col].values.reshape(1, -1)

            # 若真實標籤全部為 0（例如沒有馬跑入前三名或資料不全），跳過避免分母為 0
            if np.sum(y_true) == 0:
                continue

            try:
                score = ndcg_score(y_true, y_pred, k=k)
                ndcg_scores.append(score)
            except Exception as e:
                logger.warning(f"⚠️ 計算賽事 {group[group_col].iloc[0]} 的 NDCG 時發生異常: {e}")

        if not ndcg_scores:
            return 0.0

        return float(np.mean(ndcg_scores))
```

---

### File: `tests\test_base_model.py`

```py
import unittest
import tempfile
import pathlib
import numpy as np
import pandas as pd

from models.base_model import BaseModel


# 建立一個測試用的 Dummy 模型繼承 BaseModel
class DummyModel(BaseModel):
    def fit(self, train_df, feature_cols, target_col, groups=None, eval_set=None, **kwargs):
        self.feature_cols = feature_cols
        self.is_fitted = True

    def predict(self, df):
        # 簡單回傳全 1 的預測陣列
        return np.ones(len(df))


class TestBaseModel(unittest.TestCase):

    def test_cannot_instantiate_abstract_base_class(self):
        """驗證抽象基底類別無法直接被實例化"""
        with self.assertRaises(TypeError):
            BaseModel()

    def test_save_and_load_model(self):
        """測試模型的 save 與 load 序列化功能是否正常"""
        model = DummyModel(model_params={"max_depth": 3})
        
        # 模擬訓練
        dummy_df = pd.DataFrame({"feat1": [1, 2], "target": [0, 1]})
        model.fit(dummy_df, ["feat1"], "target")

        # 使用暫存檔案測試存取
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "dummy_model.pkl"
            
            # 存檔
            model.save(str(model_path))
            self.assertTrue(model_path.exists())

            # 載入
            loaded_model = DummyModel.load(str(model_path))
            
            # 驗證狀態
            self.assertIsInstance(loaded_model, DummyModel)
            self.assertEqual(loaded_model.model_params, {"max_depth": 3})
            self.assertTrue(loaded_model.is_fitted)
            
            # 驗證預測功能
            preds = loaded_model.predict(dummy_df)
            np.testing.assert_array_equal(preds, np.array([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_data_loader.py`

```py
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# 匯入要測試的 DataLoader 類別
from models.data_loader import RaceDataLoader


class TestRaceDataLoader(unittest.TestCase):

    def setUp(self):
        """建立符合系統 Schema 的 Mock 原始數據 (模擬 DBManager 回傳的 DataFrame)"""
        self.mock_raw_df = pd.DataFrame(
            {
                "race_id": ["R101", "R101", "R101", "R102", "R102"],
                "horse_id": ["H1", "H2", "H3", "H1", "H2"],
                "horse_name": ["Gold", "Silver", "Bronze", "Gold", "Silver"],
                "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-15", "2024-01-15"],
                "placing": [1, 2, 5, 3, 1],  # 用於衍生 is_win, is_top3
                "win_odds": [2.5, 4.0, 15.0, 5.0, 3.2],
                "draw": [1, 5, 12, 3, 8],
                "jockey": ["Z Purton", "K Teetan", "C Ho", "Z Purton", "K Teetan"],
                "trainer": ["J Size", "KW Lui", "AS Cruz", "J Size", "KW Lui"],
                "brand_prefix": ["A", "B", "C", "A", "B"],
                "course_type": ["TURF", "TURF", "TURF", "ALL WEATHER", "ALL WEATHER"],
                "track_draw_key": ["TURF_1", "TURF_5", "TURF_12", "AW_3", "AW_8"],
                # 模擬一些特徵工程產生的數值特徵
                "age": [4, 5, 6, 4, 5],
                "actual_weight": [133, 125, 118, 130, 120],
                "odds_implied_prob": [0.4, 0.25, 0.06, 0.2, 0.31]
            }
        )

        # 建立 Mock DBManager，使其回傳 mock_raw_df
        self.mock_db = MagicMock()
        self.mock_db.load_feature_result.return_value = self.mock_raw_df

        # 初始化 DataLoader 實例 (注入 Mock DB)
        self.loader = RaceDataLoader(db_manager=self.mock_db)

    def test_process_dataframe_target_derivation(self):
        """測試是否正確自動衍生 is_win 與 is_top3 標籤"""
        processed_df = self.loader.process_dataframe(self.mock_raw_df)

        # 檢查欄位是否存在
        self.assertIn("is_win", processed_df.columns)
        self.assertIn("is_top3", processed_df.columns)

        # 驗證邏輯：placing == 1 則 is_win == 1，否則 0
        self.assertEqual(processed_df.loc[processed_df["race_id"] == "R101", "is_win"].tolist(), [1, 0, 0])
        # 驗證邏輯：placing <= 3 則 is_top3 == 1，否則 0
        self.assertEqual(processed_df.loc[processed_df["race_id"] == "R101", "is_top3"].tolist(), [1, 1, 0])

    def test_process_dataframe_categorical_casting(self):
        """測試指定欄位是否正確轉換為 category 型態"""
        processed_df = self.loader.process_dataframe(self.mock_raw_df)

        from config.settings import settings
        for col in settings.categorical_cols:
            if col in processed_df.columns:
                self.assertEqual(str(processed_df[col].dtype), "category")

    def test_get_feature_cols_with_odds(self):
        """測試保留賠率特徵時，特徵清單是否正確排除 id, target, eval 欄位"""
        df = self.loader.process_dataframe(self.mock_raw_df)
        feature_cols = self.loader.get_feature_cols(df, include_odds=True)

        # 檢查不應該出現在特徵中的欄位
        self.assertNotIn("race_id", feature_cols)
        self.assertNotIn("horse_id", feature_cols)
        self.assertNotIn("placing", feature_cols)
        self.assertNotIn("is_win", feature_cols)
        self.assertNotIn("win_odds", feature_cols)  # 評估欄位

        # 檢查應該要有的特徵 (如 age, actual_weight)
        self.assertIn("age", feature_cols)
        self.assertIn("actual_weight", feature_cols)

    def test_get_feature_cols_without_odds(self):
        """測試剔除賠率特徵 (include_odds=False) 時，是否成功過濾掉賠率相關欄位"""
        df = self.loader.process_dataframe(self.mock_raw_df)
        feature_cols = self.loader.get_feature_cols(df, include_odds=False)

        # 檢查包含 'odds' 字眼的欄位是否被剔除
        for col in feature_cols:
            self.assertNotIn("odds", col.lower())

    def test_prepare_ranking_groups(self):
        """測試 XGBRanker 所需的 group 陣列計算是否正確 (按 race_id 統計筆數)"""
        df = self.loader.process_dataframe(self.mock_raw_df)
        groups = self.loader.prepare_ranking_groups(df)

        # R101 有 3 筆，R102 有 2 筆
        np.testing.assert_array_equal(groups, np.array([3, 2]))

    def test_load_dataset_end_to_end(self):
        """測試整套 load_dataset API 流程"""
        df, feature_cols, groups = self.loader.load_dataset(include_odds=True)

        # 驗證輸出型態與基本維度
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(feature_cols, list)
        self.assertIsInstance(groups, np.ndarray)
        
        self.assertEqual(len(df), 5)
        np.testing.assert_array_equal(groups, np.array([3, 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_pipeline.py`

```py
import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from models.model_pipeline import ModelPipeline
from database.db_manager import DBManager


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        """模擬 DatabaseManager 與 RaceDataLoader 回傳的資料"""
        self.mock_db = MagicMock(spec=DBManager)

        # 建立包含日期與目標欄位的 Mock 訓練數據
        self.mock_df = pd.DataFrame(
            {
                "date": ["2026-05-01", "2026-05-01", "2026-05-15", "2026-05-15", "2026-05-30", "2026-05-30"],
                "race_id": ["R01", "R01", "R02", "R02", "R03", "R03"],
                "horse_id": ["H1", "H2", "H1", "H2", "H1", "H2"],
                "speed_mps_overall": [15.5, 14.2, 16.0, 15.1, 15.8, 14.9],
                "horse_rolling_win_rate_3": [0.3, 0.0, 0.5, 0.1, 0.4, 0.2],
                "placing": [1, 2, 2, 1, 1, 2],
                "relevance_score": [3, 2, 2, 3, 3, 2]
            }
        )

    def test_pipeline_train_and_inference(self):
        """測試 ModelPipeline 的訓練與推論流程是否能順暢串聯"""
        pipeline = ModelPipeline(db_manager=self.mock_db)

        # Mock RaceDataLoader 的 load_dataset 方法，回傳 (df, feature_cols, groups)
        mock_features = ["speed_mps_overall", "horse_rolling_win_rate_3"]
        mock_groups = np.array([2, 2, 2])
        pipeline.data_loader.load_dataset = MagicMock(
            return_value=(self.mock_df, mock_features, mock_groups)
        )

        # 執行訓練 Pipeline (val_days 設小一點以涵蓋 R03 作為驗證集)
        model, metrics = pipeline.run_train_pipeline(
            model_name="xgb_ranker",
            model_params={"n_estimators": 5, "max_depth": 2},
            val_days=10,
            feature_cols=mock_features
        )

        # 驗證是否有產出模型與評估指標
        self.assertIsNotNone(model)
        self.assertIn("top1_win_rate", metrics)
        self.assertIn("ndcg@5", metrics)

        # 執行推論 Pipeline
        inference_input = pd.DataFrame(
            {
                "race_id": ["R04", "R04"],
                "horse_id": ["H1", "H2"],
                "speed_mps_overall": [15.7, 15.0],
                "horse_rolling_win_rate_3": [0.4, 0.1]
            }
        )
        inf_result = pipeline.run_inference_pipeline(model, inference_input)

        # 驗證推論結果是否包含預測分數與排序
        self.assertIn("pred_score", inf_result.columns)
        self.assertIn("pred_rank", inf_result.columns)
        self.assertEqual(len(inf_result), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_ranking.py`

```py
import unittest
import pandas as pd
import numpy as np

from models.metrics.ranking import RankingMetrics


class TestRankingMetrics(unittest.TestCase):

    def setUp(self):
        """建立模擬賽事的預測與實際結果 DataFrame"""
        self.df = pd.DataFrame(
            {
                "race_id": ["R01", "R01", "R01", "R02", "R02", "R02"],
                "horse_id": ["H1", "H2", "H3", "H1", "H2", "H3"],
                "pred_score": [2.5, 1.2, 0.5, 3.1, 1.0, 2.2],  # 模型預測得分
                "placing": [1, 3, 2, 2, 3, 1],                # 實際名次 (1為冠軍)
                "relevance": [3, 1, 2, 1, 0, 3]               # NDCG 相關性分數
            }
        )

    def test_top_1_win_rate(self):
        """測試 Top-1 冠中率計算"""
        # R01 預測最高分是 H1 (pred_score=2.5)，其實際 placing=1 (命中)
        # R02 預測最高分是 H1 (pred_score=3.1)，其實際 placing=2 (未命中)
        # 總共 2 場，命中 1 場，勝率應為 0.5
        win_rate = RankingMetrics.top_k_win_rate(
            self.df, 
            pred_score_col="pred_score", 
            target_placing_col="placing", 
            group_col="race_id", 
            k=1
        )
        self.assertEqual(win_rate, 0.5)

    def test_top_3_rate(self):
        """測試 Top-3 上名率計算"""
        rate = RankingMetrics.top_k_win_rate(
            self.df, 
            pred_score_col="pred_score", 
            target_placing_col="placing", 
            group_col="race_id", 
            k=3
        )
        # 兩場比賽中，預測前三名都有涵蓋實際前三名的馬，命中率應為 1.0
        self.assertEqual(rate, 1.0)

    def test_mean_ndcg_score(self):
        """測試平均 NDCG 分數計算"""
        ndcg = RankingMetrics.mean_ndcg_score(
            self.df,
            pred_score_col="pred_score",
            target_relevance_col="relevance",
            group_col="race_id",
            k=3
        )
        self.assertIsInstance(ndcg, float)
        self.assertGreaterEqual(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_missing_column_raises_error(self):
        """測試缺少必要欄位時是否會拋出 ValueError"""
        with self.assertRaises(ValueError):
            RankingMetrics.top_k_win_rate(self.df, pred_score_col="non_existent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_registry`

```text
import unittest
import numpy as np
import pandas as pd

from models.registry import ModelRegistry
from models.base_model import BaseModel


# 建立一個測試用的 Mock 模型並註冊
@ModelRegistry.register("mock_model")
class MockModel(BaseModel):
    def fit(self, train_df, feature_cols, target_col, **kwargs):
        self.feature_cols = feature_cols

    def predict(self, df):
        return np.zeros(len(df))


class TestModelRegistry(unittest.TestCase):

    def test_register_and_create(self):
        """測試是否能透過註冊名稱成功動態創建模型實例"""
        model = ModelRegistry.create("mock_model", model_params={"param1": 123})
        
        self.assertIsInstance(model, MockModel)
        self.assertEqual(model.model_params, {"param1": 123})
        self.assertIn("mock_model", ModelRegistry.list_models())

    def test_create_non_existent_model_raises_error(self):
        """測試創建不存在的模型時是否會正確拋出 ValueError"""
        with self.assertRaises(ValueError):
            ModelRegistry.create("non_existent_model")

    def test_register_invalid_class_raises_error(self):
        """測試註冊未繼承自 BaseModel 的類別時是否會拋出 TypeError"""
        with self.assertRaises(TypeError):
            @ModelRegistry.register("invalid_model")
            class InvalidModel:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_time_split.py`

```py
import unittest
import pandas as pd
import numpy as np

from models.validation.time_split import TimeSeriesSplitter


class TestTimeSeriesSplitter(unittest.TestCase):

    def setUp(self):
        """建立跨多天的 Mock 賽事資料"""
        self.df = pd.DataFrame(
            {
                "date": [
                    "2026-01-01", "2026-01-01",  # R01 (2匹馬)
                    "2026-01-15", "2026-01-15",  # R02 (2匹馬)
                    "2026-02-01", "2026-02-01", "2026-02-01"  # R03 (3匹馬)
                ],
                "race_id": [
                    "R01", "R01",
                    "R02", "R02",
                    "R03", "R03", "R03"
                ],
                "horse_id": ["H1", "H2", "H1", "H2", "H1", "H2", "H3"],
                "feature1": [1, 2, 3, 4, 5, 6, 7]
            }
        )

    def test_split_by_days(self):
        """測試依天數切分是否正確分配訓練與驗證集，並計算正確的 groups"""
        splitter = TimeSeriesSplitter(date_col="date", group_col="race_id")
        
        # 假設以最後一天 (2026-02-01) 往前推 10 天作為驗證集
        train_df, val_df, train_groups, val_groups = splitter.split_by_days(self.df, val_days=10)

        # 驗證日期切分是否正確 (R01 與 R02 應在訓練集，R03 應在驗證集)
        self.assertTrue(all(train_df["date"] <= "2026-01-15"))
        self.assertTrue(all(val_df["date"] > "2026-01-15"))

        # 驗證 groups 陣列是否正確對應各場次的馬匹數量
        # train 包含 R01 (2匹) 與 R02 (2匹) -> groups 應為 [2, 2]
        np.testing.assert_array_equal(train_groups, np.array([2, 2]))
        
        # val 包含 R03 (3匹) -> groups 應為 [3]
        np.testing.assert_array_equal(val_groups, np.array([3]))

    def test_missing_column_raises_error(self):
        """測試缺少必要欄位時是否會正確拋出 ValueError"""
        splitter = TimeSeriesSplitter(date_col="non_existent_date")
        with self.assertRaises(ValueError):
            splitter.split_by_days(self.df, val_days=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `tests\test_xgb_wrapper.py`

```py
import unittest
import tempfile
import pathlib
import numpy as np
import pandas as pd

from models.wrappers.xgb_wrapper import XGBRankerWrapper


class TestXGBRankerWrapperWithRealSchema(unittest.TestCase):

    def setUp(self):
        """建立完全符合真實資料庫 Schema 的 Mock 訓練與驗證數據集"""
        # 模擬 2 場賽事 (R101 有 3 匹馬, R102 有 2 匹馬)
        self.train_df = pd.DataFrame(
            {
                "race_id": ["R101", "R101", "R101", "R102", "R102"],
                "horse_id": ["H1", "H2", "H3", "H1", "H2"],
                # 使用真實 feature_matrix 的部分特徵
                "speed_mps_overall": [15.5, 15.0, 14.8, 16.1, 15.8],
                "horse_rolling_win_rate_3": [0.33, 0.0, 0.0, 0.5, 0.2],
                "win_odds_race_z": [-1.2, 0.5, 1.8, -0.8, 0.2],
                # 模擬衍生或對應的目標相關性分數 (Relevance: 數字越大代表名次越前面，例如 1st=3, 2nd=2, 其他=0)
                "relevance_score": [3, 2, 0, 3, 1]
            }
        )
        self.feature_cols = [
            "speed_mps_overall",
            "horse_rolling_win_rate_3",
            "win_odds_race_z"
        ]
        self.train_groups = np.array([3, 2])

        # 模擬驗證集
        self.val_df = pd.DataFrame(
            {
                "race_id": ["R103", "R103"],
                "horse_id": ["H1", "H2"],
                "speed_mps_overall": [15.2, 14.9],
                "horse_rolling_win_rate_3": [0.2, 0.1],
                "win_odds_race_z": [-0.5, 0.4],
                "relevance_score": [3, 1]
            }
        )
        self.val_groups = np.array([2])

    def test_init_default_params(self):
        """測試預設超參數是否正確載入"""
        wrapper = XGBRankerWrapper()
        self.assertEqual(wrapper.model_params["objective"], "rank:ndcg")
        self.assertEqual(wrapper.model_params["lambdarank_pair_method"], "topk")
        self.assertEqual(wrapper.model_params["max_depth"], 4)

    def test_fit_and_predict_with_real_schema(self):
        """使用真實特徵欄位測試模型訓練 (含 eval_set 與 early stopping) 以及預測功能"""
        # 將 early_stopping_rounds 透過 model_params 傳入
        wrapper = XGBRankerWrapper(model_params={"n_estimators": 10, "max_depth": 2, "early_stopping_rounds": 5})

        # 執行訓練 (不再於 fit 傳入 early_stopping_rounds)
        wrapper.fit(
            train_df=self.train_df,
            feature_cols=self.feature_cols,
            target_col="relevance_score",
            groups=self.train_groups,
            eval_set=(self.val_df, self.feature_cols, "relevance_score"),
            eval_groups=self.val_groups,
            verbose=False
        )

        self.assertEqual(wrapper.feature_cols, self.feature_cols)
        preds = wrapper.predict(self.val_df)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(len(preds), len(self.val_df))

    def test_fit_missing_groups_raises_error(self):
        """測試未提供訓練 groups 時是否會正確拋出 ValueError"""
        wrapper = XGBRankerWrapper()
        with self.assertRaises(ValueError):
            wrapper.fit(
                train_df=self.train_df,
                feature_cols=self.feature_cols,
                target_col="relevance_score",
                groups=None  # 故意不給
            )

    def test_save_and_load_wrapper(self):
        """測試繼承自 BaseModel 的序列化與反序列化 (save / load)"""
        wrapper = XGBRankerWrapper(model_params={"n_estimators": 5})
        wrapper.fit(
            train_df=self.train_df,
            feature_cols=self.feature_cols,
            target_col="relevance_score",
            groups=self.train_groups
        )

        # 使用暫存資料夾測試存檔與讀取
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = pathlib.Path(tmpdir) / "xgb_ranker.pkl"
            
            # 存檔
            wrapper.save(str(model_path))
            self.assertTrue(model_path.exists())

            # 載入
            loaded_wrapper = XGBRankerWrapper.load(str(model_path))
            
            # 驗證狀態
            self.assertIsInstance(loaded_wrapper, XGBRankerWrapper)
            self.assertEqual(loaded_wrapper.feature_cols, self.feature_cols)
            
            # 驗證載入後的模型能否正常預測
            preds = loaded_wrapper.predict(self.train_df)
            self.assertEqual(len(preds), len(self.train_df))


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### File: `validation\__init__.py`

```py

```

---

### File: `validation\time_split.py`

```py
import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TimeSeriesSplitter:
    """
    時間序列賽事切分工具 (Time Series Splitter)
    專門用來將歷史賽事資料依據日期進行切分，確保不會發生未來資料洩漏 (Data Leakage)。
    """

    def __init__(self, date_col: str = "date", group_col: str = "race_id"):
        """
        :param date_col: 日期欄位名稱
        :param group_col: 賽事群組欄位名稱 (如 race_id)
        """
        self.date_col = date_col
        self.group_col = group_col

    def split_by_days(
        self, 
        df: pd.DataFrame, 
        val_days: int = 30
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        以最後一天往前推指定天數作為驗證集，其餘為訓練集。
        
        :param df: 包含日期與賽事群組的完整 DataFrame
        :param val_days: 驗證集包含最近多少天的賽事
        :return: (train_df, val_df, train_groups, val_groups)
        """
        if self.date_col not in df.columns:
            raise ValueError(f"【錯誤】DataFrame 中找不到指定的日期欄位: '{self.date_col}'")
        if self.group_col not in df.columns:
            raise ValueError(f"【錯誤】DataFrame 中找不到指定的賽事群組欄位: '{self.group_col}'")

        # 確保日期格式為 datetime
        df_sorted = df.copy()
        df_sorted[self.date_col] = pd.to_datetime(df_sorted[self.date_col])

        # 確保資料按照時間與賽事順序排列（Ranker 的硬性要求：同一場比賽的資料必須連續）
        df_sorted = df_sorted.sort_values(by=[self.date_col, self.group_col]).reset_index(drop=True)

        # 計算切分時間點
        max_date = df_sorted[self.date_col].max()
        split_date = max_date - pd.Timedelta(days=val_days)

        logger.info(f"📅 資料集總日期範圍: {df_sorted[self.date_col].min().strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        logger.info(f"✂️ 切分點設定: 驗證集為最近 {val_days} 天 (大於 {split_date.strftime('%Y-%m-%d')})")

        # 劃分 Train 與 Val
        train_df = df_sorted[df_sorted[self.date_col] <= split_date].copy()
        val_df = df_sorted[df_sorted[self.date_col] > split_date].copy()

        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError("【錯誤】切分後的訓練集或驗證集為空！請檢查資料量或 val_days 設定。")

        # 計算 XGBRanker 所需的 groups (每場賽事的馬匹數量)
        # 必須確保 sort=False，且順序與 DataFrame 完全一致
        train_groups = train_df.groupby(self.group_col, sort=False).size().values
        val_groups = val_df.groupby(self.group_col, sort=False).size().values

        logger.info(f"📊 切分完成：訓練集樣本數 {len(train_df)} ({len(train_groups)} 場), 驗證集樣本數 {len(val_df)} ({len(val_groups)} 場)")

        return train_df, val_df, train_groups, val_groups
```

---

### File: `wrappers\__init__.py`

```py

```

---

### File: `wrappers\xgb_wrapper.py`

```py
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRanker

from models.base_model import BaseModel
from models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgb_ranker")
class XGBRankerWrapper(BaseModel):
    """基於 XGBoost Ranker 的賽馬排序模型封裝

    具備完整的例外處理與資料防禦機制。
    """

    def __init__(self, model_params: Optional[dict] = None):
        super().__init__(model_params)

        default_params = {
    "objective": "rank:ndcg",
    "max_depth": 5,
    "learning_rate": 0.02241986575232448,
    "n_estimators": 1100,
    "early_stopping_rounds": 137,
    "subsample": 0.7551375320147171,
    "colsample_bytree": 0.5807601058725049,
    "reg_alpha": 0.07583278924335168,
    "reg_lambda": 20.477793985681824,
    # 💡 建議搭配的通用硬體與重現性設定：
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",  # 若有 GPU 可改為 "hist" 並搭配 device="cuda"
}

        if self.model_params:
            default_params.update(self.model_params)

        # 清理潛在會觸發 C++ 報錯的相容性參數
        default_params.pop("eval_metric", None)
        default_params.pop("lambdarank_pair_method", None)

        self.model_params = default_params
        self.feature_dtypes = {}

        try:
            self.model = XGBRanker(**self.model_params)
        except Exception as e:
            logger.error(f"❌ 初始化 XGBRanker 失敗，請檢查參數設定: {self.model_params}")
            raise RuntimeError(f"XGBRanker 初始化異常: {e}") from e

    def _preprocess_features(
        self, df: pd.DataFrame, feature_cols: List[str], is_training: bool = True
    ) -> pd.DataFrame:
        """資料型態預處理與異常檢查"""
        if df is None or df.empty:
            raise ValueError("【錯誤】輸入的 DataFrame 為空或 None！")

        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"【錯誤】DataFrame 中找不到以下特徵欄位: {missing_cols}")

        X = df[feature_cols].copy()

        try:
            for col in feature_cols:
                if is_training:
                    # 1. 如果是 object 型態，先嘗試轉為數值 (無法轉的設為 NaN 或保留)
                    if X[col].dtype == "object":
                        converted = pd.to_numeric(X[col], errors="coerce")
                        # 如果轉完後全都是 NaN，說明它是真正的字串欄位 (如文字類別)，改轉為 category
                        if converted.isna().all() and not X[col].isna().all():
                            X[col] = X[col].astype("category")
                        else:
                            X[col] = converted

                    # 2. 若原本就是 category，保持 category
                    elif str(X[col].dtype) == "category":
                        X[col] = X[col].astype("category")

                    self.feature_dtypes[col] = X[col].dtype

                else:
                    # 預測階段：對齊訓練時的 dtype
                    target_dtype = self.feature_dtypes.get(col)
                    if target_dtype is not None:
                        if str(target_dtype) == "category":
                            categories = getattr(target_dtype, "categories", None)
                            X[col] = pd.Categorical(X[col], categories=categories)
                        else:
                            X[col] = pd.to_numeric(X[col], errors="coerce").astype(target_dtype)

        except Exception as e:
            logger.error(f"❌ 特徵型態預處理過程發生異常 (is_training={is_training}): {e}")
            raise TypeError(f"特徵預處理失敗: {e}") from e

        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        groups: np.ndarray = None,
        eval_set: Optional[Tuple[pd.DataFrame, List[str], str]] = None,
        eval_groups: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """訓練 XGBRanker 模型（含完整例外處理）"""
        # 1. 基礎輸入參數校驗
        if groups is None or len(groups) == 0:
            raise ValueError("【錯誤】XGBRanker 訓練必須提供非空的 groups 陣列（每場賽事的馬匹數量）！")

        if target_col not in train_df.columns:
            raise KeyError(f"【錯誤】訓練資料集中不存在目標標籤欄位 '{target_col}'！")

        if sum(groups) != len(train_df):
            raise ValueError(
                f"【錯誤】groups 的總和 ({sum(groups)}) 與訓練樣本總數 ({len(train_df)}) 不一致！"
            )

        self.feature_cols = feature_cols

        # 2. 特徵預處理
        try:
            X_train = self._preprocess_features(train_df, feature_cols, is_training=True)
            y_train = train_df[target_col]
        except Exception as e:
            logger.error(f"❌ 訓練集資料準備失敗: {e}")
            raise

        fit_params = {"group": groups}

        # 3. 驗證集與 Early Stopping 檢測
        if eval_set is not None:
            if eval_groups is None or len(eval_groups) == 0:
                raise ValueError("【錯誤】提供了 eval_set 時，必須同時提供非空的 eval_groups！")

            try:
                val_df, val_feature_cols, val_target_col = eval_set
                
                if sum(eval_groups) != len(val_df):
                    raise ValueError(
                        f"【錯誤】eval_groups 的總和 ({sum(eval_groups)}) 與驗證集樣本數 ({len(val_df)}) 不一致！"
                    )

                X_val = self._preprocess_features(val_df, val_feature_cols, is_training=False)
                y_val = val_df[val_target_col]

                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["eval_group"] = [eval_groups]

                if "early_stopping_rounds" not in self.model.get_params():
                    self.model.set_params(early_stopping_rounds=50)

            except Exception as e:
                logger.error(f"❌ 驗證集 (eval_set) 處理失敗: {e}")
                raise

            if "verbose" not in kwargs:
                kwargs["verbose"] = False
        else:
            if "early_stopping_rounds" in self.model.get_params():
                self.model.set_params(early_stopping_rounds=None)

        # 4. 執行 fit 並捕獲 XGBoost 底層 C++ / Runtime 異常
        kwargs.pop("eval_metric", None)  # 確保不透傳引發衝突的 metric

        logger.info(
            f"🚀 開始訓練 XGBRanker 模型，特徵數: {len(feature_cols)}, 訓練樣本數: {len(X_train)}"
        )

        try:
            self.model.fit(X_train, y_train, **fit_params, **kwargs)
            logger.info("✅ XGBRanker 模型訓練成功！")

        except xgb.core.XGBoostError as e:
            logger.error(f"❌ XGBoost 底層 C++ 引擎拋出錯誤: {e}")
            raise RuntimeError(f"XGBoost 訓練引擎崩潰: {e}") from e

        except MemoryError as e:
            logger.error("❌ 訓練過程記憶體溢出 (Out of Memory)！請嘗試減少 n_estimators 或 max_depth。")
            raise MemoryError("模型訓練記憶體不足") from e

        except Exception as e:
            logger.error(f"❌ 訓練過程中發生未預期的錯誤: {e}")
            raise RuntimeError(f"模型 fit 失敗: {e}") from e

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """預測賽事中各馬匹的排序得分（含例外處理）"""
        if self.model is None:
            raise RuntimeError("【錯誤】模型尚未訓練或載入，無法進行預測！")

        try:
            X = self._preprocess_features(df, self.feature_cols, is_training=False)
            scores = self.model.predict(X)

            if len(scores) != len(df):
                raise ValueError(f"【錯誤】預測結果數量 ({len(scores)}) 與輸入資料筆數 ({len(df)}) 不符！")

            return scores

        except KeyError as e:
            logger.error(f"❌ 推論失敗，缺失必要特徵欄位: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 模型推論 (predict) 過程發生異常: {e}")
            raise RuntimeError(f"模型預測失敗: {e}") from e
```

---

