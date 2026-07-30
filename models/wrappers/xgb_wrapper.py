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