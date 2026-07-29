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