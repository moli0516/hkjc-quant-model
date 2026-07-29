import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

from config.settings import settings

# 設定 Logging
logger = logging.getLogger(__name__)


class RaceDataLoader:
    """
    賽馬訓練數據加載與預處理器 (Data Feed Provider)
    職責：
    1. 從 DBManager 撈取 feature_matrix 與 race_results 整合數據。
    2. 自動由 placing 衍生二元標籤 (is_win, is_top3)（若資料庫中未包含）。
    3. 根據 config/settings.json 進行類別型態轉換 (astype('category'))。
    4. 動態分離並提取 Feature、Target、ID 與 Evaluation 欄位。
    5. 確保資料按 race_id 排序，並計算 XGBRanker/LGBMRanker 所需的 group 陣列。
    """

    def __init__(self, db_manager=None):
        """
        :param db_manager: DBManager 實例。若未傳入，將自動初始化新實例。
        """
        if db_manager is None:
            from database.db_manager import DBManager
            self.db = DBManager()
        else:
            self.db = db_manager

    def get_feature_cols(self, df: pd.DataFrame, include_odds: bool = True) -> List[str]:
        """
        根據 settings 配置與參數，動態過濾並提取用於訓練的特徵欄位清單。
        
        :param df: 包含所有欄位的 DataFrame
        :param include_odds: 是否包含賠率衍生特徵
        :return: 純訓練特徵欄位名稱清單
        """
        exclude_set = set(settings.id_cols + settings.target_cols + settings.eval_cols)
        
        # 若指定剔除賠率特徵 (訓練純基本面模型)
        if not include_odds:
            odds_features = [col for col in df.columns if 'odds' in col.lower()]
            exclude_set.update(odds_features)
            
        feature_cols = [c for c in df.columns if c not in exclude_set]
        return feature_cols

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        對原始 DataFrame 進行標籤衍生、形態轉換與排序等預處理步驟。
        
        :param df: 原始 merged DataFrame
        :return: 預處理後的 DataFrame
        """
        df = df.copy()

        # 1. 若原始資料僅有 placing，自動衍生二元分類目標標籤 (is_win, is_top3)
        if 'placing' in df.columns:
            if 'is_win' not in df.columns:
                df['is_win'] = (df['placing'] == 1).astype(int)
            if 'is_top3' not in df.columns:
                df['is_top3'] = (df['placing'] <= 3).astype(int)

        # 2. 根據 settings 轉換類別型特徵 (Categorical Types)
        for col in settings.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # 3. 確保資料嚴格按 race_id 排序 (Ranking 演算法必備)
        if 'race_id' in df.columns:
            df = df.sort_values('race_id').reset_index(drop=True)
            
        return df

    @staticmethod
    def prepare_ranking_groups(df: pd.DataFrame) -> np.ndarray:
        """
        計算每場賽事 (race_id) 的馬匹數量陣列，供 XGBRanker / LightGBMRanker 使用。
        
        :param df: 已按 race_id 排序的 DataFrame
        :return: 包含每場賽事出賽馬匹數量的 1D numpy array
        """
        if 'race_id' not in df.columns:
            raise KeyError("DataFrame 中找不到 'race_id' 欄位，無法計算 ranking groups！")
            
        groups = df.groupby('race_id', sort=False).size().to_numpy()
        return groups

    def load_dataset(self, include_odds: bool = True) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
        """
        [主入口 API] 發起數據載入、標籤衍生、類型轉換與特徵提取流程。
        
        :param include_odds: 是否包含賠率相關特徵
        :return: (processed_df, feature_cols, groups)
        """
        logger.info("📦 開始從 DBManager 載入特徵矩陣與賽果數據...")
        raw_df = self.db.load_feature_result()
        
        if raw_df is None or raw_df.empty:
            raise ValueError("【錯誤】資料庫中的 feature_matrix 或 race_results 為空，請先執行特徵工程！")

        logger.info(f"📊 原始數據載入完成，共 {len(raw_df)} 條記錄。開始進行數據預處理...")
        
        # 1. 預處理 (自動補齊標籤、類別轉型、排序)
        df = self.process_dataframe(raw_df)

        # 2. 提取特徵欄位清單
        feature_cols = self.get_feature_cols(df, include_odds=include_odds)

        # 3. 計算 Ranking Groups
        groups = self.prepare_ranking_groups(df)

        logger.info(
            f"✅ DataLoader 處理完畢："
            f"記錄數={len(df)}, 賽事場數={len(groups)}, 特徵數={len(feature_cols)} (含賠率={include_odds})"
        )

        return df, feature_cols, groups