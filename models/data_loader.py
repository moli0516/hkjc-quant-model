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