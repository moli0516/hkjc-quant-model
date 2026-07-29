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