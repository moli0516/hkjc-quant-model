import numpy as np
import pandas as pd
from features.utils import BayesianSmoother, LeakageGuard, SpeedTimeCalculator


class SectionalSpeedGenerator:
    """生成分段時間、末腳爆發力與走位卡位能力特徵 (適用於 SQL 轉置欄位結構 - 防洩漏與崩潰修正版)。"""

    EXECUTION_ORDER = 30

    def __init__(self, key_cols: list[str] = None):
        self.key_cols = key_cols or ["race_id", "horse_id"]

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df[self.key_cols].copy()

        # 1. 計算全場平均速度 (m/s)
        if "distance" in df.columns and "finish_time_sec" in df.columns:
            features["speed_mps_overall"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    df["distance"], df["finish_time_sec"]
                ).astype("float32")
            )

        # 2. 動態提取「最後一段分段時間 (sectional_time_last)」
        sec_cols = [
            c for c in ["sec1_time", "sec2_time", "sec3_time", "sec4_time", "sec5_time", "sec6_time"]
            if c in df.columns
        ]
        
        if sec_cols:
            # 安全防護：使用 ffill 提取最末有效分段時間
            sec_df = df[sec_cols].ffill(axis=1)
            if not sec_df.empty and sec_df.shape[1] > 0:
                features["sectional_time_last"] = sec_df.iloc[:, -1]

        # 3. 計算末腳衝刺速度 (m/s)
        if "sectional_time_last" in features.columns:
            features["speed_mps_last_sectional"] = (
                SpeedTimeCalculator.calc_speed_mps(
                    pd.Series(400.0, index=df.index), features["sectional_time_last"]
                ).astype("float32")
            )

        # 4. 計算走位變化/衝刺追趕能力 (Position Gain)
        pos_cols = [
            c for c in ["pos_sec1", "pos_sec2", "pos_sec3", "pos_sec4", "pos_sec5", "pos_sec6"]
            if c in df.columns
        ]
        
        if len(pos_cols) >= 2:
            pos_df = df[pos_cols]
            first_pos = pos_df.bfill(axis=1).iloc[:, 0]
            last_pos = pos_df.ffill(axis=1).iloc[:, -1]
            features["position_gain_first_to_last"] = (first_pos - last_pos).astype("float32")

        # 5. 計算馬匹歷史近 3 場末腳平均速度 (Rolling Mean - 🔒 防洩漏與索引對齊修復)
        if "speed_mps_last_sectional" in features.columns:
            work_df = df.copy()
            work_df["speed_mps_last_sectional"] = features["speed_mps_last_sectional"]
            
            # 確保按照時間順序排序以進行歷史滾動
            if "date" in work_df.columns:
                work_df = work_df.sort_values(["horse_id", "date"])

            # 呼叫 BayesianSmoother (內部需包含 shift(1) 防止 Leakage)
            rolling_speed = BayesianSmoother.calc_rolling_stat(
                work_df,
                group_cols="horse_id",
                value_col="speed_mps_last_sectional",
                window_size=3,
                stat_type="mean",
            )
            
            # 🔒 關鍵修復：使用 reindex 安全地按原始 df.index 對齊，避免 .loc 找不到索引或轉型失敗
            if isinstance(rolling_speed, pd.Series):
                features["horse_rolling_last_sec_speed_mean_3"] = (
                    rolling_speed.reindex(df.index).fillna(0.0).astype("float32")
                )
            else:
                features["horse_rolling_last_sec_speed_mean_3"] = np.float32(0.0)

        LeakageGuard.validate_feature_dataframe(features, self.key_cols)
        return features