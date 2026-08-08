# features/example_generator.py
from typing import Optional
import numpy as np
import pandas as pd
from features.utils import LeakageGuard


class ExampleDataGenerator:

    """生成用於單元測試 (Unit Testing) 與 CI/CD 流程之合成賽馬數據生成器 (Dummy Race Data Generator)。"""

    EXECUTION_ORDER = 0

    def __init__(self, key_cols: Optional[list[str]] = None, seed: int = 42):
        self.key_cols = key_cols or ["race_id", "horse_id"]
        self.seed = seed

    def generate(
        self,
        num_races: int = 50,
        horses_per_race: int = 12,
        start_date: str = "2026-01-01",
    ) -> pd.DataFrame:
        """生成對齊正式 Data Schema 的 Dummy 賽事資料集。

        Parameters
        ----------
        num_races : int
            欲生成的總賽事場數。
        horses_per_race : int
            每場賽事的參賽馬匹數。
        start_date : str
            模擬賽事的起始日期 (YYYY-MM-DD)。

        Returns
        -------
        pd.DataFrame
            包含賽事基礎資訊、賠率、名次及模擬特徵之合成資料集。
        """
        np.random.seed(self.seed)
        records = []
        base_date = pd.Timestamp(start_date)

        for race_idx in range(1, num_races + 1):
            # 模擬賽事日期與 ID
            race_date = base_date + pd.Timedelta(days=race_idx // 2)
            race_id = f"{race_date.strftime('%Y%m%d')}_{race_idx:02d}"

            # 模擬隨機比賽結果 (1 ~ N 名)
            placings = np.random.permutation(
                np.arange(1, horses_per_race + 1)
            )

            # 模擬獨贏賠率 (1.5 ~ 99.0)
            odds = np.round(
                np.random.uniform(1.5, 99.0, size=horses_per_race), 1
            )

            # 模擬負磅與馬匹體重
            actual_weights = np.random.choice(
                np.arange(113, 136), size=horses_per_race
            )
            horse_weights = np.random.randint(
                1000, 1250, size=horses_per_race
            )

            for h_idx in range(horses_per_race):
                records.append(
                    {
                        "race_id": race_id,
                        "date": race_date.strftime("%Y-%m-%d"),
                        "horse_id": f"H{h_idx + 1:03d}",
                        "jockey_id": f"J{np.random.randint(1, 20):02d}",
                        "trainer_id": f"T{np.random.randint(1, 15):02d}",
                        "placing": placings[h_idx],
                        "win_odds": odds[h_idx],
                        "actual_weight": float(actual_weights[h_idx]),
                        "horse_weight": float(horse_weights[h_idx]),
                        # 模擬特徵工程關鍵指標 (對齊系統特徵 Schema)
                        "race_field_speed_zscore": float(
                            np.random.normal(0, 1)
                        ),
                        "horse_rolling_speed_z_mean_5": float(
                            np.random.normal(0, 1)
                        ),
                        "jt_combo_win_rate_smooth": float(
                            np.random.uniform(0.05, 0.35)
                        ),
                        "draw_zscore_in_race": float(np.random.normal(0, 1)),
                    }
                )

        df = pd.DataFrame(records)

        # 1. 確保時間格式與排序符合規範
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["horse_id", "date"]).reset_index(drop=True)

        # 2. 計算標準排名訓練目標 (Relevance Score)
        df["relevance_score"] = (
            df["placing"]
            .apply(lambda p: max(0, 4 - p) if p <= 3 else 0)
            .astype("int32")
        )

        # 3. 執行防洩漏與 Key 欄位校驗
        LeakageGuard.validate_feature_dataframe(df, self.key_cols)

        return df