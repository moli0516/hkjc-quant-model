import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
from features.utils.leak_guard import LeakageGuard


class BaseTargetBuilder:
    """專門負責建立特徵工程的核心基底（Skeleton DataFrame）與目標變數（Targets）。

    與 Schema 對齊：
    - 主表: races (race_id, date, venue, distance, track_type, track_condition, track_texture, race_class)
    - 賽果: race_results (race_id, horse_id, placing, draw, jockey, trainer, actual_weight, declared_weight, win_odds, finish_time_sec)
    """

    PRIMARY_KEYS = ["race_id", "horse_id"]
    TIME_KEYS = ["race_date"]

    # 保留特徵工程所需的最基本 Context 欄位
    CONTEXT_COLS = [
        "race_date",
        "race_id",
        "horse_id",
        "horse_name",
        "jockey",
        "trainer",
        "venue",
        "race_no",
        "race_class",
        "distance",
        "track_condition",
        "track_texture",
        "track_type",
        "draw",
        "actual_weight",
        "declared_weight",
        "win_odds",
        "finish_time_sec",
        "margin_len",
        "rating",
    ]

    @classmethod
    def build_from_dataframe(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """從傳入的 Raw DataFrame 建立骨架。"""
        df = df_raw.copy()

        # 1. 重命名欄位 (對齊系統標準命名)
        if "date" in df.columns and "race_date" not in df.columns:
            df = df.rename(columns={"date": "race_date"})

        # 2. 數據清洗與名次處理
        df = cls._sanitize_data(df)

        # 3. 按時間嚴格排序 (防止 Data Leakage)
        df = cls._sort_by_time(df)

        # 4. 生成目標變數 (Targets)
        df = cls._create_targets(df)

        # 5. 保留 Context + Targets
        target_cols = ["target_win", "target_place", "target_rank_score"]
        keep_cols = [
            c for c in cls.CONTEXT_COLS if c in df.columns
        ] + target_cols
        df_base = df[keep_cols].copy()

        # 6. 資料品質檢查
        LeakageGuard.validate_feature_dataframe(
            df_base, required_keys=cls.PRIMARY_KEYS
        )

        return df_base

    @classmethod
    def build_from_sqlite(
        cls, db_path: str, query: Optional[str] = None
    ) -> pd.DataFrame:
        """根據傳入的 DB Schema 從 SQLite 資料庫做 JOIN 並載入數據。"""
        if query is None:
            query = """
                SELECT 
                    r.date AS race_date,
                    r.race_id,
                    r.venue,
                    r.race_no,
                    r.race_class,
                    r.distance,
                    r.track_condition,
                    r.track_texture,
                    r.track_type,
                    res.horse_id,
                    res.horse_name,
                    res.placing,
                    res.draw,
                    res.jockey,
                    res.trainer,
                    res.actual_weight,
                    res.declared_weight,
                    res.win_odds,
                    res.finish_time_sec,
                    res.margin_len,
                    res.rating
                FROM races r
                INNER JOIN race_results res ON r.race_id = res.race_id
            """
        with sqlite3.connect(db_path) as conn:
            df_raw = pd.read_sql_query(query, conn)

        return cls.build_from_dataframe(df_raw)

    @staticmethod
    def _sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
        """清洗退跑、無效名次與 Null 值。"""
        # 處理名次欄位 (placing)
        df["placing_str"] = df["placing"].astype(str).str.upper().str.strip()

        # 剔除退跑/未完成/異常標籤
        invalid_pos_keywords = [
            "WV",
            "SCR",
            "DNF",
            "DISQ",
            "FE",
            "PU",
            "UR",
            "NAN",
            "NONE",
        ]
        valid_mask = ~df["placing_str"].isin(invalid_pos_keywords) & df[
            "placing"
        ].notnull()
        df = df[valid_mask].copy()

        # 轉為純數值名次
        df["craft_rank"] = pd.to_numeric(df["placing"], errors="coerce")
        df = df[df["craft_rank"].notnull() & (df["craft_rank"] > 0)].copy()

        # 確保 Primary Keys 沒有缺失
        df = df[df["race_id"].notnull() & df["horse_id"].notnull()].copy()
        return df

    @staticmethod
    def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
        """嚴格按 race_date, race_id, horse_id 時間升冪排序。"""
        df["race_date"] = pd.to_datetime(df["race_date"])
        df = df.sort_values(
            by=["race_date", "race_id", "horse_id"], ascending=[True, True, True]
        ).reset_index(drop=True)
        return df

    @staticmethod
    def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """生成三個標準標籤。"""
        # Target 1: 勝出 (1st)
        df["target_win"] = (df["craft_rank"] == 1).astype(int)

        # Target 2: 上名 (Top 3)
        df["target_place"] = (df["craft_rank"] <= 3).astype(int)

        # Target 3: 排序分數 (LTR)
        df["target_rank_score"] = 1.0 / df["craft_rank"]

        return df