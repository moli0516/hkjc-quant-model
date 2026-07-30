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
    - 馬匹: horses (horse_code, import_date, sire)
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
        "import_date",  # 🌟 補齊 SQL 選取的馬匹抵港日期
        "sire",         # 🌟 補齊 SQL 選取的父系/種馬
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

        # 4. 嚴格防範 import_date 未來時間洩漏 (Temporal Leakage Guard)
        df = cls._apply_temporal_guard(df)

        # 5. 生成目標變數 (Targets)
        df = cls._create_targets(df)

        # 6. 保留 Context + Targets
        target_cols = ["target_win", "target_place", "target_rank_score"]
        keep_cols = [
            c for c in cls.CONTEXT_COLS if c in df.columns
        ] + target_cols
        df_base = df[keep_cols].copy()

        # 7. 資料品質檢查
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
                    res.rating,
                    h.import_date,
                    h.sire
                FROM race_results res
                INNER JOIN races r ON res.race_id = r.race_id
                LEFT JOIN horses h ON res.horse_id = h.horse_code
            """
        with sqlite3.connect(db_path) as conn:
            df_raw = pd.read_sql_query(query, conn)

        return cls.build_from_dataframe(df_raw)

    @staticmethod
    def _sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
        """清洗退跑、無效名次與 Null 值。"""
        df["placing_str"] = df["placing"].astype(str).str.upper().str.strip()

        invalid_pos_keywords = [
            "WV", "SCR", "DNF", "DISQ", "FE", "PU", "UR", "NAN", "NONE"
        ]
        valid_mask = ~df["placing_str"].isin(invalid_pos_keywords) & df["placing"].notnull()
        df = df[valid_mask].copy()

        df["craft_rank"] = pd.to_numeric(df["placing"], errors="coerce")
        df = df[df["craft_rank"].notnull() & (df["craft_rank"] > 0)].copy()

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
    def _apply_temporal_guard(df: pd.DataFrame) -> pd.DataFrame:
        """🔒 防洩漏關鍵：若 import_date 晚於比賽日期，將其屏蔽為 NaT，避免未來的抵港紀錄滲透到過去賽事。"""
        if "import_date" in df.columns:
            import_dt = pd.to_datetime(df["import_date"], errors="coerce")
            race_dt = pd.to_datetime(df["race_date"], errors="coerce")
            
            # 若抵港時間在比賽時間之後，判定為時間異常/未到港資訊洩漏，改為 NaT
            future_mask = import_dt > race_dt
            df.loc[future_mask, "import_date"] = pd.NaT
        return df

    @staticmethod
    def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """生成三個標準標籤。"""
        df["target_win"] = (df["craft_rank"] == 1).astype(int)
        df["target_place"] = (df["craft_rank"] <= 3).astype(int)
        df["target_rank_score"] = 1.0 / df["craft_rank"]
        return df