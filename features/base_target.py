# features/builder/base_target.py

from typing import Optional
import numpy as np
import pandas as pd
from database.db_manager import DBManager
from features.utils.leak_guard import LeakageGuard


class BaseTargetBuilder:
    """專門負責建立特徵工程的核心基底（Skeleton DataFrame）與目標變數（Targets）。

    數據來源透過 DBManager 統一對齊：
    - 主表: races (race_id, date, venue, distance, track_type, track_condition, track_texture, race_class)
    - 賽果: race_results (race_id, horse_id, placing, draw, jockey, trainer, actual_weight, declared_weight, win_odds, finish_time_sec)
    - 馬匹: horses (horse_code, import_date, sire)
    - 分段: race_sectionals (sectional_time_sec, position)
    """

    PRIMARY_KEYS = ["race_id", "horse_id"]
    TIME_KEYS = ["race_date"]

    # 保留特徵工程所需的最基本 Context 欄位 (補齊 DBManager 提供的分段特徵)
    CONTEXT_COLS = [
        "race_date",
        "date",  # 🌟 補齊相容欄位，確保 downstream 模組調用不碰撞
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
        "import_date",
        "sire",
        # 🌟 補齊 DBManager 轉置後的分段數據
        "sec1_time",
        "sec2_time",
        "sec3_time",
        "sec4_time",
        "sec5_time",
        "sec6_time",
        "pos_sec1",
        "pos_sec2",
        "pos_sec3",
        "pos_sec4",
        "pos_sec5",
        "pos_sec6",
    ]

    @classmethod
    def build_from_dataframe(cls, df_raw: pd.DataFrame) -> pd.DataFrame:
        """從傳入的 Raw DataFrame 建立骨架。"""
        df = df_raw.copy()

        # 1. 雙向對齊日期欄位名稱 (同時相容 race_date 與 date)
        if "date" in df.columns and "race_date" not in df.columns:
            df["race_date"] = df["date"]
        elif "race_date" in df.columns and "date" not in df.columns:
            df["date"] = df["race_date"]

        # 2. 數據清洗與名次處理 (支援平馬如 '1 DH' 正則提取)
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

        # 7. 資料品質與防洩漏檢查
        LeakageGuard.validate_feature_dataframe(
            df_base, required_keys=cls.PRIMARY_KEYS
        )

        return df_base

    @classmethod
    def build_from_sqlite(
        cls,
        db_path: Optional[str] = None,
        db_manager: Optional[DBManager] = None,
    ) -> pd.DataFrame:
        """透過 DBManager 載入完整賽事、馬匹與分段資料並建立 Target 骨架。"""
        if db_manager is None:
            if db_path is not None:
                db_manager = DBManager(db_path=db_path)
            else:
                db_manager = DBManager()

        # 使用 DBManager 封裝好的多表 JOIN + 分段轉置查詢
        df_raw = db_manager.load_all_merged_race_data()

        return cls.build_from_dataframe(df_raw)

    @staticmethod
    def _sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
        """清洗退跑、無效名次與 Null 值 (支援平馬如 '1 DH' 提取)。"""
        df["placing_str"] = df["placing"].astype(str).str.upper().str.strip()

        invalid_pos_keywords = [
            "WV", "SCR", "DNF", "DISQ", "FE", "PU", "UR", "NAN", "NONE"
        ]
        valid_mask = (
            ~df["placing_str"].isin(invalid_pos_keywords)
            & df["placing"].notnull()
        )
        df = df[valid_mask].copy()

        # 🔒 使用正則提取數字，避免 '1 DH' 等平馬名次被 to_numeric 轉成 NaN
        extracted_rank = df["placing_str"].str.extract(r"(\d+)")[0]
        df["craft_rank"] = pd.to_numeric(extracted_rank, errors="coerce")

        df = df[df["craft_rank"].notnull() & (df["craft_rank"] > 0)].copy()
        df = df[df["race_id"].notnull() & df["horse_id"].notnull()].copy()
        return df

    @staticmethod
    def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
        """嚴格按 race_date, race_id, horse_id 時間升冪排序。"""
        df["race_date"] = pd.to_datetime(df["race_date"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

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