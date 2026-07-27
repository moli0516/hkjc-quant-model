import json
import pathlib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config.settings import settings
from database.db_manager import DBManager  # 引用你專案中的 DBManager


class Feature_pipeline:

    def __init__(self):
        # 1. 初始化 DB 連線
        self.db = DBManager()
        self.engine = self.db.engine
        self.alphas = settings.smoothing_alphas

        # 2. 從資料庫讀取資料（SQL JOIN 替換原本的 JSON 載入）
        print("📂 開始從 SQLite 資料庫讀取 Raw Data...")
        self.df = self._load_data_from_db()

        # 3. 確定按日期和賽事 ID 排序，確保 Time-Series 計算無 Leakage
        self.df = self.df.sort_values(
            by=["date", "races.race_id"]
        ).reset_index(drop=True)

        self.bl_win = 1 / 14
        self.bl_place = 3 / 14

    def _load_data_from_db(self):
        """從 SQLite 資料庫讀取 races 與 race_results 並自動完成欄位映射"""
        query = """
        SELECT 
            r.race_id AS "races.race_id",
            r.date,
            r.venue,
            r.race_no,
            r.race_class AS "class",
            r.distance AS length,
            r.track_condition AS "races.track_condition",
            r.track_texture,
            r.track_type,
            
            res.horse_id,
            res.horse_name,
            res.placing,
            res.draw,
            res.jockey,
            res.trainer,
            res.actual_weight AS weight,
            res.declared_weight AS rank_weight,
            res.win_odds AS odds,
            res.finish_time_sec AS finished_time_sec,
            res.margin_len,
            res.rating
        FROM race_results res
        INNER JOIN races r ON res.race_id = r.race_id
        WHERE res.placing IS NOT NULL
        """

        with self.engine.connect() as conn:
            df = pd.read_sql_query(query, conn)

        print(f"✅ 資料庫資料讀取成功！共載入 {len(df)} 筆馬匹賽果紀錄。")

        # 數據類型補正
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y/%m/%d")
        df["length"] = pd.to_numeric(df["length"], errors="coerce")
        df["finished_time_sec"] = pd.to_numeric(
            df["finished_time_sec"], errors="coerce"
        )
        df["placing"] = pd.to_numeric(df["placing"], errors="coerce")
        df["draw"] = pd.to_numeric(df["draw"], errors="coerce")
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["rank_weight"] = pd.to_numeric(df["rank_weight"], errors="coerce")
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        return df

    def _build_win_placing(self):
        self.df["is_win"] = (self.df["placing"] == 1).astype(int)
        self.df["is_place"] = (self.df["placing"] < 4).astype(int)

    def _build_rating_features(self):
        class_avg_map = {1: 100, 2: 85, 3: 70, 4: 50, 5: 30}
        self.df["rating_is_real"] = self.df["rating"].notna()
        mask = self.df["rating_is_real"] == False

        # 針對每一班的缺失評分進行填充
        for cls, avg_val in class_avg_map.items():
            # 處理 class 可能為字串或數字的狀況
            self.df.loc[
                mask
                & (
                    (self.df["class"] == cls)
                    | (self.df["class"] == str(cls))
                ),
                "rating",
            ] = avg_val

        self.df["rating"] = self.df["rating"].astype(float)

    def _build_h_speed_z_features(self):
        # 1. 計算絕對速度 (Length / Finished Time)
        self.df["h_speed"] = self.df["length"] / self.df["finished_time_sec"]

        # 2. 計算單場賽事內的速度 Z-score
        self.df["h_speed_z"] = (
            self.df.groupby("races.race_id")["h_speed"]
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
            .fillna(0)
        )

        # 3. 按馬匹分組計算歷史速度特徵
        h_grpby_z = self.df.groupby("horse_id")["h_speed_z"]

        r2_mean = h_grpby_z.transform(
            lambda x: x.shift(1).rolling(window=2, min_periods=1).mean()
        )
        r15_mean = h_grpby_z.transform(
            lambda x: x.shift(1).rolling(window=15, min_periods=1).mean()
        )

        self.df["h_mean_speed_z_15"] = r15_mean.fillna(0)
        self.df["h_speed_z_momentum"] = (r2_mean - r15_mean).fillna(0)

        self.df["h_rolling_2_speed_z_std"] = h_grpby_z.transform(
            lambda x: x.shift(1).rolling(window=2, min_periods=1).std()
        ).fillna(0)
        self.df["h_rolling_15_speed_z_std"] = h_grpby_z.transform(
            lambda x: x.shift(1).rolling(window=15, min_periods=1).std()
        ).fillna(0)

        self.df["h_race_count_history"] = h_grpby_z.transform(
            lambda x: x.shift(1).expanding(min_periods=1).count()
        ).fillna(0)

    def _build_advanced_texture_and_weight_features(self):
        self.df["is_turf"] = (
            self.df["track_type"]
            .astype(str)
            .str.lower()
            .str.contains("草地")
            .astype(int)
        )
        self.df["is_sand"] = (~self.df["is_turf"].astype(bool)).astype(int)

        grouped = self.df.sort_values(["horse_id", "date"]).groupby("horse_id")

        sand_place_numerator = (
            grouped.apply(
                lambda x: (x["is_place"] * x["is_sand"])
                .shift(1)
                .rolling(window=5, min_periods=1)
                .sum()
            )
        ).reset_index(level=0, drop=True)
        sand_place_denominator = (
            grouped.apply(
                lambda x: x["is_sand"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .sum()
            )
        ).reset_index(level=0, drop=True)
        self.df["h_smoothed_rolling_5_texture_place_rate_sand"] = (
            sand_place_numerator / (sand_place_denominator + 1e-6)
        ).fillna(self.bl_place)

        turf_win_numerator = (
            grouped.apply(
                lambda x: (x["is_win"] * x["is_turf"])
                .shift(1)
                .rolling(window=5, min_periods=1)
                .sum()
            )
        ).reset_index(level=0, drop=True)
        turf_win_denominator = (
            grouped.apply(
                lambda x: x["is_turf"]
                .shift(1)
                .rolling(window=5, min_periods=1)
                .sum()
            )
        ).reset_index(level=0, drop=True)
        self.df["h_smoothed_rolling_5_texture_win_rate_turf"] = (
            turf_win_numerator / (turf_win_denominator + 1e-6)
        ).fillna(self.bl_win)

        avg_weight_last_5 = grouped["weight"].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        self.df["weight_impact_score"] = (
            self.df["weight"] / (avg_weight_last_5 + 1e-6)
        ).fillna(1.0)

    def _build_smoothing_features(self, src, target):
        alpha = self.alphas.get(
            src,
            settings._data.get("smoothing_params", {}).get("default_alpha", 20),
        )
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]

        grp_cnt = grp_src.transform(
            lambda x: x.shift(1).expanding(min_periods=1).count()
        ).fillna(0)
        grp_sum = grp_src.transform(
            lambda x: x.shift(1).expanding(min_periods=1).sum()
        ).fillna(0)

        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)

    def _build_smoothing_rolling_n_features(self, src, target, n):
        alpha = self.alphas.get(
            src,
            settings._data.get("smoothing_params", {}).get("default_alpha", 20),
        )
        bl = self.bl_win if "win" in target else self.bl_place
        grp_src = self.df.groupby(src)[target]

        grp_cnt = grp_src.transform(
            lambda x: x.shift(1).rolling(window=n, min_periods=1).count()
        ).fillna(0)
        grp_sum = grp_src.transform(
            lambda x: x.shift(1).rolling(window=n, min_periods=1).sum()
        ).fillna(0)

        return ((grp_sum + alpha * bl) / (grp_cnt + alpha)).fillna(bl)

    def _build_advanced_draw_features(self):
        track_bias_map = {"A": 1.2, "B": 1.1, "C": 0.8, "C+3": 0.7}
        self.df["track_bias_factor"] = (
            self.df["track_type"].map(track_bias_map).fillna(1.0)
        )
        self.df["adj_draw"] = self.df["draw"] * self.df["track_bias_factor"]

        self.df["draw_speed_interaction"] = (
            self.df["draw"] * self.df["h_mean_speed_z_15"]
        )

        self.df["rating"] = self.df["rating"].astype(float)
        race_avg_rating = self.df.groupby("races.race_id")["rating"].transform(
            "mean"
        )

        self.df["rating_vs_race_avg"] = self.df["rating"] - race_avg_rating
        self.df["rating_strength_score"] = self.df[
            "rating_vs_race_avg"
        ] * self.df["rating_is_real"].astype(int)

    def _build_rank_weight_features(self):
        self.df["z_rating_vs_race_avg"] = (
            self.df.groupby("race_unique_id")["rating_vs_race_avg"]
            .transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
            .fillna(0)
        )

        self.df["rating_x_rank_weight"] = (
            self.df["rating_vs_race_avg"] * self.df["rank_weight"]
        )
        self.df["jockey_adaptability_x_rank_weight"] = (
            self.df["j_track_smoothed_place_rate"] * self.df["rank_weight"]
        )
        self.df["form_x_rank_weight"] = (
            self.df["h_smoothed_rolling_5_place_rate"] * self.df["rank_weight"]
        )

    def _build_weight_features(self):
        self.df = self.df.sort_values(by=["horse_id", "race_unique_id"])

        self.df["avg_rank_weight_last_3"] = self.df.groupby("horse_id")[
            "rank_weight"
        ].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )

        self.df["avg_rank_weight_last_3"] = self.df[
            "avg_rank_weight_last_3"
        ].fillna(self.df["rank_weight"])
        self.df["weight_delta"] = (
            self.df["rank_weight"] - self.df["avg_rank_weight_last_3"]
        )

        self.df["load_ratio"] = self.df["weight"] / (
            self.df["rank_weight"] + 1e-6
        )

        self.df["delta_x_rank"] = (
            self.df["weight_delta"] * self.df["rank_weight"]
        )
        self.df["load_ratio_x_rank"] = (
            self.df["load_ratio"] * self.df["rank_weight"]
        )

    def run(self):
        # 1. 建立基準標籤與 ID
        self._build_win_placing()
        self.df["race_unique_id"] = (
            self.df["date"].astype(str)
            + "_"
            + self.df["races.race_id"].astype(str)
        )

        # 2. 建立馬匹速度特徵
        self._build_h_speed_z_features()

        # 3. 建立評分與基本特徵
        self._build_rating_features()

        # 4. 賠率與高維度 ID 交互組合
        self.df["win_odds_inv"] = 1 / (self.df["odds"] + 1e-6)
        self.df["log_win_odds"] = np.log1p(self.df["odds"])
        self.df["jockey_trainer"] = (
            self.df["jockey"].astype(str) + "_" + self.df["trainer"].astype(str)
        )
        self.df["track_detailed"] = (
            self.df["venue"].astype(str)
            + "_"
            + self.df["track_texture"].astype(str)
        )
        self.df["rail_detailed"] = (
            self.df["track_detailed"].astype(str)
            + "_"
            + self.df["track_type"].astype(str)
        )
        self.df["yeild_detailed"] = (
            self.df["venue"].astype(str)
            + "_"
            + self.df["races.track_condition"].astype(str)
        )
        self.df["env_core"] = (
            self.df["track_detailed"].astype(str)
            + "_"
            + self.df["length"].astype(str)
        )
        self.df["env_detail"] = (
            self.df["env_core"]
            + "_"
            + self.df["track_type"].astype(str)
            + "_"
            + self.df["races.track_condition"].astype(str)
        )

        self.df["horse_track_detailed"] = (
            self.df["horse_id"].astype(str)
            + "_"
            + self.df["track_detailed"].astype(str)
        )
        self.df["horse_env_core"] = (
            self.df["horse_id"].astype(str)
            + "_"
            + self.df["env_core"].astype(str)
        )
        self.df["horse_yeild_detailed"] = (
            self.df["horse_id"].astype(str)
            + "_"
            + self.df["yeild_detailed"].astype(str)
        )

        self.df["jockey_track_detailed"] = (
            self.df["jockey"].astype(str)
            + "_"
            + self.df["track_detailed"].astype(str)
        )
        self.df["jockey_env_core"] = (
            self.df["jockey"].astype(str)
            + "_"
            + self.df["env_core"].astype(str)
        )
        self.df["jockey_yeild_detailed"] = (
            self.df["jockey"].astype(str)
            + "_"
            + self.df["yeild_detailed"].astype(str)
        )

        self.df["trainer_track_detailed"] = (
            self.df["trainer"].astype(str)
            + "_"
            + self.df["track_detailed"].astype(str)
        )
        self.df["trainer_yeild_detailed"] = (
            self.df["trainer"].astype(str)
            + "_"
            + self.df["yeild_detailed"].astype(str)
        )

        self.df["actual_rank_score"] = (
            self.df["placing"]
            .map({1: 15, 2: 7, 3: 3, 4: 1})
            .fillna(0)
            .astype(int)
        )

        # 5. 計算全局平滑特徵
        smooths = {
            "j_smoothed_win_rate": ("jockey", "is_win"),
            "t_smoothed_win_rate": ("trainer", "is_win"),
            "jt_smoothed_win_rate": ("jockey_trainer", "is_win"),
            "h_smoothed_win_rate": ("horse_id", "is_win"),
            "d_smoothed_win_rate": ("draw", "is_win"),
            "j_smoothed_place_rate": ("jockey", "is_place"),
            "t_smoothed_place_rate": ("trainer", "is_place"),
            "jt_smoothed_place_rate": ("jockey_trainer", "is_place"),
            "h_smoothed_place_rate": ("horse_id", "is_place"),
            "d_smoothed_place_rate": ("draw", "is_place"),
            "h_track_smoothed_win_rate": ("horse_track_detailed", "is_win"),
            "h_track_smoothed_place_rate": ("horse_track_detailed", "is_place"),
            "h_env_smoothed_win_rate": ("horse_env_core", "is_win"),
            "h_env_smoothed_place_rate": ("horse_env_core", "is_place"),
            "h_yield_smoothed_win_rate": ("horse_yeild_detailed", "is_win"),
            "h_yield_smoothed_place_rate": (
                "horse_yeild_detailed",
                "is_place",
            ),
            "j_track_smoothed_place_rate": (
                "jockey_track_detailed",
                "is_place",
            ),
            "j_env_smoothed_place_rate": ("jockey_env_core", "is_place"),
            "j_yield_smoothed_place_rate": (
                "jockey_yeild_detailed",
                "is_place",
            ),
            "t_track_smoothed_place_rate": (
                "trainer_track_detailed",
                "is_place",
            ),
            "t_yield_smoothed_place_rate": (
                "trainer_yeild_detailed",
                "is_place",
            ),
        }

        for name, (src, target) in smooths.items():
            self.df[name] = self._build_smoothing_features(src, target)

        # 6. 計算滑動視窗平滑特徵
        smooth_rollings_n = {
            "j_smoothed_rolling_30_win_rate": ("jockey", "is_win", 30),
            "t_smoothed_rolling_30_win_rate": ("trainer", "is_win", 30),
            "jt_smoothed_rolling_15_win_rate": ("jockey_trainer", "is_win", 15),
            "h_smoothed_rolling_5_win_rate": ("horse_id", "is_win", 5),
            "j_smoothed_rolling_30_place_rate": ("jockey", "is_place", 30),
            "t_smoothed_rolling_30_place_rate": ("trainer", "is_place", 40),
            "jt_smoothed_rolling_15_place_rate": (
                "jockey_trainer",
                "is_place",
                15,
            ),
            "h_smoothed_rolling_5_place_rate": ("horse_id", "is_place", 5),
        }

        for name, (src, target, n) in smooth_rollings_n.items():
            self.df[name] = self._build_smoothing_rolling_n_features(
                src, target, n
            )

        # 7. 建立優化後的沙/草特徵與負重影響特徵
        self._build_advanced_texture_and_weight_features()

        # 8. 計算檔位與評分優勢基本衍生
        self._build_advanced_draw_features()

        # 9. 計算重量特徵
        self._build_weight_features()

        # 10. 計算交叉權重特徵
        self._build_rank_weight_features()

        # 最終檢查與匯出
        print(f"📅 資料集最大日期: {self.df['date'].max()}")
        self.df.to_parquet(settings.features_parquet_path, index=False)
        print(
            f"🎉 特徵工程完成！檔案已儲存至 {settings.features_parquet_path}"
        )


if __name__ == "__main__":
    feature_pipeline = Feature_pipeline()
    feature_pipeline.run()