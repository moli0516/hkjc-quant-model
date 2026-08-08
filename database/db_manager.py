import gc
import pathlib
from typing import Dict
import pandas as pd
from sqlalchemy import Column, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# 🌟 從重構後的 database.models 模組統一載入 Base
from database.models import Base


class DBManager:

    def __init__(
        self, db_path=pathlib.Path(__file__).parent / "hkjc_racing.db"
    ):
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def init_db(self):
        """初始化資料庫表格"""
        try:
            Base.metadata.create_all(self.engine)
            print("【成功】SQLAlchemy 資料庫表格已成功初始化！")
        except Exception as e:
            print(f"【錯誤】初始化資料庫失敗: {e}")

    def create_performance_indexes(self):
        """建立關鍵 JOIN 與時間範圍查詢之複合索引，防止 Temp 檔膨脹"""
        index_sqls = [
            # 1. 晨操表：馬匹與日期的複合索引 (極度重要！)
            "CREATE INDEX IF NOT EXISTS idx_trackwork_horse_date ON race_trackwork(horse_id, work_date);",
            # 2. 賽事表：日期與 ID 複合索引
            "CREATE INDEX IF NOT EXISTS idx_races_date_id ON races(date, race_id);",
            # 3. 賽果表： race_id 與 horse_id
            "CREATE INDEX IF NOT EXISTS idx_results_race_horse ON race_results(race_id, horse_id);",
            # 4. 分段表： race_id, horse_id 與 section_no
            "CREATE INDEX IF NOT EXISTS idx_sectionals_composite ON race_sectionals(race_id, horse_id, section_no);"
        ]
        with self.engine.begin() as conn:
            for sql in index_sqls:
                conn.execute(text(sql))
        print("⚡ [DBManager] 效能優化複合索引已確認建立！")

    def insert_dataframes(self, tables_dict: dict[str, pd.DataFrame]):
        if not tables_dict:
            return

        for table_name, df in tables_dict.items():
            if df is not None and not df.empty:
                df.to_sql(
                    table_name, con=self.engine, if_exists="replace", index=False
                )
        print("【成功】資料庫數據已覆蓋寫入！")

    def has_race_results(self) -> bool:
        """檢查 race_results 表格是否存在且有資料"""
        inspector = inspect(self.engine)
        if not inspector.has_table("race_results"):
            return False

        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM race_results")
            ).scalar()
            return result > 0

    def get_pending_horse_ids(self) -> list[str]:
        """從 race_results 表中提取所有不重複，且尚未存在於 horses 表格中的 horse_id"""
        if not self.has_race_results():
            return []

        inspector = inspect(self.engine)
        has_horses_table = inspector.has_table("horses")

        with self.engine.connect() as conn:
            if has_horses_table:
                query = text("""
                    SELECT DISTINCT r.horse_id 
                    FROM race_results r
                    LEFT JOIN horses h ON r.horse_id = h.horse_code
                    WHERE r.horse_id IS NOT NULL 
                      AND r.horse_id != '' 
                      AND h.horse_code IS NULL
                    ORDER BY r.horse_id DESC
                """)
            else:
                query = text("""
                    SELECT DISTINCT horse_id 
                    FROM race_results
                    WHERE horse_id IS NOT NULL AND horse_id != ''
                    ORDER BY horse_id DESC
                """)

            results = conn.execute(query).fetchall()
            return [row[0] for row in results if row[0]]

    # ---------------- 記憶體優化與分批讀寫增強 ----------------

    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """優化 DataFrame 記憶體佔用 (float64 -> float32, int64 -> int32)"""
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype("float32")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype("int32")
        return df

    def get_all_race_dates(self) -> list[str]:
        """取得資料庫中所有不重複的賽事日期 (按時間升序排序)"""
        if not self.has_race_results():
            return []

        with self.engine.connect() as conn:
            query = text("SELECT DISTINCT date FROM races ORDER BY date ASC")
            results = conn.execute(query).fetchall()
            return [row[0] for row in results if row[0]]
        
    def get_horse_name_id_mapping(self) -> Dict[str, str]:
        """讀取馬名至烙號 mapping（優先讀取 horses 主表，再以 race_results 補充）"""
        mapping = {}
        with self.engine.connect() as conn:
            query = text("""
                SELECT DISTINCT horse_name, horse_id
                FROM race_results
                WHERE horse_id IS NOT NULL AND horse_id != '' AND horse_name IS NOT NULL
            """)
            df = pd.read_sql_query(query, conn)
            for _, row in df.iterrows():
                mapping[row['horse_name']] = row['horse_id']
                
        return mapping

    def load_all_merged_race_data(self) -> pd.DataFrame:
        """一次性載入資料庫內所有賽事、馬匹與歷史晨操 / 試閘 Raw Context (已嚴格防範時間洩漏)"""
        
        self.create_performance_indexes()

        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA temp_store = MEMORY;"))
            conn.execute(text("PRAGMA cache_size = -64000;"))

        query = text("""
            WITH trackwork_raw AS (
                -- 🔒 僅提取基礎 Raw 數據，將特徵邏輯完全交給 Generator 處理
                SELECT 
                    res.race_id,
                    res.horse_id,
                    COUNT(DISTINCT CASE 
                        WHEN tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-14 days') 
                        THEN tw.work_date END) AS raw_tw_count_14d,
                    
                    COUNT(DISTINCT CASE 
                        WHEN tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-14 days') 
                            AND (tw.workout_type LIKE '%快跳%' OR tw.workout_type LIKE '%試閘%') 
                        THEN tw.work_date END) AS raw_tw_fast_count_14d,
                    
                    AVG(CASE 
                        WHEN tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-14 days') 
                            AND tw.work_time_sec > 0 AND tw.distance > 0 
                        THEN (tw.distance * 1.0 / tw.work_time_sec) END) AS raw_tw_avg_speed_14d,
                    
                    MAX(CASE 
                        WHEN tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-7 days') 
                            AND tw.remarks IS NOT NULL AND tw.remarks != '' 
                        THEN 1.0 ELSE 0.0 END) AS raw_tw_gear_flag,
                    
                    JULIANDAY(REPLACE(r.date, '/', '-')) - JULIANDAY(MAX(tw.work_date)) AS raw_tw_days_since_last,
                    
                    MAX(CASE 
                        WHEN tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-7 days') 
                            AND tw.rider = res.jockey 
                        THEN 1.0 ELSE 0.0 END) AS raw_tw_rider_is_jockey

                FROM race_results res
                INNER JOIN races r ON res.race_id = r.race_id
                INNER JOIN race_trackwork tw 
                    ON res.horse_id = tw.horse_id 
                AND tw.work_date >= DATE(REPLACE(r.date, '/', '-'), '-28 days') 
                AND tw.work_date < REPLACE(r.date, '/', '-')
                GROUP BY res.race_id, res.horse_id
            ),

            -- =====================================================
            -- 試閘 (Barrier Trials) Raw Context
            -- =====================================================
            trails_raw AS (
                SELECT 
                    res.race_id,
                    res.horse_id,

                    -- 1. 距離最近一次試閘天數
                    JULIANDAY(REPLACE(r.date, '/', '-')) - JULIANDAY(MAX(t.date)) 
                        AS raw_tr_days_since_last,

                    -- 2. 過去 90 天試閘次數
                    COUNT(DISTINCT CASE 
                        WHEN t.date >= DATE(REPLACE(r.date, '/', '-'), '-90 days') 
                        THEN t.trial_id END) AS raw_tr_count_90d,

                    -- 3. 過去 90 天及格次數
                    COUNT(DISTINCT CASE 
                        WHEN t.date >= DATE(REPLACE(r.date, '/', '-'), '-90 days') 
                            AND tr.result_remark LIKE '%及格%'
                        THEN t.trial_id END) AS raw_tr_pass_count_90d,

                    -- 4. 最近一次試閘是否及格
                    MAX(CASE WHEN t.date = max_t.max_date THEN 
                        CASE 
                            WHEN tr.result_remark LIKE '%及格%' THEN 1.0
                            WHEN tr.result_remark LIKE '%不及格%' THEN 0.0
                            ELSE NULL 
                        END 
                    END) AS raw_tr_last_pass_flag,

                    -- 5. 最近一次試閘相對時間差（秒）
                    MAX(CASE WHEN t.date = max_t.max_date 
                        THEN tr.finish_time_sec - t.finish_time_sec END) 
                        AS raw_tr_last_relative_time,

                    -- 6. 最近一次試閘距離
                    MAX(CASE WHEN t.date = max_t.max_date THEN t.distance END) 
                        AS raw_tr_last_distance,

                    -- 7. 最近一次試閘場地類型
                    MAX(CASE WHEN t.date = max_t.max_date THEN t.track_type END) 
                        AS raw_tr_last_track_type,

                    -- 8. 最近一次試閘 margin
                    MAX(CASE WHEN t.date = max_t.max_date THEN tr.margin_len END) 
                        AS raw_tr_last_margin_len,

                    -- 9. 最近一次試閘走位
                    MAX(CASE WHEN t.date = max_t.max_date THEN tr.running_position END) 
                        AS raw_tr_last_running_position,

                    -- 10. 最近一次試閘評述
                    MAX(CASE WHEN t.date = max_t.max_date THEN tr.performance_comment END) 
                        AS raw_tr_last_comment

                FROM race_results res
                INNER JOIN races r ON res.race_id = r.race_id
                INNER JOIN trail_results tr ON res.horse_id = tr.horse_id
                INNER JOIN trails t 
                    ON tr.trial_id = t.trial_id
                AND t.date < REPLACE(r.date, '/', '-')
                AND t.date >= DATE(REPLACE(r.date, '/', '-'), '-180 days')
                -- 先算出每匹馬在該場賽事前的最新試閘日期
                INNER JOIN (
                    SELECT 
                        tr2.horse_id,
                        r2.race_id,
                        MAX(t2.date) AS max_date
                    FROM race_results res2
                    JOIN races r2 ON res2.race_id = r2.race_id
                    JOIN trail_results tr2 ON res2.horse_id = tr2.horse_id
                    JOIN trails t2 
                        ON tr2.trial_id = t2.trial_id
                    AND t2.date < REPLACE(r2.date, '/', '-')
                    AND t2.date >= DATE(REPLACE(r2.date, '/', '-'), '-180 days')
                    GROUP BY tr2.horse_id, r2.race_id
                ) max_t 
                    ON max_t.horse_id = res.horse_id 
                AND max_t.race_id = res.race_id
                GROUP BY res.race_id, res.horse_id
            )

            SELECT 
                r.race_id,
                r.date,
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
                h.sire,

                -- 🌟 分段時間轉置
                MAX(CASE WHEN sec.section_no = 1 THEN sec.sectional_time_sec END) AS sec1_time,
                MAX(CASE WHEN sec.section_no = 2 THEN sec.sectional_time_sec END) AS sec2_time,
                MAX(CASE WHEN sec.section_no = 3 THEN sec.sectional_time_sec END) AS sec3_time,
                MAX(CASE WHEN sec.section_no = 4 THEN sec.sectional_time_sec END) AS sec4_time,
                MAX(CASE WHEN sec.section_no = 5 THEN sec.sectional_time_sec END) AS sec5_time,
                MAX(CASE WHEN sec.section_no = 6 THEN sec.sectional_time_sec END) AS sec6_time,

                -- 🌟 分段名次轉置
                MAX(CASE WHEN sec.section_no = 1 THEN sec.position END) AS pos_sec1,
                MAX(CASE WHEN sec.section_no = 2 THEN sec.position END) AS pos_sec2,
                MAX(CASE WHEN sec.section_no = 3 THEN sec.position END) AS pos_sec3,
                MAX(CASE WHEN sec.section_no = 4 THEN sec.position END) AS pos_sec4,
                MAX(CASE WHEN sec.section_no = 5 THEN sec.position END) AS pos_sec5,
                MAX(CASE WHEN sec.section_no = 6 THEN sec.position END) AS pos_sec6,

                -- 🔒 防洩漏晨操上下文聚合欄位
                COALESCE(tw.raw_tw_count_14d, 0) AS raw_tw_count_14d,
                COALESCE(tw.raw_tw_fast_count_14d, 0) AS raw_tw_fast_count_14d,
                COALESCE(tw.raw_tw_avg_speed_14d, 0.0) AS raw_tw_avg_speed_14d,
                COALESCE(tw.raw_tw_gear_flag, 0.0) AS raw_tw_gear_flag,
                COALESCE(tw.raw_tw_days_since_last, 999.0) AS raw_tw_days_since_last,
                COALESCE(tw.raw_tw_rider_is_jockey, 0.0) AS raw_tw_rider_is_jockey,

                -- 🔒 防洩漏試閘上下文聚合欄位
                COALESCE(tr.raw_tr_days_since_last, 999.0) AS raw_tr_days_since_last,
                tr.raw_tr_last_pass_flag,
                tr.raw_tr_last_relative_time,
                tr.raw_tr_last_distance,
                tr.raw_tr_last_track_type,
                COALESCE(tr.raw_tr_count_90d, 0) AS raw_tr_count_90d,
                COALESCE(tr.raw_tr_pass_count_90d, 0) AS raw_tr_pass_count_90d,
                tr.raw_tr_last_margin_len,
                tr.raw_tr_last_running_position,
                tr.raw_tr_last_comment

            FROM race_results res
            INNER JOIN races r ON res.race_id = r.race_id
            LEFT JOIN horses h ON res.horse_id = h.horse_code
            LEFT JOIN race_sectionals sec ON res.race_id = sec.race_id AND res.horse_id = sec.horse_id
            LEFT JOIN trackwork_raw tw ON res.race_id = tw.race_id AND res.horse_id = tw.horse_id
            LEFT JOIN trails_raw tr ON res.race_id = tr.race_id AND res.horse_id = tr.horse_id

            GROUP BY 
                r.race_id, r.date, r.venue, r.race_no, r.race_class, r.distance, 
                r.track_condition, r.track_texture, r.track_type, 
                res.horse_id, res.horse_name, res.placing, res.draw, res.jockey, res.trainer, 
                res.actual_weight, res.declared_weight, res.win_odds, res.finish_time_sec, 
                res.margin_len, res.rating, h.import_date, h.sire,

                -- 晨操欄位
                tw.raw_tw_count_14d, tw.raw_tw_fast_count_14d, tw.raw_tw_avg_speed_14d,
                tw.raw_tw_gear_flag, tw.raw_tw_days_since_last, tw.raw_tw_rider_is_jockey,

                -- 試閘欄位
                tr.raw_tr_days_since_last, tr.raw_tr_last_pass_flag, tr.raw_tr_last_relative_time,
                tr.raw_tr_last_distance, tr.raw_tr_last_track_type, tr.raw_tr_count_90d,
                tr.raw_tr_pass_count_90d, tr.raw_tr_last_margin_len,
                tr.raw_tr_last_running_position, tr.raw_tr_last_comment

            ORDER BY r.date ASC, r.race_id ASC, res.horse_id ASC
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql_query(query, conn)

        return self.optimize_memory(df)
    
    def save_feature_matrix(
        self,
        df: pd.DataFrame,
        table_name: str = "feature_matrix",
        if_exists: str = "append",
    ):
        """將特徵矩陣分批寫入資料庫"""
        if df is None or df.empty:
            return

        df = self.optimize_memory(df)

        with self.engine.begin() as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                index=False,
                dtype={"race_id": String(), "horse_id": String()},
            )
            
    def load_feature_result(self) -> pd.DataFrame:
        query = """
        SELECT 
            f.*,
            r.placing,
            r.win_odds,
            CASE WHEN r.placing = 1 THEN 1 ELSE 0 END AS is_win,
            CASE WHEN r.placing BETWEEN 1 AND 3 THEN 1 ELSE 0 END AS is_top3
        FROM feature_matrix f
        INNER JOIN race_results r 
          ON f.race_id = r.race_id 
         AND f.horse_id = r.horse_id
        ORDER BY f.race_id ASC;
        """
        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                query,
                conn
            )
            return df