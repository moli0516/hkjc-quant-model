import gc
import pathlib
import pandas as pd
from database.models import Base  # 從 models 載入 Base
from sqlalchemy import Column, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker


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

    def load_merged_race_data_by_dates(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """只提取指定日期範圍內的賽事數據，避免全量載入導致 OOM"""
        query = text("""
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
                
                sec.section_no,
                sec.position,
                sec.sectional_time_sec,
                sec.margin_behind
                
            FROM race_results res
            INNER JOIN races r ON res.race_id = r.race_id
            LEFT JOIN horses h ON res.horse_id = h.horse_code
            LEFT JOIN race_sectionals sec ON res.race_id = sec.race_id AND res.horse_id = sec.horse_id
            WHERE r.date >= :start_date AND r.date <= :end_date
            ORDER BY r.date ASC, r.race_id ASC, res.horse_id ASC
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params={"start_date": start_date, "end_date": end_date},
            )

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