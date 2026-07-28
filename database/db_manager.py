import pathlib
import pandas as pd
from database.models import Base  # 從 models 載入 Base
from sqlalchemy import create_engine, inspect, text
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