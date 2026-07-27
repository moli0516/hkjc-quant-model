import pathlib
import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    inspect
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
Base = declarative_base()


# ==========================================
# SQLAlchemy Models (3NF Normalized Schema)
# ==========================================
class Race(Base):
    __tablename__ = "races"

    race_id = Column(String(50), primary_key=True)  # 例: "2024-01-01_ST_1"
    date = Column(String(20), nullable=False)
    venue = Column(String(10), nullable=False)
    race_no = Column(Integer, nullable=False)
    race_class = Column(String(10))
    distance = Column(Integer)
    track_condition = Column(String(20))
    track_texture = Column(String(20))
    track_type = Column(String(10))

    results = relationship(
        "RaceResult", back_populates="race", cascade="all, delete-orphan"
    )
    sectionals = relationship(
        "RaceSectional", back_populates="race", cascade="all, delete-orphan"
    )


class RaceResult(Base):
    __tablename__ = "race_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False
    )
    horse_id = Column(String(10))
    horse_name = Column(String(50), nullable=False)
    placing = Column(Integer)
    draw = Column(Integer)
    jockey = Column(String(50))
    trainer = Column(String(50))
    actual_weight = Column(Float)
    declared_weight = Column(Float)
    win_odds = Column(Float)
    finish_time_sec = Column(Float)
    margin_len = Column(Float)
    rating = Column(Integer)

    race = relationship("Race", back_populates="results")


class RaceSectional(Base):
    __tablename__ = "race_sectionals"

    sec_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50), ForeignKey("races.race_id", ondelete="CASCADE"), nullable=False
    )
    horse_id = Column(String(10))
    horse_name = Column(String(50), nullable=False)
    section_no = Column(Integer, nullable=False)
    position = Column(Integer)
    sectional_time_sec = Column(Float)
    margin_behind = Column(String(20))

    race = relationship("Race", back_populates="sectionals")


# ==========================================
# DB Manager
# ==========================================
class DBManager:

    def __init__(self, db_path=pathlib.Path(__file__).parent / "hkjc_racing.db"):
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

    def insert_dataframes(self, tables_dict):
        if not tables_dict:
            return
        
        for table_name, df in tables_dict.items():
            if df is not None and not df.empty:
                df.to_sql(
                    table_name, 
                    con=self.engine, 
                    if_exists="replace",  # 這裡改成 replace
                    index=False
                )
        print("【成功】資料庫數據已覆蓋寫入！")
    def has_race_results(self) -> bool:
        """檢查 race_results 表格是否存在且有資料"""
        inspector = inspect(self.engine)
        if not inspector.has_table("race_results"):
            return False

        with self.engine.connect() as conn:
            from sqlalchemy import text

            result = conn.execute(
                text("SELECT COUNT(*) FROM race_results")
            ).scalar()
            return result > 0

    def get_pending_horse_ids(self) -> list:
        """從 race_results 表中提取所有不重複的 horse_id"""
        if not self.has_race_results():
            return []

        with self.engine.connect() as conn:
            from sqlalchemy import text

            # 抓取賽果出現過，但在 horse_profiles 還沒抓過 (或需要更新) 的 horse_id
            query = text("""
                SELECT DISTINCT res.horse_id 
                FROM race_results res
                LEFT JOIN horse_profiles hp ON res.horse_id = hp.horse_id
                WHERE hp.horse_id IS NULL AND res.horse_id IS NOT NULL
            """)
            results = conn.execute(query).fetchall()
            return [row[0] for row in results if row[0]]