from sqlalchemy import Column, Float, Integer, String
from database.models.base import Base


class RaceTrackwork(Base):
    __tablename__ = "race_trackwork"

    trackwork_id = Column(Integer, primary_key=True, autoincrement=True)
    horse_id = Column(String(10), index=True, nullable=False)
    horse_name = Column(String(50), nullable=True)
    work_date = Column(String(20), index=True, nullable=False)
    work_type = Column(String(50), nullable=True)
    workout_desc = Column(String(255), nullable=True)
    rider = Column(String(50), nullable=True)
    distance = Column(Integer, nullable=True)
    finish_time_sec = Column(Float, nullable=True)
    last_sectional_sec = Column(Float, nullable=True)
    sectional_count = Column(Integer, default=0)
    remark = Column(String(255), nullable=True)
