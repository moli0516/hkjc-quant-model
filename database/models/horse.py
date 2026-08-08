from sqlalchemy import Column, Date, Float, Integer, String
from database.models.base import Base


class Horse(Base):
    __tablename__ = "horses"

    horse_code = Column(String(20), primary_key=True)
    origin = Column(String(50), nullable=True)
    age = Column(Integer, nullable=True)
    color = Column(String(20), nullable=True)
    sex = Column(String(20), nullable=True)
    import_type = Column(String(50), nullable=True)
    season_stakes = Column(Float, default=0.0)
    total_stakes = Column(Float, default=0.0)
    wins = Column(Integer, default=0)
    seconds = Column(Integer, default=0)
    thirds = Column(Integer, default=0)
    total_runs = Column(Integer, default=0)
    recent_10_races_count = Column(Integer, nullable=True)
    current_location = Column(String(50), nullable=True)
    location_arrival_date = Column(Date, nullable=True)
    import_date = Column(Date, nullable=True)
    trainer = Column(String(50), index=True, nullable=True)
    owner = Column(String(100), nullable=True)
    current_rating = Column(Integer, index=True, nullable=True)
    season_start_rating = Column(Integer, nullable=True)
    sire = Column(String(100), index=True, nullable=True)
    dam = Column(String(100), nullable=True)
    damsire = Column(String(100), nullable=True)
