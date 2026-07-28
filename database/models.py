from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Race(Base):
    __tablename__ = "races"

    race_id = Column(String(50), primary_key=True)
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
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
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
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
    )
    horse_id = Column(String(10))
    horse_name = Column(String(50), nullable=False)
    section_no = Column(Integer, nullable=False)
    position = Column(Integer)
    sectional_time_sec = Column(Float)
    margin_behind = Column(String(20))

    race = relationship("Race", back_populates="sectionals")


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