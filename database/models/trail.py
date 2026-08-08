from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import relationship
from database.models import Base  # 或 from database.models.base import Base


class RaceTrial(Base):
    """HKJC 試閘組別元數據表 (Barrier Trials Metadata)"""

    __tablename__ = "trials"

    trial_id = Column(String(50), primary_key=True)
    date = Column(String(20), index=True, nullable=False)
    group_no = Column(Integer, nullable=False)
    basic_info = Column(String(255), nullable=True)
    venue = Column(String(20), nullable=True)
    track_type = Column(String(50), nullable=True)
    distance = Column(Integer, nullable=True)
    track_condition = Column(String(20), nullable=True)
    finish_time_sec = Column(Float, nullable=True)
    sec1_time = Column(Float, nullable=True)
    sec2_time = Column(Float, nullable=True)
    sec3_time = Column(Float, nullable=True)
    sec4_time = Column(Float, nullable=True)

    # 雙向一對多關係 (1 Trial -> N Trial Results)
    results = relationship(
        "RaceTrialResult",
        back_populates="trial",
        cascade="all, delete-orphan",
    )


class RaceTrialResult(Base):
    """HKJC 試閘參賽馬匹詳細結果表 (Barrier Trial Individual Results)"""

    __tablename__ = "trial_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    trial_id = Column(
        String(50),
        ForeignKey("trials.trial_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    horse_id = Column(String(20), index=True, nullable=False)
    horse_code = Column(String(20), index=True, nullable=True)
    horse_name = Column(String(50), nullable=True)
    placing = Column(Integer, nullable=True)
    draw = Column(Integer, nullable=True)
    jockey = Column(String(50), nullable=True)
    trainer = Column(String(50), nullable=True)
    gear = Column(String(50), nullable=True)
    margin_len = Column(Float, nullable=True)
    running_position = Column(String(100), nullable=True)
    finish_time_sec = Column(Float, nullable=True)
    result_remark = Column(String(255), nullable=True)
    performance_comment = Column(String(255), nullable=True)
    is_withdrawn = Column(Boolean, default=False, nullable=False)

    # 雙向多對一關係
    trial = relationship("RaceTrial", back_populates="results")