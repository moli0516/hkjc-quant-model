from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from database.models.base import Base


class RaceResult(Base):
    __tablename__ = "race_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    horse_id = Column(String(10), index=True)
    horse_name = Column(String(50), nullable=False)
    placing = Column(Integer)
    draw = Column(Integer)
    jockey = Column(String(50), index=True)
    trainer = Column(String(50), index=True)
    actual_weight = Column(Float)
    declared_weight = Column(Float)
    win_odds = Column(Float)
    finish_time_sec = Column(Float)
    margin_len = Column(Float)
    rating = Column(Integer)

    race = relationship("Race", back_populates="results")
