from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from database.models.base import Base


class RaceSectional(Base):
    __tablename__ = "race_sectionals"

    sec_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(
        String(50),
        ForeignKey("races.race_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    horse_id = Column(String(10), index=True)
    horse_name = Column(String(50), nullable=False)
    section_no = Column(Integer, nullable=False)
    position = Column(Integer)
    sectional_time_sec = Column(Float)
    margin_behind = Column(String(20))

    race = relationship("Race", back_populates="sectionals")
