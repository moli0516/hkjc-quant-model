from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from database.models.base import Base


class Race(Base):
    __tablename__ = "races"

    race_id = Column(String(50), primary_key=True)
    date = Column(String(20), nullable=False, index=True)
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
