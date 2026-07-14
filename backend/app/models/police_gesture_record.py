from sqlalchemy import Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class PoliceGestureRecord(BaseModel):
    __tablename__ = "police_gesture_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gesture: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    keypoints: Mapped[list | None] = mapped_column(JSON, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
