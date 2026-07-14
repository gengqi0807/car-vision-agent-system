from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class OwnerGestureRecord(BaseModel):
    __tablename__ = "owner_gesture_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    gesture: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    control_action: Mapped[str] = mapped_column(String(128))
    hand_landmarks: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    is_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
