from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class PoliceGestureRecord(BaseModel):
    __tablename__ = "police_gesture_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    gesture: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    source_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
