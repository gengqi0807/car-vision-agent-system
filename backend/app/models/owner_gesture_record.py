from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class OwnerGestureRecord(BaseModel):
    __tablename__ = "owner_gesture_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    gesture: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    control_action: Mapped[str] = mapped_column(String(128))
