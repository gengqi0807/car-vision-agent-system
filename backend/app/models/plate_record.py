from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PlateRecord(Base):
    __tablename__ = "plate_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    plate_number: Mapped[str] = mapped_column(String(16), index=True)
    plate_color: Mapped[str | None] = mapped_column(String(16), nullable=True)
    bbox: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
