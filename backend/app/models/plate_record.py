from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class PlateRecord(BaseModel):
    __tablename__ = "plate_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    plate_number: Mapped[str] = mapped_column(String(32), index=True)
    plate_color: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    source_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
