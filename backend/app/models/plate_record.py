from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class PlateRecord(BaseModel):
    __tablename__ = "plate_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate_number: Mapped[str] = mapped_column(String(32), index=True)
    plate_color: Mapped[str] = mapped_column(String(32))
    vehicle_type: Mapped[str] = mapped_column(String(32), default="未识别")
    confidence: Mapped[float] = mapped_column(Float)
    source_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
