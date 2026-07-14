from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class SystemMetric(BaseModel):
    __tablename__ = "system_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), index=True)
    metric_value: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
