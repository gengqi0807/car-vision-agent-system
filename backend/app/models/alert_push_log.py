from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class AlertPushLog(BaseModel):
    __tablename__ = "alert_push_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    channel: Mapped[str] = mapped_column(String(32))
    target: Mapped[str] = mapped_column(String(128))
    success: Mapped[bool] = mapped_column(Boolean, default=False)
