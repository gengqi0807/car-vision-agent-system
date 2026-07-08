from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class AlertLog(BaseModel):
    __tablename__ = "alert_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    level: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(128))
    summary: Mapped[str] = mapped_column(Text)
    impact_scope: Mapped[str | None] = mapped_column(String(255), nullable=True)
    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    suggested_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_json: Mapped[str | None] = mapped_column(Text, nullable=True)
