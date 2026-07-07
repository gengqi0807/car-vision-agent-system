from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class UserOperationLog(BaseModel):
    __tablename__ = "user_operation_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    operation_type: Mapped[str] = mapped_column(String(32), index=True)
    response_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
