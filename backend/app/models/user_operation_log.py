from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel


class UserOperationLog(BaseModel):
    __tablename__ = "user_operation_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64))
    module: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(128))
