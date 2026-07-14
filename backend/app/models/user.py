from __future__ import annotations
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel
from app.utils.crypto import crypto_manager, normalize_email, normalize_phone, normalize_sensitive_value


class User(BaseModel):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    _legacy_email: Mapped[str | None] = mapped_column("email", String(128), unique=True, nullable=True)
    _legacy_phone: Mapped[str | None] = mapped_column("phone", String(20), unique=True, nullable=True)
    _legacy_wechat_openid: Mapped[str | None] = mapped_column("wechat_openid", String(128), unique=True, nullable=True)

    email_encrypted: Mapped[str | None] = mapped_column(String(512), nullable=True)
    email_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    phone_encrypted: Mapped[str | None] = mapped_column(String(512), nullable=True)
    phone_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    wechat_openid_encrypted: Mapped[str | None] = mapped_column(String(512), nullable=True)
    wechat_openid_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    role: Mapped[str] = mapped_column(
        Enum("admin", "user", name="user_role"),
        nullable=False,
        default="user",
        server_default="user",
    )
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=datetime.utcnow,
    )

    @staticmethod
    def build_email_hash(value: str | None) -> str | None:
        normalized = normalize_email(value)
        return crypto_manager.fingerprint(normalized) if normalized else None

    @staticmethod
    def build_phone_hash(value: str | None) -> str | None:
        normalized = normalize_phone(value)
        return crypto_manager.fingerprint(normalized) if normalized else None

    @staticmethod
    def build_wechat_openid_hash(value: str | None) -> str | None:
        normalized = normalize_sensitive_value(value)
        return crypto_manager.fingerprint(normalized) if normalized else None

    @staticmethod
    def _decrypt_or_none(ciphertext: str | None, legacy_value: str | None, *, normalizer) -> str | None:
        if ciphertext:
            return crypto_manager.decrypt(ciphertext)
        return normalizer(legacy_value)

    @property
    def email(self) -> str | None:
        return self._decrypt_or_none(self.email_encrypted, self._legacy_email, normalizer=normalize_email)

    @email.setter
    def email(self, value: str | None) -> None:
        normalized = normalize_email(value)
        self.email_encrypted = crypto_manager.encrypt(normalized) if normalized else None
        self.email_hash = self.build_email_hash(normalized)
        self._legacy_email = None

    @property
    def phone(self) -> str | None:
        return self._decrypt_or_none(self.phone_encrypted, self._legacy_phone, normalizer=normalize_phone)

    @phone.setter
    def phone(self, value: str | None) -> None:
        normalized = normalize_phone(value)
        self.phone_encrypted = crypto_manager.encrypt(normalized) if normalized else None
        self.phone_hash = self.build_phone_hash(normalized)
        self._legacy_phone = None

    @property
    def wechat_openid(self) -> str | None:
        return self._decrypt_or_none(
            self.wechat_openid_encrypted,
            self._legacy_wechat_openid,
            normalizer=normalize_sensitive_value,
        )

    @wechat_openid.setter
    def wechat_openid(self, value: str | None) -> None:
        normalized = normalize_sensitive_value(value)
        self.wechat_openid_encrypted = crypto_manager.encrypt(normalized) if normalized else None
        self.wechat_openid_hash = self.build_wechat_openid_hash(normalized)
        self._legacy_wechat_openid = None
