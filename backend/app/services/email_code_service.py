from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from secrets import randbelow
import smtplib
from threading import Lock

from fastapi import HTTPException, status

from app.core.config import settings
from app.utils.crypto import normalize_email


@dataclass
class EmailCodeEntry:
    code: str
    expires_at: datetime
    last_sent_at: datetime


class EmailCodeService:
    _entries: dict[str, EmailCodeEntry] = {}
    _lock = Lock()

    def send_code(self, email: str, username: str) -> None:
        normalized_email = normalize_email(email)
        if normalized_email is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="邮箱格式不正确")
        now = datetime.now(timezone.utc)

        with self._lock:
            entry = self._entries.get(normalized_email)
            if entry and (now - entry.last_sent_at).total_seconds() < settings.email_code_cooldown_seconds:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"验证码发送过于频繁，请 {settings.email_code_cooldown_seconds} 秒后重试",
                )

            code = f"{randbelow(1_000_000):06d}"
            self._entries[normalized_email] = EmailCodeEntry(
                code=code,
                expires_at=now + timedelta(minutes=settings.email_code_expire_minutes),
                last_sent_at=now,
            )

        self._send_email(normalized_email, username, code)

    def verify_code(self, email: str, code: str) -> bool:
        normalized_email = normalize_email(email)
        if normalized_email is None:
            return False
        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(normalized_email)
            if entry is None:
                return False
            if entry.expires_at < now:
                self._entries.pop(normalized_email, None)
                return False
            if entry.code != code:
                return False

            self._entries.pop(normalized_email, None)
            return True

    def _send_email(self, email: str, username: str, code: str) -> None:
        if not settings.smtp_user or not settings.smtp_password:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="邮箱服务尚未配置，请联系管理员",
            )

        message = EmailMessage()
        message["From"] = f"{settings.smtp_sender_name} <{settings.smtp_user}>"
        message["To"] = email
        message["Subject"] = "智能车载视觉系统邮箱验证码"
        message.set_content(
            "\n".join(
                [
                    f"{username}，你好：",
                    "",
                    f"你的登录验证码是：{code}",
                    f"验证码 {settings.email_code_expire_minutes} 分钟内有效，请勿泄露给他人。",
                ]
            )
        )

        try:
            if settings.smtp_use_ssl:
                with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15) as server:
                    server.login(settings.smtp_user, settings.smtp_password)
                    server.send_message(message)
            else:
                with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
                    server.starttls()
                    server.login(settings.smtp_user, settings.smtp_password)
                    server.send_message(message)
        except smtplib.SMTPAuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="邮箱认证失败，请确认已开启 163 SMTP 服务，并在 SMTP_PASSWORD 中填写客户端授权码",
            ) from exc
        except (OSError, smtplib.SMTPException) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="验证码邮件发送失败，请检查邮箱配置或稍后重试",
            ) from exc
