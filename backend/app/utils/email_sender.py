from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from app.core.config import settings
from app.core.logger import get_logger


logger = get_logger(__name__)


def send_email_message(message: EmailMessage) -> None:
    last_error: Exception | None = None

    for use_ssl, port in _build_delivery_attempts():
        try:
            _deliver_message(message, use_ssl=use_ssl, port=port)
            if use_ssl != settings.smtp_use_ssl or port != settings.smtp_port:
                logger.warning(
                    "SMTP delivery succeeded after fallback: host=%s port=%s use_ssl=%s",
                    settings.smtp_host,
                    port,
                    use_ssl,
                )
            return
        except smtplib.SMTPAuthenticationError:
            raise
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPException, OSError) as exc:
            last_error = exc
            logger.warning(
                "SMTP delivery attempt failed: host=%s port=%s use_ssl=%s error=%s",
                settings.smtp_host,
                port,
                use_ssl,
                exc,
            )

    if last_error is not None:
        raise last_error
    raise smtplib.SMTPException("SMTP delivery failed before any attempt could complete.")


def _build_delivery_attempts() -> list[tuple[bool, int]]:
    attempts: list[tuple[bool, int]] = [(settings.smtp_use_ssl, settings.smtp_port)]
    fallback = (False, 587) if settings.smtp_use_ssl else (True, 465)
    if fallback not in attempts:
        attempts.append(fallback)
    return attempts


def _deliver_message(message: EmailMessage, *, use_ssl: bool, port: int) -> None:
    tls_context = ssl.create_default_context()

    if use_ssl:
        with smtplib.SMTP_SSL(settings.smtp_host, port, timeout=15, context=tls_context) as server:
            server.ehlo()
            _smtp_login(server)
            server.send_message(message)
        return

    with smtplib.SMTP(settings.smtp_host, port, timeout=15) as server:
        server.ehlo()
        server.starttls(context=tls_context)
        server.ehlo()
        _smtp_login(server)
        server.send_message(message)


def _smtp_login(server: smtplib.SMTP) -> None:
    auth_features = str(server.esmtp_features.get("auth", ""))
    auth_mechanisms = {item.strip().upper() for item in auth_features.split() if item.strip()}
    server.user = settings.smtp_user
    server.password = settings.smtp_password

    if "LOGIN" in auth_mechanisms:
        server.auth("LOGIN", server.auth_login, initial_response_ok=False)
        return

    if "PLAIN" in auth_mechanisms:
        server.auth("PLAIN", server.auth_plain, initial_response_ok=False)
        return

    server.login(settings.smtp_user, settings.smtp_password, initial_response_ok=False)
