from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from urllib import request

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.email_sender import send_email_message
from app.utils.websocket_manager import websocket_manager


logger = get_logger(__name__)


class Notifier:
    async def notify_alert(self, message: dict) -> list[dict]:
        results: list[dict] = []

        delivered = await websocket_manager.broadcast("alerts", {"type": "alert.created", "data": message})
        results.append(
            {
                "channel": "websocket",
                "target": f"alerts:{delivered}",
                "success": True,
            }
        )

        recipients = self._resolve_email_recipients()
        if recipients:
            results.append(self._send_email(message, recipients=recipients, default_subject="系统告警"))

        if settings.alert_webhook_url:
            results.append(self._send_webhook(message))

        return results

    def notify_monitor_log(self, message: dict) -> list[dict]:
        recipients = self._resolve_email_recipients()
        if not recipients:
            return []
        return [self._send_email(message, recipients=recipients, default_subject="系统日志")]

    def _resolve_email_recipients(self) -> list[str]:
        recipients = {item.strip() for item in settings.alert_email_recipients if item and item.strip()}
        return sorted(recipients)

    def _send_email(self, message: dict, *, recipients: list[str], default_subject: str) -> dict:
        if not settings.smtp_user or not settings.smtp_password:
            return {
                "channel": "email",
                "target": ",".join(recipients),
                "success": False,
            }

        email = EmailMessage()
        email["From"] = f"{settings.smtp_sender_name} <{settings.smtp_user}>"
        email["To"] = ", ".join(recipients)
        level_text = {
            "critical": "critical",
            "warning": "warning",
            "info": "info",
        }.get(str(message.get("level", "info")), str(message.get("level", "info")))
        email["Subject"] = f"[{level_text}] {message.get('title', default_subject)}"
        email.set_content(
            "\n".join(
                [
                    f"Title: {message.get('title', '')}",
                    f"Source: {message.get('source', '')}",
                    f"Event Type: {message.get('event_type', '')}",
                    f"Status: {message.get('status', '')}",
                    f"Confidence: {message.get('confidence', '')}",
                    f"Summary: {message.get('summary', '')}",
                    f"Details: {json.dumps(message.get('details', {}), ensure_ascii=False)}",
                    f"Root Cause: {message.get('root_cause', '')}",
                    f"Impact Scope: {message.get('impact_scope', '')}",
                    f"Suggested Action: {message.get('suggested_action', '')}",
                ]
            )
        )

        try:
            send_email_message(email)
            success = True
        except Exception as exc:
            logger.exception("Failed to send notification email to %s: %s", ",".join(recipients), exc)
            success = False

        return {
            "channel": "email",
            "target": ",".join(recipients),
            "success": success,
        }

    def _send_webhook(self, message: dict) -> dict:
        req = request.Request(
            settings.alert_webhook_url,
            data=json.dumps(message).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        success = True
        try:
            with request.urlopen(req, timeout=settings.alert_webhook_timeout_seconds):
                pass
        except Exception:
            success = False

        return {
            "channel": "webhook",
            "target": settings.alert_webhook_url,
            "success": success,
        }
