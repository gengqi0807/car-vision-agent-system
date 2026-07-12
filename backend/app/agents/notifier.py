from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from urllib import request

from app.core.config import settings
from app.utils.websocket_manager import websocket_manager


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

        if settings.alert_email_recipients:
            results.append(self._send_email(message))

        if settings.alert_webhook_url:
            results.append(self._send_webhook(message))

        return results

    def _send_email(self, message: dict) -> dict:
        if not settings.smtp_user or not settings.smtp_password:
            return {
                "channel": "email",
                "target": ",".join(settings.alert_email_recipients),
                "success": False,
            }

        email = EmailMessage()
        email["From"] = f"{settings.smtp_sender_name} <{settings.smtp_user}>"
        email["To"] = ", ".join(settings.alert_email_recipients)
        level_text = {
            "critical": "严重",
            "warning": "警告",
            "info": "提示",
        }.get(str(message.get("level", "info")), str(message.get("level", "info")))
        email["Subject"] = f"[{level_text}] {message.get('title', '系统告警')}"
        email.set_content(
            "\n".join(
                [
                    f"Title: {message.get('title', '')}",
                    f"Source: {message.get('source', '')}",
                    f"Event Type: {message.get('event_type', '')}",
                    f"Summary: {message.get('summary', '')}",
                    f"Root Cause: {message.get('root_cause', '')}",
                    f"Impact Scope: {message.get('impact_scope', '')}",
                    f"Suggested Action: {message.get('suggested_action', '')}",
                ]
            )
        )

        try:
            if settings.smtp_use_ssl:
                with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15) as server:
                    server.login(settings.smtp_user, settings.smtp_password)
                    server.send_message(email)
            else:
                with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
                    server.starttls()
                    server.login(settings.smtp_user, settings.smtp_password)
                    server.send_message(email)
            success = True
        except Exception:
            success = False

        return {
            "channel": "email",
            "target": ",".join(settings.alert_email_recipients),
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
