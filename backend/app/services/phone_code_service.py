from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import randbelow
from threading import Lock
from urllib.parse import quote
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.crypto import normalize_phone

logger = get_logger(__name__)


@dataclass
class PhoneCodeEntry:
    code: str
    expires_at: datetime
    last_sent_at: datetime


class PhoneCodeService:
    _entries: dict[str, PhoneCodeEntry] = {}
    _lock = Lock()

    def send_code(self, phone: str, username: str) -> str | None:
        normalized_phone = normalize_phone(phone)
        if normalized_phone is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="手机号格式不正确")

        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(normalized_phone)
            if entry and (now - entry.last_sent_at).total_seconds() < settings.phone_code_cooldown_seconds:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"验证码发送过于频繁，请 {settings.phone_code_cooldown_seconds} 秒后重试",
                )

            code = f"{randbelow(1_000_000):06d}"
            self._entries[normalized_phone] = PhoneCodeEntry(
                code=code,
                expires_at=now + timedelta(minutes=settings.phone_code_expire_minutes),
                last_sent_at=now,
            )

        return self._dispatch_code(normalized_phone, username, code)

    def verify_code(self, phone: str, code: str) -> bool:
        normalized_phone = normalize_phone(phone)
        if normalized_phone is None:
            return False

        now = datetime.now(timezone.utc)
        with self._lock:
            entry = self._entries.get(normalized_phone)
            if entry is None:
                return False
            if entry.expires_at < now:
                self._entries.pop(normalized_phone, None)
                return False
            if entry.code != code:
                return False

            self._entries.pop(normalized_phone, None)
            return True

    def _dispatch_code(self, phone: str, username: str, code: str) -> str | None:
        if settings.sms_mock_mode:
            logger.info(
                "Mock SMS login code generated for %s (%s): %s",
                username,
                phone,
                code,
            )
            return code

        provider = settings.sms_provider.strip().lower()
        if provider != "aliyun":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"暂不支持当前短信服务商: {settings.sms_provider}",
            )

        self._send_aliyun_sms(phone, code)
        logger.info("Aliyun SMS login code dispatched for %s (%s)", username, phone)
        return None

    def _send_aliyun_sms(self, phone: str, code: str) -> None:
        self._ensure_aliyun_config()

        template_param = json.dumps({"code": code}, ensure_ascii=False, separators=(",", ":"))
        params = {
            "AccessKeyId": settings.aliyun_sms_access_key_id,
            "Action": "SendSms",
            "Format": "JSON",
            "PhoneNumbers": phone,
            "RegionId": settings.aliyun_sms_region_id,
            "SignName": settings.aliyun_sms_sign_name,
            "SignatureMethod": "HMAC-SHA1",
            "SignatureNonce": uuid4().hex,
            "SignatureVersion": "1.0",
            "TemplateCode": settings.aliyun_sms_template_code,
            "TemplateParam": template_param,
            "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Version": settings.aliyun_sms_api_version,
        }
        params["Signature"] = self._sign_aliyun_request(params, settings.aliyun_sms_access_key_secret)
        body = self._build_form_body(params)

        request = Request(
            settings.aliyun_sms_endpoint,
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=settings.aliyun_sms_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="阿里云短信服务请求失败，请稍后重试",
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="阿里云短信服务返回了无效响应",
            ) from exc

        if payload.get("Code") != "OK":
            detail = str(payload.get("Message") or "阿里云短信发送失败")
            code_value = payload.get("Code")
            if code_value and code_value != "OK":
                detail = f"{code_value}: {detail}"
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

    def _ensure_aliyun_config(self) -> None:
        required_values = {
            "ALIYUN_SMS_ACCESS_KEY_ID": settings.aliyun_sms_access_key_id,
            "ALIYUN_SMS_ACCESS_KEY_SECRET": settings.aliyun_sms_access_key_secret,
            "ALIYUN_SMS_SIGN_NAME": settings.aliyun_sms_sign_name,
            "ALIYUN_SMS_TEMPLATE_CODE": settings.aliyun_sms_template_code,
        }
        missing = [key for key, value in required_values.items() if not value.strip()]
        if missing:
            joined = ", ".join(missing)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"阿里云短信配置不完整，请补充: {joined}",
            )

    def _sign_aliyun_request(self, params: dict[str, str], access_key_secret: str) -> str:
        sorted_items = sorted((key, value) for key, value in params.items() if key != "Signature")
        canonicalized_query = "&".join(
            f"{self._percent_encode(key)}={self._percent_encode(value)}"
            for key, value in sorted_items
        )
        string_to_sign = f"POST&{self._percent_encode('/')}"+ f"&{self._percent_encode(canonicalized_query)}"
        digest = hmac.new(
            f"{access_key_secret}&".encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _build_form_body(self, params: dict[str, str]) -> str:
        sorted_items = sorted(params.items())
        return "&".join(
            f"{self._percent_encode(key)}={self._percent_encode(value)}"
            for key, value in sorted_items
        )

    @staticmethod
    def _percent_encode(value: object) -> str:
        return quote(str(value), safe="~")
