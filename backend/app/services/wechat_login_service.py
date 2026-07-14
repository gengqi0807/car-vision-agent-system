from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from threading import Lock
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logger import get_logger
from app.utils.crypto import normalize_sensitive_value

logger = get_logger(__name__)


@dataclass
class WechatLoginSession:
    session_id: str
    scene_token: str
    qr_code_url: str
    status: str
    created_at: datetime
    expires_at: datetime
    openid: str | None = None


class WechatLoginService:
    _sessions: dict[str, WechatLoginSession] = {}
    _lock = Lock()

    def create_session(self) -> WechatLoginSession:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=settings.wechat_login_expire_minutes)
        session_id = token_urlsafe(18)
        qr_code_url = self._build_qr_code_url(session_id)
        session = WechatLoginSession(
            session_id=session_id,
            scene_token=session_id,
            qr_code_url=qr_code_url,
            status="pending",
            created_at=now,
            expires_at=expires_at,
        )

        with self._lock:
            self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> WechatLoginSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="扫码会话不存在")
            if session.expires_at < datetime.now(timezone.utc):
                session.status = "expired"
            return session

    def confirm_session(self, session_id: str, openid: str | None = None) -> WechatLoginSession:
        normalized_openid = normalize_sensitive_value(openid)

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="扫码会话不存在")
            if session.expires_at < datetime.now(timezone.utc):
                session.status = "expired"
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="扫码会话已过期")

            session.status = "confirmed"
            session.openid = normalized_openid or f"mock-openid-{session.session_id}"
            logger.info("Wechat login session confirmed: %s", session.session_id)
            return session

    def reject_session(self, session_id: str) -> WechatLoginSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="扫码会话不存在")
            if session.expires_at < datetime.now(timezone.utc):
                session.status = "expired"
            else:
                session.status = "rejected"
            return session

    def consume_confirmed_session(self, session_id: str) -> WechatLoginSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="扫码会话不存在")
            if session.expires_at < datetime.now(timezone.utc):
                session.status = "expired"
                raise HTTPException(status_code=status.HTTP_410_GONE, detail="扫码会话已过期")
            if session.status != "confirmed":
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="扫码尚未确认")

            session.status = "authenticated"
            return session

    def handle_callback(self, code: str | None, state: str | None, error: str | None = None) -> WechatLoginSession:
        session_id = normalize_sensitive_value(state)
        if session_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="缺少扫码会话标识")

        if error:
            return self.reject_session(session_id)

        normalized_code = normalize_sensitive_value(code)
        if normalized_code is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="微信回调缺少授权码")

        openid = self._exchange_code_for_openid(normalized_code)
        return self.confirm_session(session_id=session_id, openid=openid)

    def uses_mock_mode(self) -> bool:
        return settings.wechat_mock_mode and not self.is_real_wechat_configured()

    def is_real_wechat_configured(self) -> bool:
        return all(
            [
                settings.wechat_open_platform_app_id.strip(),
                settings.wechat_open_platform_app_secret.strip(),
                settings.wechat_open_platform_redirect_uri.strip(),
            ]
        )

    def _build_qr_code_url(self, state: str) -> str:
        if self.is_real_wechat_configured():
            redirect_uri = quote(settings.wechat_open_platform_redirect_uri, safe="")
            return (
                "https://open.weixin.qq.com/connect/qrconnect"
                f"?appid={settings.wechat_open_platform_app_id}"
                f"&redirect_uri={redirect_uri}"
                f"&response_type=code&scope={settings.wechat_open_platform_scope}"
                f"&state={state}#wechat_redirect"
            )

        if self.uses_mock_mode():
            return f"weixin://mock-login?state={state}"

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="微信开放平台尚未配置，请联系管理员",
        )

    def _exchange_code_for_openid(self, code: str) -> str:
        if not self.is_real_wechat_configured():
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="微信开放平台尚未配置")

        params = urlencode(
            {
                "appid": settings.wechat_open_platform_app_id,
                "secret": settings.wechat_open_platform_app_secret,
                "code": code,
                "grant_type": "authorization_code",
            }
        )
        endpoint = f"https://api.weixin.qq.com/sns/oauth2/access_token?{params}"

        try:
            with urlopen(endpoint, timeout=settings.wechat_open_platform_api_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="微信授权服务请求失败，请稍后重试",
            ) from exc
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="微信授权服务返回了无效响应",
            ) from exc

        openid = normalize_sensitive_value(payload.get("openid"))
        errcode = payload.get("errcode")
        errmsg = payload.get("errmsg")

        if openid is None:
            detail = "微信授权失败，请重新扫码"
            if errcode is not None:
                detail = f"微信授权失败: {errcode} {errmsg or ''}".strip()
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)

        return openid
