from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.logger import get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.agents.notifier import Notifier
from app.models.alert_log import AlertLog
from app.models.monitor_log import MonitorLog
from app.models.user import User
from app.models.user_operation_log import UserOperationLog
from app.schemas.auth import (
    EmailLoginRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserProfile,
    _validate_email,
    _validate_phone,
)
from app.services.email_code_service import EmailCodeService
from app.utils.user_uid import generate_user_uid


logger = get_logger(__name__)


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.email_code_service = EmailCodeService()
        self.notifier = Notifier()

    def register(self, payload: RegisterRequest) -> UserProfile:
        self._ensure_unique_user_fields(payload.username, payload.email, payload.phone)

        user = User(
            uid=self._generate_unique_uid(),
            username=payload.username,
            password_hash=hash_password(payload.password),
            email=payload.email,
            phone=payload.phone,
            role="user",
        )
        self.db.add(user)
        self.db.flush()
        self._log_operation(user.id, "register", "Success")
        self._log_monitor_event(
            user_id=user.id,
            operation_type="register",
            response_status="Success",
            title="用户注册完成",
            summary=f"用户 {payload.username} 完成了注册。",
        )
        self.db.commit()
        self.db.refresh(user)
        return UserProfile.model_validate(user)

    def login(self, payload: LoginRequest) -> TokenResponse:
        user = self.db.scalar(select(User).where(User.username == payload.username))
        if user is None or not verify_password(payload.password, user.password_hash):
            self._log_monitor_event(
                user_id=user.id if user else None,
                operation_type="login",
                response_status="Rejected",
                title="登录尝试被拒绝",
                summary=f"用户名 {payload.username} 的登录请求被拒绝。",
                level="warning",
            )
            self.db.commit()
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

        user.last_login = datetime.utcnow()
        logger.info("Login succeeded for account: %s", user.username)
        self._log_operation(user.id, "login", "Success")
        self._log_monitor_event(
            user_id=user.id,
            operation_type="login",
            response_status="Success",
            title="用户登录成功",
            summary=f"用户 {user.username} 登录成功。",
        )
        self.db.commit()
        self.db.refresh(user)

        token = create_access_token(str(user.id), extra_claims={"username": user.username, "role": user.role})
        return TokenResponse(access_token=token, user=UserProfile.model_validate(user))

    def send_email_login_code(self, email: str) -> None:
        user = self.db.scalar(select(User).where(User.email_hash == User.build_email_hash(email)))
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="该邮箱未绑定账号")
        self.email_code_service.send_code(email=email, username=user.username)

    def email_login(self, payload: EmailLoginRequest) -> TokenResponse:
        user = self.db.scalar(select(User).where(User.email_hash == User.build_email_hash(payload.email)))
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="该邮箱未绑定账号")
        if not self.email_code_service.verify_code(payload.email, payload.code):
            self._log_monitor_event(
                user_id=user.id,
                operation_type="email_login",
                response_status="Rejected",
                title="邮箱登录被拒绝",
                summary=f"用户 {user.username} 的邮箱登录被拒绝。",
                level="warning",
            )
            self.db.commit()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="验证码无效或已过期")

        user.last_login = datetime.utcnow()
        logger.info("Email login succeeded for account: %s", user.username)
        self._log_operation(user.id, "email_login", "Success")
        self._log_monitor_event(
            user_id=user.id,
            operation_type="email_login",
            response_status="Success",
            title="邮箱登录成功",
            summary=f"用户 {user.username} 通过邮箱验证码登录成功。",
        )
        self.db.commit()
        self.db.refresh(user)

        token = create_access_token(str(user.id), extra_claims={"username": user.username, "role": user.role})
        return TokenResponse(access_token=token, user=UserProfile.model_validate(user))

    def get_profile(self, user_id: int) -> UserProfile:
        user = self.db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return UserProfile.model_validate(user)

    def update_profile(self, user_id: int, payload: UpdateProfileRequest) -> UserProfile:
        user = self.db.get(User, user_id)
        if user is None:
            alert_summary = f"用户 ID {user_id} 修改个人资料失败，原因：用户不存在。"
            self._log_monitor_event(
                user_id=user_id,
                operation_type="update_profile",
                response_status="Failed",
                title="个人资料修改失败",
                summary=alert_summary,
                level="warning",
            )
            self._log_alert_event(
                level="warning",
                source="auth",
                event_type="update_profile",
                title="个人资料修改失败",
                summary=alert_summary,
                root_cause="用户资料修改请求对应的账号不存在。",
                impact_scope="本次用户资料修改未生效。",
                suggested_action="请重新登录后再尝试修改资料，或联系管理员检查账号状态。",
            )
            self.db.commit()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

        try:
            validated_username = self._validate_profile_username(payload.username)
            self._ensure_profile_contact_lengths(payload.email, payload.phone)
            validated_email = _validate_email(payload.email)
            validated_phone = _validate_phone(payload.phone)
            self._ensure_unique_user_fields(
                validated_username,
                validated_email,
                validated_phone,
                exclude_user_id=user.id,
            )
        except ValueError as exc:
            self._reject_profile_update(user, str(exc), status.HTTP_400_BAD_REQUEST)
        except HTTPException as exc:
            self._reject_profile_update(user, str(exc.detail), exc.status_code)

        old_username = user.username
        old_email = user.email
        old_phone = user.phone
        account_uid = user.uid
        changed_fields = self._build_profile_changed_fields(
            old_username=old_username,
            old_email=old_email,
            old_phone=old_phone,
            new_username=validated_username,
            new_email=validated_email,
            new_phone=validated_phone,
        )
        change_content = "、".join(changed_fields) if changed_fields else "无字段变化"

        user.username = validated_username
        user.email = validated_email
        user.phone = validated_phone
        logger.info("Profile updated for UID: %s; changed fields: %s", account_uid, change_content)
        self._log_operation(user.id, "update_profile", "Success")
        self._log_monitor_event(
            user_id=user.id,
            operation_type="update_profile",
            response_status="Success",
            title="个人资料已更新",
            summary=f"用户 UID {account_uid} 更新了个人资料，修改内容：{change_content}。",
        )
        self.db.commit()
        self.db.refresh(user)
        return UserProfile.model_validate(user)

    def _reject_profile_update(self, user: User, reason: str, status_code: int) -> None:
        alert_summary = f"用户 UID {user.uid} 修改个人资料失败，原因：{reason}。"
        logger.warning("Profile update failed for UID: %s; reason: %s", user.uid, reason)
        self._log_operation(user.id, "update_profile", "Failed")
        self._log_monitor_event(
            user_id=user.id,
            operation_type="update_profile",
            response_status="Failed",
            title="个人资料修改失败",
            summary=alert_summary,
            level="warning",
        )
        self._log_alert_event(
            level="warning",
            source="auth",
            event_type="update_profile",
            title="个人资料修改失败",
            summary=alert_summary,
            root_cause=f"资料字段校验未通过：{reason}。",
            impact_scope="本次用户资料修改未生效，账号原资料保持不变。",
            suggested_action="请调整用户名、邮箱或手机号后重新保存。",
        )
        self.db.commit()
        raise HTTPException(status_code=status_code, detail=reason)

    @staticmethod
    def _validate_profile_username(value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3 or len(normalized) > 64:
            raise ValueError("用户名长度需为 3-64 个字符")
        return normalized

    @staticmethod
    def _ensure_profile_contact_lengths(email: str | None, phone: str | None) -> None:
        if email is not None and len(email.strip()) > 128:
            raise ValueError("邮箱长度不能超过 128 个字符")
        if phone is not None and len(phone.strip()) > 20:
            raise ValueError("手机号长度不能超过 20 个字符")

    def _build_profile_changed_fields(
        self,
        *,
        old_username: str,
        old_email: str | None,
        old_phone: str | None,
        new_username: str,
        new_email: str | None,
        new_phone: str | None,
    ) -> list[str]:
        fields = [
            ("用户名", old_username, new_username),
            ("邮箱", old_email, new_email),
            ("手机号", old_phone, new_phone),
        ]
        return [
            label
            for label, old_value, new_value in fields
            if self._normalize_profile_log_value(old_value) != self._normalize_profile_log_value(new_value)
        ]

    @staticmethod
    def _normalize_profile_log_value(value: str | None) -> str:
        return (value or "").strip()

    def _generate_unique_uid(self) -> str:
        while True:
            uid = generate_user_uid()
            if self.db.scalar(select(User.id).where(User.uid == uid)) is None:
                return uid

    def _ensure_unique_user_fields(
        self,
        username: str,
        email: str | None,
        phone: str | None,
        exclude_user_id: int | None = None,
    ) -> None:
        conditions = [User.username == username]
        email_hash = User.build_email_hash(email)
        phone_hash = User.build_phone_hash(phone)
        if email_hash:
            conditions.append(User.email_hash == email_hash)
        if phone_hash:
            conditions.append(User.phone_hash == phone_hash)

        statement = select(User).where(or_(*conditions))
        if exclude_user_id is not None:
            statement = statement.where(User.id != exclude_user_id)

        existing_user = self.db.scalar(statement)
        if existing_user is None:
            return

        if existing_user.username == username:
            detail = "用户名已存在"
        elif email_hash and existing_user.email_hash == email_hash:
            detail = "邮箱已存在"
        else:
            detail = "手机号已存在"

        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    def _log_operation(self, user_id: int, operation_type: str, response_status: str) -> None:
        self.db.add(
            UserOperationLog(
                user_id=user_id,
                operation_type=operation_type,
                response_status=response_status,
            )
        )

    def _log_monitor_event(
        self,
        *,
        user_id: int | None,
        operation_type: str,
        response_status: str,
        title: str,
        summary: str,
        level: str = "info",
    ) -> None:
        self.db.add(
            MonitorLog(
                category="user_operation",
                source="auth",
                event_type=operation_type,
                level=level,
                title=title,
                summary=summary,
                status=response_status,
                user_id=user_id,
            )
        )
        self.notifier.notify_monitor_log(
            {
                "level": level,
                "source": "auth",
                "event_type": operation_type,
                "title": title,
                "summary": summary,
                "status": response_status,
            }
        )

    def _log_alert_event(
        self,
        *,
        level: str,
        source: str,
        event_type: str,
        title: str,
        summary: str,
        root_cause: str,
        impact_scope: str,
        suggested_action: str,
    ) -> None:
        self.db.add(
            AlertLog(
                level=level,
                source=source,
                event_type=event_type,
                title=title,
                summary=summary,
                root_cause=root_cause,
                impact_scope=impact_scope,
                suggested_action=suggested_action,
            )
        )
