from datetime import datetime

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.models.user_operation_log import UserOperationLog
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserProfile


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def register(self, payload: RegisterRequest) -> UserProfile:
        self._ensure_unique_user_fields(payload.username, payload.email, payload.phone)

        user = User(
            username=payload.username,
            password_hash=hash_password(payload.password),
            email=payload.email,
            phone=payload.phone,
            role="user",
        )
        self.db.add(user)
        self.db.flush()
        self._log_operation(user.id, "register", "Success")
        self.db.commit()
        self.db.refresh(user)
        return UserProfile.model_validate(user)

    def login(self, payload: LoginRequest) -> TokenResponse:
        user = self.db.scalar(select(User).where(User.username == payload.username))
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

        user.last_login = datetime.utcnow()
        self._log_operation(user.id, "login", "Success")
        self.db.commit()
        self.db.refresh(user)

        token = create_access_token(str(user.id), extra_claims={"username": user.username, "role": user.role})
        return TokenResponse(access_token=token, user=UserProfile.model_validate(user))

    def get_profile(self, user_id: int) -> UserProfile:
        user = self.db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        return UserProfile.model_validate(user)

    def _ensure_unique_user_fields(self, username: str, email: str | None, phone: str | None) -> None:
        conditions = [User.username == username]
        if email:
            conditions.append(User.email == email)
        if phone:
            conditions.append(User.phone == phone)

        existing_user = self.db.scalar(select(User).where(or_(*conditions)))
        if existing_user is None:
            return

        if existing_user.username == username:
            detail = "用户名已存在"
        elif email and existing_user.email == email:
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
