from datetime import datetime
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
ALLOWED_EMAIL_SUFFIXES = (".com", ".cn", ".com.cn", ".net", ".org", ".edu", ".edu.cn", ".gov.cn")
PHONE_PATTERN = re.compile(r"^1[3-9]\d{9}$")


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_email(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    if not EMAIL_PATTERN.fullmatch(normalized) or not normalized.lower().endswith(ALLOWED_EMAIL_SUFFIXES):
        raise ValueError("邮箱格式不正确，仅支持 .com、.cn、.com.cn、.net、.org、.edu、.edu.cn、.gov.cn 后缀")
    return normalized


def _validate_phone(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None
    if not PHONE_PATTERN.fullmatch(normalized):
        raise ValueError("手机号格式不正确，请输入 11 位中国大陆手机号")
    return normalized


class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "demo_user",
                "password": "SecurePass123",
                "email": "demo@example.com",
                "phone": "13800138000",
            }
        }
    )
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    email: str | None = Field(default=None, max_length=128)
    phone: str | None = Field(default=None, max_length=20)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        return _validate_email(value)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str | None) -> str | None:
        return _validate_phone(value)


class LoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"username": "demo_user", "password": "SecurePass123"}}
    )
    username: str
    password: str


class EmailCodeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "demo@example.com"}}
    )

    email: str = Field(min_length=5, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = _validate_email(value)
        if normalized is None:
            raise ValueError("邮箱不能为空")
        return normalized


class EmailLoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "demo@example.com", "code": "123456"}}
    )

    email: str = Field(min_length=5, max_length=128)
    code: str = Field(min_length=4, max_length=6)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = _validate_email(value)
        if normalized is None:
            raise ValueError("邮箱不能为空")
        return normalized


class UpdateProfileRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "demo_user",
                "email": "demo@example.com",
                "phone": "13800138000",
            }
        }
    )

    username: str
    email: str | None = None
    phone: str | None = None


class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    uid: str
    username: str
    email: str | None = None
    phone: str | None = None
    role: str
    created_at: datetime
    last_login: datetime | None = None


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": 1,
                    "uid": "739281604512",
                    "username": "demo_user",
                    "email": "demo@example.com",
                    "phone": "13800138000",
                    "role": "user",
                    "created_at": "2026-07-08T10:00:00",
                    "last_login": "2026-07-08T12:00:00",
                },
            }
        }
    )
    access_token: str
    token_type: str = "bearer"
    user: UserProfile
