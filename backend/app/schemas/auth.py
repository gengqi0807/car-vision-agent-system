from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


class EmailLoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "demo@example.com", "code": "123456"}}
    )

    email: str = Field(min_length=5, max_length=128)
    code: str = Field(min_length=4, max_length=6)


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

    username: str = Field(min_length=3, max_length=64)
    email: str | None = Field(default=None, max_length=128)
    phone: str | None = Field(default=None, max_length=20)
class UserProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
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
