from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import EmailCodeRequest, EmailLoginRequest, LoginRequest, RegisterRequest, TokenResponse, UpdateProfileRequest, UserProfile
from app.services.auth_service import AuthService

router = APIRouter()


@router.post(
    "/register",
    response_model=UserProfile,
    status_code=201,
    summary="注册账号",
    description="使用用户名和密码注册系统账号，可选绑定邮箱和手机号。",
    responses={
        201: {"description": "注册成功"},
        400: {"description": "用户名、邮箱或手机号已存在"},
    },
)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> UserProfile:
    service = AuthService(db)
    return service.register(payload)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="账号密码登录",
    description="使用用户名和密码登录，返回 Bearer Token 与当前用户资料。",
    responses={
        200: {"description": "登录成功"},
        401: {"description": "用户名或密码错误"},
    },
)
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    service = AuthService(db)
    return service.login(payload)


@router.post(
    "/email-code",
    status_code=204,
    summary="发送邮箱验证码",
    description="向已绑定邮箱发送一次性登录验证码。",
    responses={
        204: {"description": "验证码发送成功"},
        404: {"description": "邮箱未绑定账号"},
        429: {"description": "发送过于频繁"},
    },
)
async def send_email_code(payload: EmailCodeRequest, db: Session = Depends(get_db)) -> None:
    service = AuthService(db)
    service.send_email_login_code(payload.email)


@router.post(
    "/email-login",
    response_model=TokenResponse,
    summary="邮箱验证码登录",
    description="使用绑定邮箱和验证码登录，返回 Bearer Token。",
    responses={
        200: {"description": "登录成功"},
        400: {"description": "验证码错误或已过期"},
        404: {"description": "邮箱未绑定账号"},
    },
)
async def email_login(payload: EmailLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    service = AuthService(db)
    return service.email_login(payload)


@router.get(
    "/me",
    response_model=UserProfile,
    summary="获取当前用户资料",
    description="根据 Bearer Token 返回当前登录用户的基础信息。",
    responses={200: {"description": "查询成功"}, 401: {"description": "未登录或令牌失效"}},
)
async def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UserProfile:
    service = AuthService(db)
    return service.get_profile(current_user.id)


@router.put(
    "/profile",
    response_model=UserProfile,
    summary="更新用户资料",
    description="更新用户名、邮箱和手机号，敏感字段将进行加密存储。",
    responses={
        200: {"description": "更新成功"},
        400: {"description": "用户名、邮箱或手机号冲突"},
        401: {"description": "未登录或令牌失效"},
    },
)
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfile:
    service = AuthService(db)
    return service.update_profile(current_user.id, payload)
