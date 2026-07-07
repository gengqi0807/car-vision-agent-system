from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import EmailCodeRequest, EmailLoginRequest, LoginRequest, RegisterRequest, TokenResponse, UpdateProfileRequest, UserProfile
from app.services.auth_service import AuthService

router = APIRouter()


@router.post("/register", response_model=UserProfile, status_code=201)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> UserProfile:
    service = AuthService(db)
    return service.register(payload)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    service = AuthService(db)
    return service.login(payload)


@router.post("/email-code", status_code=204)
async def send_email_code(payload: EmailCodeRequest, db: Session = Depends(get_db)) -> None:
    service = AuthService(db)
    service.send_email_login_code(payload.email)


@router.post("/email-login", response_model=TokenResponse)
async def email_login(payload: EmailLoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    service = AuthService(db)
    return service.email_login(payload)


@router.get("/me", response_model=UserProfile)
async def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> UserProfile:
    service = AuthService(db)
    return service.get_profile(current_user.id)


@router.put("/profile", response_model=UserProfile)
async def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfile:
    service = AuthService(db)
    return service.update_profile(current_user.id, payload)
