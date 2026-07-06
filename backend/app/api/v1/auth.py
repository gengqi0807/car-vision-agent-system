from fastapi import APIRouter

from app.schemas.auth import LoginRequest, TokenResponse, UserProfile
from app.services.auth_service import AuthService

router = APIRouter()
service = AuthService()


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    return service.login(payload)


@router.get("/me", response_model=UserProfile)
async def get_profile() -> UserProfile:
    return service.get_demo_profile()
