from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, TokenResponse, UserProfile


class AuthService:
    def login(self, payload: LoginRequest) -> TokenResponse:
        token = create_access_token(payload.username)
        return TokenResponse(access_token=token)

    def get_demo_profile(self) -> UserProfile:
        return UserProfile(id=1, username="demo_admin", role="admin")
