from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass
class CurrentUser:
    id: int
    username: str


async def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )
    return CurrentUser(id=1, username="demo_admin")
