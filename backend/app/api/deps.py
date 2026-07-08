from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.user import User
from app.services.monitor_service import MonitorService

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        await MonitorService(db).capture_event(
            category="security",
            source="auth",
            event_type="unauthorized_access",
            title="缺少 Bearer 令牌",
            summary="受保护接口在未携带 Bearer 令牌的情况下被访问。",
            level="warning",
            status="rejected",
            details={"reason": "missing_token"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer 令牌",
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        await MonitorService(db).capture_event(
            category="security",
            source="auth",
            event_type="unauthorized_access",
            title="Bearer 令牌无效",
            summary="受保护接口被无效或已过期的令牌访问。",
            level="warning",
            status="rejected",
            details={"reason": str(exc)},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    user_id = payload.get("sub")
    if user_id is None:
        await MonitorService(db).capture_event(
            category="security",
            source="auth",
            event_type="unauthorized_access",
            title="令牌载荷无效",
            summary="受保护接口被缺少用户主体信息的令牌访问。",
            level="warning",
            status="rejected",
            details={"reason": "missing_subject"},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌载荷无效")

    user = db.get(User, int(user_id))
    if user is None:
        await MonitorService(db).capture_event(
            category="security",
            source="auth",
            event_type="unauthorized_access",
            title="认证用户不存在",
            summary="受保护接口被引用了不存在用户的令牌访问。",
            level="warning",
            status="rejected",
            details={"reason": "user_not_found", "user_id": user_id},
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")

    return user
