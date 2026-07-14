from fastapi import APIRouter

from app.api.v1.alert import router as alert_router
from app.api.v1.auth import router as auth_router
from app.api.v1.custom_gesture import router as custom_gesture_router
from app.api.v1.owner_gesture import router as owner_gesture_router
from app.api.v1.plate import router as plate_router
from app.api.v1.police_gesture import router as police_gesture_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(plate_router, prefix="/plate", tags=["plate"])
api_router.include_router(police_gesture_router, prefix="/police-gesture", tags=["police-gesture"])
api_router.include_router(owner_gesture_router, prefix="/owner-gesture", tags=["owner-gesture"])
api_router.include_router(custom_gesture_router, prefix="/owner-gesture", tags=["custom-gesture"])
api_router.include_router(alert_router, prefix="/alerts", tags=["alerts"])
