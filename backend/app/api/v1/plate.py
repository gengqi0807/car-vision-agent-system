from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_current_user
from app.models_infer.errors import PlateInferenceError
from app.models.user import User
from app.schemas.plate import PlateRecognitionResponse, PlateRecordSummary
from app.services.plate_service import PlateService

router = APIRouter()
service = PlateService()


@router.post("/image", response_model=PlateRecognitionResponse)
async def recognize_plate_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> PlateRecognitionResponse:
    image_bytes = await file.read()
    try:
        return service.recognize_image_bytes(
            image_bytes,
            file.filename or "unknown.jpg",
            save_history=True,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/history", response_model=list[PlateRecordSummary])
async def get_plate_history(current_user: User = Depends(get_current_user)) -> list[PlateRecordSummary]:
    return service.list_history(current_user.id)
