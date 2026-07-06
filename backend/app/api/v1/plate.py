from fastapi import APIRouter, File, UploadFile

from app.schemas.plate import PlateRecognitionResponse, PlateRecordSummary
from app.services.plate_service import PlateService

router = APIRouter()
service = PlateService()


@router.post("/image", response_model=PlateRecognitionResponse)
async def recognize_plate_image(file: UploadFile = File(...)) -> PlateRecognitionResponse:
    return await service.recognize_image(file.filename or "unknown.jpg")


@router.get("/history", response_model=list[PlateRecordSummary])
async def get_plate_history() -> list[PlateRecordSummary]:
    return service.list_history()
