from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_current_user
from app.models_infer.errors import PlateInferenceError
from app.models.user import User
from app.schemas.plate import PlateRecognitionResponse, PlateRecordSummary
from app.services.plate_service import PlateService

router = APIRouter()
service = PlateService()


@router.post(
    "/image",
    response_model=PlateRecognitionResponse,
    summary="上传图片进行车牌识别",
    description="接收单张道路场景图片，返回检测到的车牌号码、颜色、置信度和检测框。",
    responses={
        200: {"description": "识别成功"},
        400: {"description": "图片格式不合法或无法解析"},
        401: {"description": "未登录或令牌失效"},
        503: {"description": "推理服务暂时不可用"},
    },
)
async def recognize_plate_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> PlateRecognitionResponse:
    image_bytes = await file.read()
    try:
        return await service.recognize_image_bytes_async(
            image_bytes,
            file.filename or "unknown.jpg",
            save_history=True,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PlateInferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/history",
    response_model=list[PlateRecordSummary],
    summary="查询车牌识别历史",
    description="返回当前登录用户的历史车牌识别记录。",
    responses={200: {"description": "查询成功"}, 401: {"description": "未登录或令牌失效"}},
)
async def get_plate_history(current_user: User = Depends(get_current_user)) -> list[PlateRecordSummary]:
    return service.list_history(current_user.id)
