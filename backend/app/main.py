from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401
from app.api.router import api_router
from app.core.config import settings
from app.core.database import init_database
from app.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=settings.openapi_description,
    lifespan=lifespan,
    contact={
        "name": settings.api_contact_name,
        "email": settings.api_contact_email,
    },
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    openapi_tags=[
        {"name": "auth", "description": "用户注册、登录、邮箱验证码登录与资料维护"},
        {"name": "plate", "description": "车牌识别上传与历史记录查询"},
        {"name": "police-gesture", "description": "交警手势识别接口"},
        {"name": "owner-gesture", "description": "车主手势控车识别接口"},
        {"name": "alerts", "description": "告警总览与时间线查询"},
        {"name": "system", "description": "系统存活与健康检查"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} backend is running"}
