from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import models  # noqa: F401
from app.api.router import api_router
from app.api.v1.plate import service as plate_service
from app.core.config import settings
from app.core.database import Base, engine
from app.core.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
media_root = (Path(__file__).resolve().parents[1] / settings.plate_upload_dir).resolve()
media_root.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    logger.info("Starting %s in %s mode", settings.app_name, settings.app_env)
    try:
        logger.info("Warming up OCR and detection models...")
        plate_service.warmup_runtime(silent=True)
        logger.info("Model warmup finished.")
    except Exception:
        logger.warning("Model warmup failed during startup; runtime will fall back to lazy initialization.", exc_info=True)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)
app.mount("/media", StaticFiles(directory=str(media_root)), name="media")


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} backend is running"}
