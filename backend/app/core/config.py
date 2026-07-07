from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Car Vision Agent System")
    app_env: str = Field(default="development")
    api_v1_prefix: str = Field(default="/api/v1")
    secret_key: str = Field(default="change-me")
    access_token_expire_minutes: int = Field(default=120)
    database_url: str = Field(default="sqlite:///./car_vision.db")
    redis_url: str = Field(default="redis://localhost:6379/0")
    llm_provider: str = Field(default="openai-compatible")
    llm_api_base: str = Field(default="")
    llm_api_key: str = Field(default="")
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    hyperlpr_detect_level: str = Field(default="low")
    hyperlpr_home_dir: str = Field(default="runtime")
    plate_max_image_side: int = Field(default=1600)
    plate_confidence_threshold: float = Field(default=0.5)
    plate_stream_recognition_max_side: int = Field(default=0)
    plate_stream_max_side: int = Field(default=960)
    plate_stream_max_fps: int = Field(default=8)
    plate_stream_process_every_n_frames: int = Field(default=8)
    plate_stream_jpeg_quality: int = Field(default=80)
    plate_stream_detection_hold_seconds: float = Field(default=0.8)
    plate_stream_tracking_max_misses: int = Field(default=6)
    plate_stream_tracking_match_threshold: float = Field(default=0.45)
    plate_stream_tracking_search_expand: float = Field(default=2.4)
    plate_stream_tracking_template_update_alpha: float = Field(default=0.18)
    plate_stream_history_interval_seconds: int = Field(default=3)
    plate_history_limit: int = Field(default=50)
    plate_save_uploads: bool = Field(default=False)
    plate_upload_dir: str = Field(default="uploads/plate")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
