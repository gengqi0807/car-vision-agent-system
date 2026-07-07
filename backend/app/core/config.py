import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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

    # ---------- model paths ----------
    models_dir: str = Field(
        default=str(_BACKEND_DIR / "models"),
        description="Directory containing downloaded .task / .onnx model files",
    )
    hand_landmarker_model: str = Field(
        default="",
        description="Path to hand_landmarker.task; falls back to {models_dir}/hand_landmarker.task",
    )
    pose_landmarker_model: str = Field(
        default="",
        description="Path to pose_landmarker_lite.task; falls back to {models_dir}/pose_landmarker_lite.task",
    )

    @property
    def resolved_hand_model_path(self) -> str:
        return self.hand_landmarker_model or os.path.join(
            self.models_dir, "hand_landmarker.task"
        )

    @property
    def resolved_pose_model_path(self) -> str:
        return self.pose_landmarker_model or os.path.join(
            self.models_dir, "pose_landmarker_lite.task"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
