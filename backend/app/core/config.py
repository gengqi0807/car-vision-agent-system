import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus           # ← 保留 HEAD 的导入（用于 SQLAlchemy URL 编码）

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# HEAD 定义的环境变量文件位置
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# wangxiaoyan 定义的 backend 目录（用于构建模型路径）
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8")

    # ---------- 基础应用配置（HEAD 原有） ----------
    app_name: str = Field(default="Car Vision Agent System")
    app_env: str = Field(default="development")
    api_v1_prefix: str = Field(default="/api/v1")
    secret_key: str = Field(default="change-me")
    access_token_expire_minutes: int = Field(default=120)

    # ---------- 数据库配置（HEAD 原有） ----------
    database_url: str | None = Field(default=None)
    mysql_host: str = Field(default="127.0.0.1")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="bank")
    mysql_password: str = Field(default="")
    mysql_database: str = Field(default="car_vision_agent_system")
    mysql_charset: str = Field(default="utf8mb4")

    redis_url: str = Field(default="redis://localhost:6379/0")

    # ---------- LLM 配置（HEAD 原有） ----------
    llm_provider: str = Field(default="openai-compatible")
    llm_api_base: str = Field(default="")
    llm_api_key: str = Field(default="")

    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ---------- 模型路径配置（来自 wangxiaoyan） ----------
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

    # ---------- 属性方法 ----------
    @property
    def sqlalchemy_database_url(self) -> str:
        """HEAD 原有的 SQLAlchemy 数据库连接 URL（使用 pymysql）"""
        if self.database_url:
            return self.database_url

        username = quote_plus(self.mysql_user)
        password = quote_plus(self.mysql_password)
        return (
            f"mysql+pymysql://{username}:{password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset={self.mysql_charset}"
        )

    @property
    def resolved_hand_model_path(self) -> str:
        """wangxiaoyan 添加的模型路径解析"""
        return self.hand_landmarker_model or os.path.join(
            self.models_dir, "hand_landmarker.task"
        )

    @property
    def resolved_pose_model_path(self) -> str:
        """wangxiaoyan 添加的模型路径解析"""
        return self.pose_landmarker_model or os.path.join(
            self.models_dir, "pose_landmarker_lite.task"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()