import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus           # HEAD 保留，用于 SQLAlchemy URL 编码

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# HEAD 定义：明确指定 .env 文件位置（项目根目录）
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# wangxiaoyan 定义：用于构建默认模型目录
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    # ===== 合并后的 model_config =====
    # 使用 HEAD 的 ENV_FILE 路径，同时保留 wangxiaoyan 的 extra="ignore"
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ---------- 基础应用配置（HEAD） ----------
    app_name: str = Field(default="Car Vision Agent System")
    app_env: str = Field(default="development")
    api_v1_prefix: str = Field(default="/api/v1")
    secret_key: str = Field(default="change-me")
    access_token_expire_minutes: int = Field(default=120)

    # ---------- 数据库配置（HEAD） ----------
    database_url: str | None = Field(default=None)
    mysql_host: str = Field(default="127.0.0.1")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="bank")
    mysql_password: str = Field(default="")
    mysql_database: str = Field(default="car_vision_agent_system")
    mysql_charset: str = Field(default="utf8mb4")

    redis_url: str = Field(default="redis://localhost:6379/0")

    # ---------- LLM 配置（HEAD） ----------
    llm_provider: str = Field(default="openai-compatible")
    llm_api_base: str = Field(default="")
    llm_api_key: str = Field(default="")

    # ---------- 邮件验证码配置 ----------
    smtp_host: str = Field(default="smtp.163.com")
    smtp_port: int = Field(default=465)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_sender_name: str = Field(default="Car Vision System")
    smtp_use_ssl: bool = Field(default=True)
    email_code_expire_minutes: int = Field(default=5)
    email_code_cooldown_seconds: int = Field(default=60)

    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    hyperlpr_detect_level: str = Field(default="high")
    hyperlpr_home_dir: str = Field(default="runtime")
    plate_confidence_threshold: float = Field(default=0.5)
    plate_history_limit: int = Field(default=50)
    plate_save_uploads: bool = Field(default=False)
    plate_upload_dir: str = Field(default="uploads/plate")

    # ---------- 模型路径配置（wangxiaoyan） ----------
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
