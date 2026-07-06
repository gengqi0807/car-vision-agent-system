from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
