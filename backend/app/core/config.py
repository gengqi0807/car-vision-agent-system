import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Car Vision Agent System")
    app_env: str = Field(default="development")
    api_v1_prefix: str = Field(default="/api/v1")
    secret_key: str = Field(default="change-me")
    access_token_expire_minutes: int = Field(default=120)
    openapi_description: str = Field(
        default=(
            "智能车载视觉感知与交互系统后端 API，覆盖用户认证、车牌识别、"
            "交警手势识别、车主手势控车与告警监控等能力。"
        )
    )
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    openapi_url: str = Field(default="/openapi.json")
    api_contact_name: str = Field(default="Car Vision Team")
    api_contact_email: str = Field(default="team@example.com")

    database_url: str | None = Field(default=None)
    mysql_host: str = Field(default="127.0.0.1")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="bank")
    mysql_password: str = Field(default="")
    mysql_database: str = Field(default="car_vision_agent_system")
    mysql_charset: str = Field(default="utf8mb4")

    redis_url: str = Field(default="redis://localhost:6379/0")

    llm_provider: str = Field(default="openai-compatible")
    llm_api_base: str = Field(default="")
    llm_api_key: str = Field(default="")
    llm_request_timeout_seconds: float = Field(default=10.0)

    data_encryption_key: str = Field(
        default="",
        description="Base64-encoded AES key for encrypting sensitive fields.",
    )
    data_hash_key: str = Field(
        default="",
        description="Optional secret used for HMAC-based lookup hashes.",
    )

    smtp_host: str = Field(default="smtp.163.com")
    smtp_port: int = Field(default=465)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_sender_name: str = Field(default="Car Vision System")
    smtp_use_ssl: bool = Field(default=True)
    email_code_expire_minutes: int = Field(default=5)
    email_code_cooldown_seconds: int = Field(default=60)

    alert_email_recipients: list[str] = Field(default_factory=list)
    alert_webhook_url: str = Field(default="")
    alert_webhook_timeout_seconds: float = Field(default=8.0)
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    alert_consecutive_failures_threshold: int = Field(default=3)
    alert_low_confidence_threshold: float = Field(default=0.6)
    alert_low_confidence_window_size: int = Field(default=3)
    alert_replay_window_minutes: int = Field(default=20)

    hyperlpr_detect_level: str = Field(default="high")
    hyperlpr_home_dir: str = Field(default="runtime")
    plate_detector_enabled: bool = Field(default=True)
    plate_detector_fallback_to_full_frame: bool = Field(default=True)
    plate_detector_crop_padding_x: float = Field(default=0.16)
    plate_detector_crop_padding_y: float = Field(default=0.30)
    plate_detector_max_candidates: int = Field(default=12)
    plate_detector_class_names: list[str] = Field(
        default_factory=lambda: ["plate", "license-plate", "licence-plate", "lp"]
    )
    plate_vehicle_detector_fallback_enabled: bool = Field(default=True)
    plate_vehicle_class_names: list[str] = Field(default_factory=lambda: ["car", "truck", "bus"])
    plate_vehicle_crop_padding_x: float = Field(default=0.08)
    plate_vehicle_crop_padding_y: float = Field(default=0.05)
    plate_vehicle_plate_top_ratio: float = Field(default=0.52)
    plate_vehicle_plate_height_ratio: float = Field(default=0.24)
    plate_vehicle_plate_width_ratio: float = Field(default=0.60)
    plate_yolo_model_path: str = Field(default="weights/open_traffic_flow_best.pt")
    plate_yolo_confidence: float = Field(default=0.18)
    plate_yolo_imgsz: int = Field(default=960)
    plate_yolo_max_det: int = Field(default=20)
    plate_yolo_device: str = Field(default="cpu")
    plate_max_image_side: int = Field(default=1600)
    plate_confidence_threshold: float = Field(default=0.5)
    plate_inference_timeout_seconds: float = Field(default=60.0)
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
    plate_video_process_every_n_frames: int = Field(default=10)
    plate_video_recognition_max_side: int = Field(default=1280)
    plate_video_detector_full_frame_fallback: bool = Field(default=False)
    plate_open_traffic_flow_lpr_enabled: bool = Field(default=True)
    plate_open_traffic_flow_lpr_model_path: str = Field(default="weights/Final_LPRNet_model.pth")
    plate_open_traffic_flow_lpr_device: str = Field(default="cpu")
    plate_open_traffic_flow_lpr_confidence_threshold: float = Field(default=0.55)
    plate_save_uploads: bool = Field(default=False)
    plate_upload_dir: str = Field(default="uploads/plate")
    plate_push_ffmpeg_bin: str = Field(default="ffmpeg")
    plate_push_stream_name: str = Field(default="plate-live")
    plate_push_rtsp_base_url: str = Field(default="rtsp://127.0.0.1:8554")
    plate_push_playback_base_url: str = Field(default="http://127.0.0.1:8889")
    plate_push_fps: int = Field(default=25)
    plate_push_bitrate: str = Field(default="2M")

    models_dir: str = Field(
        default=str(BACKEND_DIR / "models"),
        description="Directory containing downloaded model files.",
    )
    hand_landmarker_model: str = Field(
        default="",
        description="Optional override for hand_landmarker.task.",
    )
    pose_landmarker_model: str = Field(
        default="",
        description="Optional override for pose_landmarker_lite.task.",
    )
    gesture_classifier_model: str = Field(
        default="gesture_classifier_svm.joblib",
        description="Relative or absolute path to the owner-gesture classifier model.",
    )
    num_hands: int = Field(default=2)
    min_hand_detection_confidence: float = Field(default=0.5)
    min_hand_presence_confidence: float = Field(default=0.5)
    min_hand_tracking_confidence: float = Field(default=0.5)

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            if self.database_url.startswith("sqlite") and self.database_url != "sqlite:///:memory:":
                prefix, _, database_path = self.database_url.partition(":///")
                if database_path.startswith("file:"):
                    return self.database_url
                if database_path and not Path(database_path).is_absolute():
                    resolved_path = (ENV_FILE.parent / database_path).resolve()
                    return f"{prefix}:///{resolved_path.as_posix()}"
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
        return self.hand_landmarker_model or os.path.join(self.models_dir, "hand_landmarker.task")

    @property
    def resolved_pose_model_path(self) -> str:
        return self.pose_landmarker_model or os.path.join(self.models_dir, "pose_landmarker_lite.task")

    @property
    def resolved_gesture_classifier_model_path(self) -> str:
        classifier_path = self.gesture_classifier_model
        if os.path.isabs(classifier_path):
            return classifier_path
        return os.path.join(self.models_dir, classifier_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
