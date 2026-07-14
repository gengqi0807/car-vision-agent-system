"""
自定义手势数据模型。
- CustomGesture: 手势元信息（名称、样本数、是否已训练）
- CustomGestureSample: 单个样本的关键点数据（21 点 JSON）
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, String, Text, Boolean
from sqlalchemy.orm import relationship

from app.core.database import Base


class CustomGesture(Base):
    __tablename__ = "custom_gestures"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="手势唯一名称（英文标识，如 peace / ok / rock）",
    )
    display_name = Column(
        String(64),
        nullable=False,
        default="",
        comment="手势显示名称（中文，如 比耶 / OK / 摇滚）",
    )
    description = Column(Text, default="", comment="可选的手势说明")
    sample_count = Column(Integer, default=0, comment="已采集样本数")
    is_trained = Column(Boolean, default=False, comment="是否已参与训练")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    samples = relationship(
        "CustomGestureSample",
        back_populates="gesture",
        cascade="all, delete-orphan",
    )


class CustomGestureSample(Base):
    __tablename__ = "custom_gesture_samples"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    gesture_id = Column(
        Integer,
        ForeignKey("custom_gestures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    keypoints = Column(
        JSON,
        nullable=False,
        comment="21 个手部关键点, 每个为 {x, y, z} 字典列表",
    )
    source_type = Column(
        String(32),
        default="upload",
        comment="样本来源: upload / camera / manual",
    )
    filename = Column(String(256), default="", comment="原始文件名（如有）")
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    gesture = relationship("CustomGesture", back_populates="samples")
