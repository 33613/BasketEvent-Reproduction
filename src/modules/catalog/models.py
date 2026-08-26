"""定义素材目录模块对外使用的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EventTag:
    """表示 PlayNet 在一个素材片段中识别出的球员级事件。"""

    event: str
    confidence: float
    player_id: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True)
class ParticipantReference:
    """表示素材中一个可检索的人物身份。"""

    participant_id: str
    track_id: str
    jersey_color: str | None = None
    jersey_number: str | None = None
    player_name: str | None = None
    identity_status: str | None = None
    reid_cluster_id: str | None = None


@dataclass(frozen=True)
class CatalogItem:
    """表示一个已经完成前处理、可进入产品素材库的视频片段。"""

    material_id: str
    source_video_id: str
    segment_id: str
    video_path: Path
    start_seconds: float
    end_seconds: float
    processing_status: str
    events: tuple[EventTag, ...] = ()
    participants: tuple[ParticipantReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """返回素材在原视频时间轴上的持续时间。"""
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_dict(self) -> dict[str, Any]:
        """转换为数据库或 JSON 接口可直接接收的字典。"""
        value = asdict(self)
        value["video_path"] = str(self.video_path)
        value["duration_seconds"] = self.duration_seconds
        return value
