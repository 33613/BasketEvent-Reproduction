"""定义身份识别阶段使用的中间数据结构。

这些结构把“取样、视觉观察、身份解析、跨片段聚类”之间的接口固定下来，
使各阶段可以独立替换实现，而不必修改上下游代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackCrop:
    """表示从一条轨迹中截取的一张球员图像。"""

    track_id: str
    image_index: int
    frame_index: int
    image: Any = field(repr=False, compare=False)
    sharpness: float = 0.0


@dataclass(frozen=True)
class IdentityObservation:
    """表示 Qwen 对单张轨迹截图给出的结构化观察。"""

    track_id: str
    image_index: int
    frame_index: int
    is_on_court_player: bool | None
    jersey_color: str | None
    jersey_number: str | None
    confidence: float
    evidence: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class IdentityCandidate:
    """表示同一轨迹内由多帧共同支持的一个球衣身份候选。"""

    jersey_color: str
    jersey_number: str
    support_count: int
    confidence_sum: float
    frames: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""
        return {
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "support_count": self.support_count,
            "confidence_sum": self.confidence_sum,
            "frames": list(self.frames),
        }


@dataclass(frozen=True)
class ResolvedIdentity:
    """表示一条轨迹经过时序聚合后的最终身份状态。"""

    track_id: str
    status: str
    accepted: bool
    is_on_court_player: bool
    jersey_color: str | None = None
    jersey_number: str | None = None
    player_name: str | None = None
    candidates: tuple[IdentityCandidate, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可审计的 JSON 字典。"""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "accepted": self.accepted,
            "is_on_court_player": self.is_on_court_player,
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "player_name": self.player_name,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


@dataclass
class IdentityCluster:
    """保存跨视频片段归并后的同一球员素材集合。"""

    cluster_id: str
    jersey_color: str | None
    jersey_number: str | None
    members: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为素材统计可直接使用的 JSON 字典。"""
        return {
            "cluster_id": self.cluster_id,
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "members": [
                {"clip_id": clip_id, "track_id": track_id}
                for clip_id, track_id in self.members
            ],
        }


@dataclass(frozen=True)
class BallCandidate:
    """保存一个篮球轨迹候选及其几何统计和截图。"""

    track_id: str
    crops: tuple[TrackCrop, ...]
    statistics: Mapping[str, float | int | str]
