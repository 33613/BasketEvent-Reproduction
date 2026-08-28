"""定义最小身份处理流程使用的数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackCrop:
    """表示从一条 SAM3 人物轨迹中截取的一张图像。"""

    track_id: str
    image_index: int
    frame_index: int
    image: Any = field(repr=False, compare=False)
    sharpness: float = 0.0


@dataclass(frozen=True)
class IdentityObservation:
    """表示 Qwen 对一张轨迹截图给出的可审计观察。"""

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
    """表示同一条轨迹中由若干帧支持的球衣属性候选。"""

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
class IdentityDecision:
    """表示固定规则对一条人物轨迹作出的身份结论。

    ``accepted`` 在当前最小方案中始终为真。Qwen 只负责提供观察，不能
    删除 SAM3 轨迹；无法确定身份时使用片段内稳定的匿名编号。
    """

    track_id: str
    status: str
    accepted: bool
    participant_id: str
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
            "is_on_court_player": True,
            "participant_id": self.participant_id,
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "player_name": self.player_name,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IdentityProcessingResult:
    """保存处理一个视频片段后的身份结果与诊断信息。"""

    game_id: str
    clip_id: str
    decisions: tuple[IdentityDecision, ...]
    observations_by_track: Mapping[str, tuple[IdentityObservation, ...]]
    errors_by_track: Mapping[str, str]
    selected_ball_id: str | None
    ball_review: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为适合诊断报告的字典。"""
        return {
            "game_id": self.game_id,
            "clip_id": self.clip_id,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "observations": {
                track_id: [asdict(item) for item in values]
                for track_id, values in self.observations_by_track.items()
            },
            "errors": dict(self.errors_by_track),
            "selected_ball_id": self.selected_ball_id,
            "ball_review": dict(self.ball_review),
        }


@dataclass(frozen=True)
class BallCandidate:
    """保存一个篮球轨迹候选及其几何统计和截图。"""

    track_id: str
    crops: tuple[TrackCrop, ...]
    statistics: Mapping[str, float | int | str]
