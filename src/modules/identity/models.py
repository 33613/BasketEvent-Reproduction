"""定义 Identity 各阶段之间传递的稳定数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TrackCrop:
    """表示从单视频的一条人物轨迹中截取的一张图像。"""

    track_id: str
    image_index: int
    frame_index: int
    image: Any = field(repr=False, compare=False)
    sharpness: float = 0.0


@dataclass(frozen=True)
class IdentityObservation:
    """表示视觉模型对一张轨迹截图给出的属性观察。"""

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
class IdentityEvidence:
    """表示一个证据提供器对单条轨迹产生的全部证据。

    Qwen 主要填写 ``observations``；人物 ReID 主要填写 ``embedding``。
    融合器只依赖这个统一结构，不依赖具体模型类。
    """

    source: str
    track_id: str
    confidence: float
    observations: tuple[IdentityObservation, ...] = ()
    embedding: tuple[float, ...] | None = None
    frame_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class IdentityCandidate:
    """表示由多帧属性观察共同支持的一个球衣身份候选。"""

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
class IdentityProfile:
    """表示人物身份库中一个可持续补充证据的人物档案。"""

    participant_id: str
    display_name: str | None = None
    jersey_color: str | None = None
    jersey_number: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class GalleryMatch:
    """表示人物身份库返回的一个候选人物。"""

    participant_id: str
    score: float
    method: str
    profile: IdentityProfile


@dataclass(frozen=True)
class IdentityDecision:
    """表示融合多种证据后对一条轨迹作出的身份结论。

    ``accepted`` 只表示轨迹是否继续交给事件模型。单独的 Qwen 失败不会把
    它设为 ``False``；证据不足时使用匿名人物，避免身份识别阻塞事件推理。
    """

    track_id: str
    status: str
    accepted: bool
    is_on_court_player: bool
    participant_id: str | None = None
    jersey_color: str | None = None
    jersey_number: str | None = None
    player_name: str | None = None
    candidates: tuple[IdentityCandidate, ...] = ()
    gallery_matches: tuple[GalleryMatch, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可审计的 JSON 字典。"""
        return {
            "track_id": self.track_id,
            "status": self.status,
            "accepted": self.accepted,
            "is_on_court_player": self.is_on_court_player,
            "participant_id": self.participant_id,
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "player_name": self.player_name,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "gallery_matches": [
                {
                    "participant_id": match.participant_id,
                    "score": match.score,
                    "method": match.method,
                }
                for match in self.gallery_matches
            ],
            "evidence_sources": list(self.evidence_sources),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrackAssociation:
    """记录某个视频片段中的轨迹与跨片段人物编号之间的关系。"""

    clip_id: str
    track_id: str
    participant_id: str
    method: str
    confidence: float


@dataclass(frozen=True)
class IdentityProcessingResult:
    """保存 IdentityService 处理一个视频片段后的完整结果。"""

    clip_id: str
    decisions: tuple[IdentityDecision, ...]
    evidence_by_track: Mapping[str, tuple[IdentityEvidence, ...]]
    selected_ball_id: str | None
    ball_review: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """转换为适合诊断报告的字典。"""
        return {
            "clip_id": self.clip_id,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "evidence": {
                track_id: [
                    {
                        "source": item.source,
                        "confidence": item.confidence,
                        "frame_indices": list(item.frame_indices),
                        "has_embedding": item.embedding is not None,
                        "observations": [asdict(value) for value in item.observations],
                    }
                    for item in values
                ]
                for track_id, values in self.evidence_by_track.items()
            },
            "selected_ball_id": self.selected_ball_id,
            "ball_review": dict(self.ball_review),
        }


@dataclass(frozen=True)
class BallCandidate:
    """保存一个篮球轨迹候选及其几何统计和截图。"""

    track_id: str
    crops: tuple[TrackCrop, ...]
    statistics: Mapping[str, float | int | str]
