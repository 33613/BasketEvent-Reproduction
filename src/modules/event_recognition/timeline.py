"""把固定分析窗口中的事件整理到源视频全局时间轴。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.segmentation import VideoSegment


# 第一版边界是产品剪辑范围，不代表动作级精确标注。
EVENT_PADDING_SECONDS: dict[str, tuple[float, float]] = {
    "Missed Shot": (5.0, 2.0),
    "Made Shot": (5.0, 2.0),
    "Free Throw": (4.0, 3.0),
    "Foul": (3.0, 4.0),
    "Turnover": (4.0, 3.0),
    "Jump Ball": (4.0, 3.0),
    "Rebound": (4.0, 2.0),
    "steal": (4.0, 3.0),
    "block": (4.0, 3.0),
    "ast": (7.0, 2.0),
}
DEFAULT_PADDING_SECONDS = (3.0, 2.0)


@dataclass(frozen=True)
class EventCandidate:
    """表示某个分析窗口中的一条球员事件证据。"""

    candidate_id: str
    source_video_id: str
    window_id: str
    local_track_id: str
    global_track_id: str | None
    event: str
    confidence: float
    temporal_score: float
    local_start_seconds: float
    local_end_seconds: float
    global_start_seconds: float
    global_end_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """转换为可以写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class MergedEvent:
    """表示在重叠窗口中消重后的全局球员事件。"""

    event_id: str
    source_video_id: str
    event: str
    global_track_id: str | None
    confidence: float
    temporal_score: float
    evidence_start_seconds: float
    evidence_end_seconds: float
    candidate_ids: tuple[str, ...]
    track_references: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换为可以写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class MaterialDraft:
    """表示尚未导出 MP4 的最终素材时间范围。"""

    material_id: str
    source_video_id: str
    start_seconds: float
    end_seconds: float
    event_ids: tuple[str, ...]
    boundary_method: str
    pre_roll_seconds: float
    post_roll_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """转换为可以写入 JSON 的字典。"""
        value = asdict(self)
        value["duration_seconds"] = self.end_seconds - self.start_seconds
        return value


class EventTimelineService:
    """完成局部时间映射、事件消重和第一版素材边界生成。"""

    def collect_candidates(
        self,
        segments: Sequence[VideoSegment],
        prediction_reports: Mapping[str, Mapping[str, Any]],
    ) -> list[EventCandidate]:
        """把每个窗口的 PlayNet 时间证据转换到源视频时间轴。"""
        candidates: list[EventCandidate] = []
        for segment in segments:
            report = prediction_reports.get(segment.segment_id)
            if report is None:
                continue
            raw_events = report.get("temporal_events", [])
            if not isinstance(raw_events, list):
                raise ValueError(f"{segment.segment_id} 的 temporal_events 必须是列表")
            for event_index, value in enumerate(raw_events):
                if not isinstance(value, Mapping):
                    raise ValueError("每条 temporal event 必须是对象")
                event_name = str(value.get("event") or "blank")
                if event_name == "blank":
                    continue
                local_start = float(value.get("start_time") or 0.0)
                local_end = float(value.get("end_time") or 0.0)
                local_start = max(0.0, local_start)
                local_end = min(segment.duration_seconds, local_end)
                if local_end <= local_start:
                    raise ValueError(
                        f"{segment.segment_id} 的事件时间范围无效："
                        f"{local_start}～{local_end}"
                    )
                player_id = str(
                    value.get("source_track_id") or value.get("player_id") or "unknown"
                )
                global_start = segment.start_seconds + local_start
                global_end = segment.start_seconds + local_end
                confidence = float(value.get("confidence") or 0.0)
                temporal_score = float(value.get("temporal_score") or confidence)
                candidates.append(
                    EventCandidate(
                        candidate_id=f"{segment.segment_id}:event_{event_index:04d}",
                        source_video_id=segment.source_video_id,
                        window_id=segment.segment_id,
                        local_track_id=player_id,
                        global_track_id=(
                            str(value["global_track_id"])
                            if value.get("global_track_id") is not None
                            else None
                        ),
                        event=event_name,
                        confidence=confidence,
                        temporal_score=temporal_score,
                        local_start_seconds=local_start,
                        local_end_seconds=local_end,
                        global_start_seconds=global_start,
                        global_end_seconds=global_end,
                    )
                )
        return sorted(
            candidates,
            key=lambda item: (
                item.source_video_id,
                item.global_start_seconds,
                item.event,
                item.window_id,
                item.local_track_id,
            ),
        )

    @staticmethod
    def _can_merge(
        group: dict[str, Any],
        candidate: EventCandidate,
        maximum_gap_seconds: float,
    ) -> bool:
        """判断候选是否是某个全局事件在另一个窗口中的重复证据。"""
        if group["source_video_id"] != candidate.source_video_id:
            return False
        if group["event"] != candidate.event:
            return False
        if candidate.global_start_seconds > group["end"] + maximum_gap_seconds:
            return False
        if candidate.global_end_seconds < group["start"] - maximum_gap_seconds:
            return False

        # 同一窗口中的不同人物预测不能因为时间重叠而被合并。
        existing_track = group["tracks_by_window"].get(candidate.window_id)
        if existing_track is not None and existing_track != candidate.local_track_id:
            return False

        known_global_tracks = group["global_track_ids"]
        if (
            candidate.global_track_id is not None
            and known_global_tracks
            and candidate.global_track_id not in known_global_tracks
        ):
            return False
        return True

    def merge_candidates(
        self,
        candidates: Sequence[EventCandidate],
        maximum_gap_seconds: float = 0.5,
    ) -> list[MergedEvent]:
        """合并重叠窗口对同一事件产生的重复预测。"""
        if maximum_gap_seconds < 0:
            raise ValueError("maximum_gap_seconds 不能为负数")
        groups: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (item.source_video_id, item.global_start_seconds),
        ):
            selected: dict[str, Any] | None = None
            for group in reversed(groups):
                if self._can_merge(group, candidate, maximum_gap_seconds):
                    selected = group
                    break
            if selected is None:
                selected = {
                    "source_video_id": candidate.source_video_id,
                    "event": candidate.event,
                    "start": candidate.global_start_seconds,
                    "end": candidate.global_end_seconds,
                    "confidence": candidate.confidence,
                    "temporal_score": candidate.temporal_score,
                    "candidate_ids": [],
                    "track_references": [],
                    "tracks_by_window": {},
                    "global_track_ids": set(),
                }
                groups.append(selected)
            selected["start"] = min(selected["start"], candidate.global_start_seconds)
            selected["end"] = max(selected["end"], candidate.global_end_seconds)
            selected["confidence"] = max(selected["confidence"], candidate.confidence)
            selected["temporal_score"] = max(
                selected["temporal_score"], candidate.temporal_score
            )
            selected["candidate_ids"].append(candidate.candidate_id)
            selected["track_references"].append(
                f"{candidate.window_id}/{candidate.local_track_id}"
            )
            selected["tracks_by_window"][candidate.window_id] = candidate.local_track_id
            if candidate.global_track_id is not None:
                selected["global_track_ids"].add(candidate.global_track_id)

        merged: list[MergedEvent] = []
        for index, group in enumerate(groups):
            global_tracks = sorted(group["global_track_ids"])
            merged.append(
                MergedEvent(
                    event_id=f"{group['source_video_id']}:event_{index:05d}",
                    source_video_id=group["source_video_id"],
                    event=group["event"],
                    global_track_id=(
                        global_tracks[0] if len(global_tracks) == 1 else None
                    ),
                    confidence=group["confidence"],
                    temporal_score=group["temporal_score"],
                    evidence_start_seconds=group["start"],
                    evidence_end_seconds=group["end"],
                    candidate_ids=tuple(group["candidate_ids"]),
                    track_references=tuple(group["track_references"]),
                )
            )
        return merged

    def build_material_drafts(
        self,
        events: Sequence[MergedEvent],
        source_duration_seconds: float,
    ) -> list[MaterialDraft]:
        """按事件类型增加上下文，生成尚未导出的视频素材范围。"""
        if source_duration_seconds <= 0:
            raise ValueError("source_duration_seconds 必须为正数")
        drafts: list[MaterialDraft] = []
        for index, event in enumerate(events):
            pre_roll, post_roll = EVENT_PADDING_SECONDS.get(
                event.event, DEFAULT_PADDING_SECONDS
            )
            start = max(0.0, event.evidence_start_seconds - pre_roll)
            end = min(
                source_duration_seconds,
                event.evidence_end_seconds + post_roll,
            )
            drafts.append(
                MaterialDraft(
                    material_id=f"{event.source_video_id}:material_{index:05d}",
                    source_video_id=event.source_video_id,
                    start_seconds=start,
                    end_seconds=end,
                    event_ids=(event.event_id,),
                    boundary_method="event_evidence_with_type_padding",
                    pre_roll_seconds=pre_roll,
                    post_roll_seconds=post_roll,
                )
            )
        return drafts

    def build_timeline(
        self,
        segments: Sequence[VideoSegment],
        prediction_reports: Mapping[str, Mapping[str, Any]],
        source_duration_seconds: float,
        maximum_gap_seconds: float = 0.5,
    ) -> dict[str, Any]:
        """一次生成局部候选、全局事件和待剪素材范围。"""
        candidates = self.collect_candidates(segments, prediction_reports)
        events = self.merge_candidates(candidates, maximum_gap_seconds)
        drafts = self.build_material_drafts(events, source_duration_seconds)
        return {
            "schema_version": "basketevent_event_timeline.v1",
            "source_duration_seconds": source_duration_seconds,
            "localization_is_diagnostic": True,
            "material_boundaries_are_editing_ranges": True,
            "candidates": [value.to_dict() for value in candidates],
            "events": [value.to_dict() for value in events],
            "material_drafts": [value.to_dict() for value in drafts],
        }

    @staticmethod
    def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
        """写出全局事件时间线报告。"""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination
