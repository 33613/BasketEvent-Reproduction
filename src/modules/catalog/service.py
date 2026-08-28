"""把模型输出整理成素材记录，并生成简单统计。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.catalog.models import (
    CatalogItem,
    EventTag,
    MaterialStatistics,
    ParticipantReference,
)


class CatalogService:
    """负责素材数据整理，不负责数据库读写。"""

    def build_final_material(
        self,
        *,
        material_id: str,
        source_video_id: str,
        video_path: str | Path,
        start_seconds: float,
        end_seconds: float,
        events: Sequence[Mapping[str, Any]],
        participants: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> CatalogItem:
        """把全局事件、最终素材路径和可选身份整理成数据库对象。"""
        if end_seconds <= start_seconds:
            raise ValueError("最终素材必须具有正数时长")
        event_tags = tuple(
            EventTag(
                event=str(value.get("event") or "unknown"),
                confidence=float(value.get("confidence") or 0.0),
                player_id=(
                    str(value["global_track_id"])
                    if value.get("global_track_id") is not None
                    else None
                ),
                start_seconds=(
                    float(value["evidence_start_seconds"])
                    if value.get("evidence_start_seconds") is not None
                    else None
                ),
                end_seconds=(
                    float(value["evidence_end_seconds"])
                    if value.get("evidence_end_seconds") is not None
                    else None
                ),
            )
            for value in events
            if str(value.get("event") or "blank") != "blank"
        )
        participant_references = tuple(
            ParticipantReference(
                participant_id=str(value["participant_id"]),
                track_id=str(value.get("track_id") or "unknown"),
                jersey_color=value.get("jersey_color"),
                jersey_number=(
                    str(value["jersey_number"])
                    if value.get("jersey_number") is not None
                    else None
                ),
                player_name=value.get("player_name"),
                identity_status=value.get("identity_status"),
            )
            for value in participants
            if value.get("participant_id") is not None
        )
        return CatalogItem(
            material_id=material_id,
            source_video_id=source_video_id,
            segment_id=material_id,
            video_path=Path(video_path),
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            processing_status="ready" if event_tags else "ready_without_event",
            events=event_tags,
            participants=participant_references,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _participant_id(prediction: Mapping[str, Any]) -> str:
        """从预测结果得到可检索的人物编号。"""
        stable_id = str(prediction.get("participant_id") or "").strip()
        if stable_id:
            return stable_id
        color = str(prediction.get("jersey_color") or "unknown").strip().lower()
        number = str(prediction.get("jersey_number") or "").strip()
        if number:
            return f"{color}#{number}"
        return str(prediction.get("player_id") or f"{color}#unknown")

    @staticmethod
    def _participant_label(prediction: Mapping[str, Any]) -> str:
        """生成统计页面使用的球衣标签。"""
        color = str(prediction.get("jersey_color") or "unknown").lower()
        number = prediction.get("jersey_number")
        if number is not None and str(number).strip():
            return f"{color} #{str(number).strip()}"
        return str(prediction.get("player_id") or f"{color} unknown")

    def build_material(
        self,
        *,
        source_video_id: str,
        segment_id: str,
        video_path: str | Path,
        start_seconds: float,
        end_seconds: float,
        prediction_report: Mapping[str, Any],
        identity_report: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CatalogItem:
        """把一个片段的身份结果和事件结果整理成素材对象。"""
        if end_seconds < start_seconds:
            raise ValueError("end_seconds 不能小于 start_seconds")

        predictions = prediction_report.get("player_predictions", [])
        temporal_events = prediction_report.get("temporal_events", [])
        if not isinstance(predictions, Sequence) or isinstance(
            predictions, (str, bytes)
        ):
            raise ValueError("player_predictions 必须是列表")
        if not isinstance(temporal_events, Sequence) or isinstance(
            temporal_events, (str, bytes)
        ):
            raise ValueError("temporal_events 必须是列表")

        identity_by_track: dict[str, Mapping[str, Any]] = {}
        if isinstance(identity_report, Mapping):
            resolutions = identity_report.get("resolutions", [])
            if isinstance(resolutions, Sequence) and not isinstance(
                resolutions, (str, bytes)
            ):
                for value in resolutions:
                    if isinstance(value, Mapping) and value.get("track_id") is not None:
                        identity_by_track[str(value["track_id"])] = value

        participants: dict[str, ParticipantReference] = {}
        for prediction in predictions:
            if not isinstance(prediction, Mapping):
                continue
            track_id = str(prediction.get("player_id") or "unknown")
            participant_id = self._participant_id(prediction)
            identity = identity_by_track.get(track_id, {})
            participants[participant_id] = ParticipantReference(
                participant_id=participant_id,
                track_id=track_id,
                jersey_color=prediction.get("jersey_color"),
                jersey_number=(
                    str(prediction["jersey_number"])
                    if prediction.get("jersey_number") is not None
                    else None
                ),
                player_name=prediction.get("player_name"),
                identity_status=identity.get("status"),
            )

        events = self._build_events(predictions, temporal_events)
        return CatalogItem(
            material_id=f"{source_video_id}:{segment_id}",
            source_video_id=source_video_id,
            segment_id=segment_id,
            video_path=Path(video_path),
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            processing_status="ready" if events else "ready_without_event",
            events=events,
            participants=tuple(participants.values()),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _build_events(
        predictions: Sequence[Any], temporal_events: Sequence[Any]
    ) -> tuple[EventTag, ...]:
        """优先读取时序事件；缺失时回退到球员级预测。"""
        events: list[EventTag] = []
        for value in temporal_events:
            if not isinstance(value, Mapping):
                continue
            event_name = str(value.get("event") or "blank")
            if event_name == "blank":
                continue
            events.append(
                EventTag(
                    event=event_name,
                    confidence=float(value.get("confidence") or 0.0),
                    player_id=(
                        str(value["player_id"])
                        if value.get("player_id") is not None
                        else None
                    ),
                    start_seconds=(
                        float(value["start_time"])
                        if value.get("start_time") is not None
                        else None
                    ),
                    end_seconds=(
                        float(value["end_time"])
                        if value.get("end_time") is not None
                        else None
                    ),
                )
            )
        if events:
            return tuple(events)

        for value in predictions:
            if not isinstance(value, Mapping):
                continue
            event_name = str(value.get("event") or "blank")
            if event_name == "blank":
                continue
            events.append(
                EventTag(
                    event=event_name,
                    confidence=float(value.get("confidence") or 0.0),
                    player_id=(
                        str(value["player_id"])
                        if value.get("player_id") is not None
                        else None
                    ),
                )
            )
        return tuple(events)

    def summarize_reports(
        self, reports: Sequence[Mapping[str, Any]]
    ) -> MaterialStatistics:
        """汇总 PlayNet 报告中的事件、人物和置信度。"""
        event_counts: Counter[str] = Counter()
        participant_counts: Counter[str] = Counter()
        confidences: list[float] = []
        prediction_count = 0
        non_background_count = 0
        temporal_event_count = 0

        for report in reports:
            predictions = report.get("player_predictions", [])
            temporal_events = report.get("temporal_events", [])
            if not isinstance(predictions, list):
                raise ValueError("player_predictions 必须是列表")
            if not isinstance(temporal_events, list):
                raise ValueError("temporal_events 必须是列表")
            temporal_event_count += len(temporal_events)
            for prediction in predictions:
                if not isinstance(prediction, Mapping):
                    raise ValueError("每条球员预测必须是对象")
                prediction_count += 1
                event = str(prediction.get("event") or "unknown")
                event_counts[event] += 1
                participant_counts[self._participant_label(prediction)] += 1
                if event != "blank":
                    non_background_count += 1
                confidence = prediction.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))

        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return MaterialStatistics(
            clip_count=len(reports),
            player_prediction_count=prediction_count,
            non_background_prediction_count=non_background_count,
            temporal_event_count=temporal_event_count,
            mean_confidence=mean_confidence,
            event_counts=dict(sorted(event_counts.items())),
            participant_counts=dict(sorted(participant_counts.items())),
        )

    def summarize_files(self, report_paths: Sequence[str | Path]) -> MaterialStatistics:
        """读取多个预测 JSON 文件并汇总。"""
        reports: list[Mapping[str, Any]] = []
        for path_value in report_paths:
            path = Path(path_value).expanduser()
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
            if not isinstance(value, Mapping):
                raise ValueError(f"预测报告根节点必须是对象：{path}")
            reports.append(value)
        return self.summarize_reports(reports)
