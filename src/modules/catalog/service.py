"""把处理结果登记为可按事件和人物检索的视频素材。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.catalog.models import CatalogItem, EventTag, ParticipantReference
from src.modules.catalog.repository import MaterialCatalog


class MaterialCatalogService:
    """把切分信息、身份结果和 PlayNet 预测组合成素材记录。"""

    def __init__(self, catalog: MaterialCatalog) -> None:
        """注入素材目录，避免业务服务绑定具体数据库。"""
        self.catalog = catalog

    @staticmethod
    def _participant_id(prediction: Mapping[str, Any]) -> str:
        """优先使用 Identity 的稳定编号，缺失时再生成临时检索键。"""
        stable_id = str(prediction.get("participant_id") or "").strip()
        if stable_id:
            return stable_id
        color = str(prediction.get("jersey_color") or "unknown").strip().lower()
        number = str(prediction.get("jersey_number") or "").strip()
        if number:
            return f"{color}#{number}"
        return str(prediction.get("player_id") or f"{color}#unknown")

    def register_processed_clip(
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
        replace: bool = False,
    ) -> CatalogItem:
        """登记一个完成事件推理的片段并返回规范化素材。"""
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

        identity_by_source: dict[str, Mapping[str, Any]] = {}
        if isinstance(identity_report, Mapping):
            for value in identity_report.get("resolutions", []):
                if isinstance(value, Mapping) and value.get("track_id") is not None:
                    identity_by_source[str(value["track_id"])] = value

        participants: dict[str, ParticipantReference] = {}
        for raw_prediction in predictions:
            if not isinstance(raw_prediction, Mapping):
                continue
            track_id = str(raw_prediction.get("player_id") or "unknown")
            participant_id = self._participant_id(raw_prediction)
            identity = identity_by_source.get(track_id, {})
            participants[participant_id] = ParticipantReference(
                participant_id=participant_id,
                track_id=track_id,
                jersey_color=raw_prediction.get("jersey_color"),
                jersey_number=(
                    str(raw_prediction["jersey_number"])
                    if raw_prediction.get("jersey_number") is not None
                    else None
                ),
                player_name=raw_prediction.get("player_name"),
                identity_status=identity.get("status"),
            )

        events: list[EventTag] = []
        for raw_event in temporal_events:
            if not isinstance(raw_event, Mapping):
                continue
            event_name = str(raw_event.get("event") or "blank")
            if event_name == "blank":
                continue
            events.append(
                EventTag(
                    event=event_name,
                    confidence=float(raw_event.get("confidence") or 0.0),
                    player_id=(
                        str(raw_event["player_id"])
                        if raw_event.get("player_id") is not None
                        else None
                    ),
                    start_seconds=(
                        float(raw_event["start_time"])
                        if raw_event.get("start_time") is not None
                        else None
                    ),
                    end_seconds=(
                        float(raw_event["end_time"])
                        if raw_event.get("end_time") is not None
                        else None
                    ),
                )
            )
        if not events:
            for raw_prediction in predictions:
                if not isinstance(raw_prediction, Mapping):
                    continue
                event_name = str(raw_prediction.get("event") or "blank")
                if event_name != "blank":
                    events.append(
                        EventTag(
                            event=event_name,
                            confidence=float(raw_prediction.get("confidence") or 0.0),
                            player_id=(
                                str(raw_prediction["player_id"])
                                if raw_prediction.get("player_id") is not None
                                else None
                            ),
                        )
                    )

        item = CatalogItem(
            material_id=f"{source_video_id}:{segment_id}",
            source_video_id=source_video_id,
            segment_id=segment_id,
            video_path=Path(video_path),
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            processing_status="ready" if events else "ready_without_event",
            events=tuple(events),
            participants=tuple(participants.values()),
            metadata=dict(metadata or {}),
        )
        self.catalog.add(item, replace=replace)
        return item
