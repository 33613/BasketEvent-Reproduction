"""复用固定窗口轨迹，为每条非空事件解析主体人物身份。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.modules.identity.models import IdentityObservation, TrackCrop
from src.modules.identity.resolver import IdentityResolver


class TrackObserver(Protocol):
    """约束把一组人物截图转换为逐帧身份观察的接口。"""

    def observe(self, crops: Sequence[TrackCrop]) -> list[IdentityObservation]:
        """返回每张截图的球衣颜色和号码观察。"""
        ...


class TrackSamplerProtocol(Protocol):
    """约束事件身份服务需要的最小轨迹取样能力。"""

    sample_count: int
    pad_ratio: float

    def load_annotations(self, path: str | Path) -> dict[str, Any]:
        """读取SAM3轨迹JSON。"""
        ...

    def sample(
        self,
        video_path: str | Path,
        annotations: Mapping[str, Any],
        track_prefix: str = "player",
        track_ids: Sequence[str] | None = None,
    ) -> dict[str, list[TrackCrop]]:
        """只为指定轨迹返回截图。"""
        ...


def _safe_component(value: str) -> str:
    """把窗口和轨迹编号转换为可移植文件名。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return normalized.strip("._") or "item"


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """以原子替换方式保存中间证据，避免中断留下半个JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parse_track_reference(reference: str) -> tuple[str, str]:
    """解析时间线中的 ``窗口编号/人物轨迹编号``。"""
    parts = str(reference).rsplit("/", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise ValueError(f"事件轨迹引用格式无效：{reference!r}")
    return parts[0].strip(), parts[1].strip()


class EventActorIdentityService:
    """只观察已被PlayNet判为非空事件的主体轨迹。

    时间线已经保存事件来源窗口和人物轨迹，因此这里直接复用窗口MP4及
    SAM3 bbox，不重新运行SAM3。每个唯一的“窗口/轨迹”只调用一次视觉
    模型，证据写入缓存后可在SSH中断或重复运行时继续复用。
    """

    def __init__(
        self,
        *,
        sampler: TrackSamplerProtocol,
        observer: TrackObserver,
        resolver: IdentityResolver,
    ) -> None:
        """注入截图取样、Qwen观察和固定规则解析组件。"""
        self.sampler = sampler
        self.observer = observer
        self.resolver = resolver

    def _cache_path(self, cache_directory: Path, window_id: str, track_id: str) -> Path:
        """返回一条窗口轨迹的可恢复证据文件路径。"""
        filename = f"{_safe_component(window_id)}__{_safe_component(track_id)}.json"
        return cache_directory / filename

    @staticmethod
    def _observations_from_document(
        document: Mapping[str, Any],
    ) -> list[IdentityObservation]:
        """从缓存JSON恢复逐帧观察。"""
        values = document.get("observations", [])
        if not isinstance(values, list):
            return []
        observations: list[IdentityObservation] = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            try:
                observations.append(
                    IdentityObservation(
                        track_id=str(value["track_id"]),
                        image_index=int(value["image_index"]),
                        frame_index=int(value["frame_index"]),
                        is_on_court_player=(
                            value.get("is_on_court_player")
                            if isinstance(value.get("is_on_court_player"), bool)
                            else None
                        ),
                        jersey_color=(
                            str(value["jersey_color"])
                            if value.get("jersey_color") is not None
                            else None
                        ),
                        jersey_number=(
                            str(value["jersey_number"])
                            if value.get("jersey_number") is not None
                            else None
                        ),
                        confidence=float(value.get("confidence") or 0.0),
                        evidence=(
                            str(value["evidence"])
                            if value.get("evidence") is not None
                            else None
                        ),
                        raw=(
                            dict(value.get("raw", {}))
                            if isinstance(value.get("raw"), Mapping)
                            else {}
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return observations

    def _read_matching_cache(self, path: Path) -> dict[str, Any] | None:
        """仅复用取样配置一致且结构完整的轨迹证据。"""
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(document, dict):
            return None
        if document.get("status") != "completed":
            return None
        configuration = document.get("sampling", {})
        if not isinstance(configuration, Mapping):
            return None
        if configuration.get("sample_count") != self.sampler.sample_count:
            return None
        if float(configuration.get("pad_ratio", -1.0)) != self.sampler.pad_ratio:
            return None
        return document

    def _observe_reference(
        self,
        *,
        window_id: str,
        track_id: str,
        window_video_directory: Path,
        raw_tracks_directory: Path,
        cache_directory: Path,
    ) -> dict[str, Any]:
        """读取一条缓存轨迹、最多抽十帧并保存Qwen证据。"""
        cache_path = self._cache_path(cache_directory, window_id, track_id)
        cached = self._read_matching_cache(cache_path)
        if cached is not None:
            cached["cache_reused"] = True
            return cached

        video_path = window_video_directory / f"{window_id}.mp4"
        raw_path = raw_tracks_directory / f"{window_id}.json"
        document: dict[str, Any] = {
            "schema_version": "basketevent_event_actor_track_evidence.v1",
            "window_id": window_id,
            "track_id": track_id,
            "video": str(video_path),
            "raw_tracks": str(raw_path),
            "sampling": {
                "sample_count": self.sampler.sample_count,
                "pad_ratio": self.sampler.pad_ratio,
            },
            "observations": [],
            "status": "pending",
            "cache_reused": False,
        }
        try:
            if not video_path.is_file():
                raise FileNotFoundError(f"找不到事件来源窗口：{video_path}")
            annotations = self.sampler.load_annotations(raw_path)
            if track_id not in annotations:
                raise KeyError(f"窗口 {window_id} 中不存在轨迹 {track_id}")
            crops = self.sampler.sample(
                video_path,
                annotations,
                track_ids=(track_id,),
            ).get(track_id, [])
            if not crops:
                raise ValueError(f"轨迹 {window_id}/{track_id} 没有可用截图")
            observations = self.observer.observe(crops)
            document.update(
                {
                    "status": "completed",
                    "crop_count": len(crops),
                    "observations": [asdict(value) for value in observations],
                }
            )
            raw_output = getattr(self.observer, "last_output_text", None)
            if isinstance(raw_output, str) and raw_output.strip():
                document["raw_qwen_output"] = raw_output
        except Exception as error:  # 单条轨迹失败不能删除事件或中断其他身份。
            document.update(
                {
                    "status": "failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
            raw_output = getattr(self.observer, "last_output_text", None)
            if isinstance(raw_output, str) and raw_output.strip():
                document["raw_qwen_output"] = raw_output
        _write_json_atomic(cache_path, document)
        return document

    def process(
        self,
        *,
        source_video_id: str,
        timeline_report: Mapping[str, Any],
        window_video_directory: str | Path,
        raw_tracks_directory: str | Path,
        cache_directory: str | Path,
    ) -> dict[str, Any]:
        """为时间线中的每条非空事件生成可查询主体身份。"""
        raw_events = timeline_report.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("事件时间线的events必须是列表")

        events = [value for value in raw_events if isinstance(value, Mapping)]
        references: dict[str, tuple[str, str]] = {}
        for event in events:
            raw_references = event.get("track_references", [])
            if not isinstance(raw_references, Sequence) or isinstance(
                raw_references, (str, bytes)
            ):
                continue
            for reference in raw_references:
                text = str(reference)
                references.setdefault(text, parse_track_reference(text))

        evidence_by_reference: dict[str, dict[str, Any]] = {}
        for reference, (window_id, track_id) in sorted(references.items()):
            print(f"正在识别事件轨迹 {reference}", flush=True)
            evidence_by_reference[reference] = self._observe_reference(
                window_id=window_id,
                track_id=track_id,
                window_video_directory=Path(window_video_directory),
                raw_tracks_directory=Path(raw_tracks_directory),
                cache_directory=Path(cache_directory),
            )

        event_resolutions: list[dict[str, Any]] = []
        for index, event in enumerate(events):
            event_id = str(event.get("event_id") or f"event_{index:05d}")
            event_name = str(event.get("event") or "unknown")
            raw_references = event.get("track_references", [])
            track_references = (
                tuple(str(value) for value in raw_references)
                if isinstance(raw_references, Sequence)
                and not isinstance(raw_references, (str, bytes))
                else ()
            )
            observations: list[IdentityObservation] = []
            failed_references: list[str] = []
            for reference in track_references:
                evidence = evidence_by_reference.get(reference, {})
                observations.extend(self._observations_from_document(evidence))
                if evidence.get("status") != "completed":
                    failed_references.append(reference)

            decision = self.resolver.resolve(
                game_id=source_video_id,
                clip_id=event_id,
                track_id=event_id,
                observations=observations,
            )
            resolution = decision.to_dict()
            resolution.update(
                {
                    "event_id": event_id,
                    "event": event_name,
                    "track_references": list(track_references),
                    "observation_count": len(observations),
                    "failed_track_references": failed_references,
                }
            )
            event_resolutions.append(resolution)

        failed_track_count = sum(
            value.get("status") != "completed"
            for value in evidence_by_reference.values()
        )
        return {
            "schema_version": "basketevent_event_actor_identity.v1",
            "source_video_id": source_video_id,
            "strategy": "reuse_window_sam3_bbox_and_observe_event_actor_only",
            "sample_count_per_track_reference": self.sampler.sample_count,
            "pad_ratio": self.sampler.pad_ratio,
            "event_count": len(event_resolutions),
            "unique_track_reference_count": len(evidence_by_reference),
            "failed_track_reference_count": failed_track_count,
            "event_resolutions": event_resolutions,
            "track_evidence": evidence_by_reference,
        }
