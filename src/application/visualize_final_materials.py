"""为最终事件素材生成适合人工复核的带标注视频。"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.materials.visualization import bbox_at_frame, draw_labeled_box


@dataclass(frozen=True)
class TrackReference:
    """保存一条事件人物轨迹及其所在窗口的全局时间。"""

    reference: str
    window_id: str
    track_id: str
    window_start: float
    window_end: float
    payload: Mapping[str, Any]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """读取JSON对象并给出面向业务文件的错误信息。"""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label}根节点必须是对象：{path}")
    return value


def _event_identity_label(identity: Mapping[str, Any] | None) -> str:
    """把身份结论整理为可直接画在视频上的短文本。"""
    if not identity:
        return "identity unavailable"
    status = str(identity.get("status") or "unknown")
    color = identity.get("jersey_color")
    number = identity.get("jersey_number")
    if color is not None or number is not None:
        return f"{status} | {color or '?'} #{number or '?'}"
    candidates = identity.get("candidates", [])
    labels: list[str] = []
    if isinstance(candidates, list):
        for value in candidates[:3]:
            if isinstance(value, Mapping):
                labels.append(
                    f"{value.get('jersey_color') or '?'} #{value.get('jersey_number') or '?'}"
                )
    return f"{status} | {' / '.join(labels)}" if labels else status


def _material_index(
    timeline: Mapping[str, Any], finalization: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """连接素材导出记录、素材草稿和全局事件。"""
    events = {
        str(value["event_id"]): value
        for value in timeline.get("events", [])
        if isinstance(value, Mapping) and value.get("event_id") is not None
    }
    drafts = {
        str(value["material_id"]): value
        for value in timeline.get("material_drafts", [])
        if isinstance(value, Mapping) and value.get("material_id") is not None
    }
    rows: list[dict[str, Any]] = []
    for exported in finalization.get("exported_materials", []):
        if not isinstance(exported, Mapping):
            continue
        material_id = str(exported.get("material_id") or "")
        draft = drafts.get(material_id, {})
        event_ids = [str(value) for value in exported.get("event_ids", [])]
        rows.append(
            {
                "material_id": material_id,
                "video_path": str(exported.get("video_path") or ""),
                "start_seconds": float(
                    exported.get("start_seconds", draft.get("start_seconds", 0.0))
                ),
                "end_seconds": float(
                    exported.get("end_seconds", draft.get("end_seconds", 0.0))
                ),
                "events": [events[event_id] for event_id in event_ids if event_id in events],
            }
        )
    return rows


def _resolve_job_path(job_root: Path, value: str) -> Path:
    """兼容报告中的绝对路径、任务相对路径和项目相对路径。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    direct = job_root / path
    if direct.exists():
        return direct
    # 长视频报告通常从项目根目录写入 tests/... 路径。
    for parent in (job_root, *job_root.parents):
        candidate = parent / path
        if candidate.exists():
            return candidate
    return direct


def _load_track_references(
    *,
    job_root: Path,
    video_id: str,
    events: Sequence[Mapping[str, Any]],
    segments: Mapping[str, Mapping[str, Any]],
) -> list[TrackReference]:
    """从已有SAM3缓存加载事件主体轨迹，不重新运行追踪。"""
    documents: dict[str, dict[str, Any]] = {}
    references: list[TrackReference] = []
    for event in events:
        raw_references = event.get("track_references", [])
        if not isinstance(raw_references, list):
            continue
        for raw_reference in raw_references:
            reference = str(raw_reference)
            parts = reference.rsplit("/", 1)
            if len(parts) != 2 or parts[0] not in segments:
                continue
            window_id, track_id = parts
            if window_id not in documents:
                path = (
                    job_root
                    / "window_artifacts"
                    / video_id
                    / "tracks"
                    / "raw"
                    / f"{window_id}.json"
                )
                documents[window_id] = _read_json(path, "SAM3轨迹")
            payload = documents[window_id].get(track_id)
            if not isinstance(payload, Mapping):
                continue
            segment = segments[window_id]
            references.append(
                TrackReference(
                    reference=reference,
                    window_id=window_id,
                    track_id=track_id,
                    window_start=float(segment["start_seconds"]),
                    window_end=float(segment["end_seconds"]),
                    payload=payload,
                )
            )
    return references


def bbox_for_global_time(
    references: Sequence[TrackReference], global_time: float
) -> tuple[list[float] | None, str | None]:
    """选择当前全局时刻最可靠的一条窗口轨迹框。"""
    candidates: list[tuple[float, list[float], str]] = []
    for reference in references:
        duration = reference.window_end - reference.window_start
        trajectory = reference.payload.get("trajectory")
        if (
            duration <= 0
            or not isinstance(trajectory, list)
            or not trajectory
            or global_time < reference.window_start
            or global_time > reference.window_end
        ):
            continue
        position = (global_time - reference.window_start) / duration
        frame_index = min(len(trajectory) - 1, max(0, int(position * len(trajectory))))
        bbox = bbox_at_frame(reference.payload, frame_index)
        if bbox is None:
            continue
        # 重叠窗口同时有框时，优先使用离窗口边界更远的证据。
        edge_margin = min(
            global_time - reference.window_start,
            reference.window_end - global_time,
        )
        candidates.append((edge_margin, bbox, reference.reference))
    if not candidates:
        return None, None
    _, bbox, reference = max(candidates, key=lambda value: value[0])
    return bbox, reference


def _draw_review_panel(
    cv2: Any,
    frame: Any,
    *,
    event: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    material_time: float,
    global_time: float,
    material_duration: float,
    material_start: float,
) -> None:
    """绘制事件、身份、双时间和事件证据时间条。"""
    height, width = frame.shape[:2]
    event_name = str(event.get("event") or "unknown")
    confidence = float(event.get("confidence") or 0.0)
    lines = [
        f"EVENT: {event_name} | confidence {confidence:.3f}",
        f"ACTOR: {_event_identity_label(identity)}",
        f"MATERIAL: {material_time:.2f}s / {material_duration:.2f}s   SOURCE: {global_time:.2f}s",
    ]
    panel_width = min(width - 1, 900)
    cv2.rectangle(frame, (0, 0), (panel_width, 86), (20, 20, 20), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 24 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    left, right = 30, max(31, width - 30)
    y = height - 32
    cv2.rectangle(frame, (0, height - 58), (width - 1, height - 1), (20, 20, 20), -1)
    cv2.line(frame, (left, y), (right, y), (170, 170, 170), 4)
    evidence_start = float(event.get("evidence_start_seconds") or material_start)
    evidence_end = float(event.get("evidence_end_seconds") or evidence_start)

    def x_at(source_seconds: float) -> int:
        ratio = (source_seconds - material_start) / max(material_duration, 1e-6)
        return int(round(left + max(0.0, min(1.0, ratio)) * (right - left)))

    cv2.line(
        frame,
        (x_at(evidence_start), y),
        (x_at(evidence_end), y),
        (0, 215, 255),
        9,
    )
    playhead = x_at(global_time)
    cv2.line(frame, (playhead, y - 15), (playhead, y + 15), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "event evidence",
        (left, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )


def _render_one(
    *,
    input_video: Path,
    output_video: Path,
    event: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    references: Sequence[TrackReference],
    material_start: float,
    material_end: float,
    ffmpeg_binary: str,
    overwrite: bool,
) -> dict[str, Any]:
    """渲染一条复核视频，并从原素材复制音频。"""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("生成复核视频需要opencv-python或opencv-python-headless") from error

    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.is_file() and output_video.stat().st_size > 0 and not overwrite:
        return {"status": "reused", "output_video": str(output_video)}
    temporary = output_video.with_name(f".{output_video.stem}.video_only.mp4")
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开最终素材：{input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建临时复核视频：{temporary}")

    frame_index = 0
    actor_frame_count = 0
    duration = material_end - material_start
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            material_time = frame_index / fps
            global_time = material_start + material_time
            bbox, reference = bbox_for_global_time(references, global_time)
            if bbox is not None:
                actor_frame_count += 1
                draw_labeled_box(
                    cv2,
                    frame,
                    bbox,
                    f"EVENT ACTOR | {_event_identity_label(identity)} | {reference}",
                    (0, 220, 0),
                    3,
                    0.58,
                )
            _draw_review_panel(
                cv2,
                frame,
                event=event,
                identity=identity,
                material_time=material_time,
                global_time=global_time,
                material_duration=duration,
                material_start=material_start,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temporary),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "completed",
        "output_video": str(output_video),
        "frame_count": frame_index,
        "actor_bbox_frame_count": actor_frame_count,
    }


def visualize_job(
    *,
    job_root: Path,
    output_directory: Path | None,
    ffmpeg_binary: str,
    overwrite: bool,
    limit: int | None,
) -> dict[str, Any]:
    """批量生成一次长视频任务的最终素材复核视频。"""
    job_root = job_root.expanduser().resolve()
    timeline = _read_json(job_root / "event_timeline.json", "事件时间线")
    identity_report = _read_json(job_root / "event_identity.json", "事件身份")
    finalization = _read_json(job_root / "finalization_report.json", "素材导出报告")
    segment_report = _read_json(job_root / "segments.json", "窗口清单")
    video_id = str(identity_report.get("source_video_id") or job_root.name)
    segments = {
        str(value["segment_id"]): value
        for value in segment_report.get("segments", [])
        if isinstance(value, Mapping) and value.get("segment_id") is not None
    }
    identity_by_event = {
        str(value["event_id"]): value
        for value in identity_report.get("event_resolutions", [])
        if isinstance(value, Mapping) and value.get("event_id") is not None
    }
    output_root = output_directory or job_root / "review_visualizations"
    rows = _material_index(timeline, finalization)
    if limit is not None:
        rows = rows[:limit]

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        events = row["events"]
        if not events:
            results.append({"material_id": row["material_id"], "status": "no_event"})
            continue
        event = events[0]
        event_id = str(event.get("event_id") or "")
        identity = identity_by_event.get(event_id)
        references = _load_track_references(
            job_root=job_root,
            video_id=video_id,
            events=events,
            segments=segments,
        )
        input_video = _resolve_job_path(job_root, row["video_path"])
        output_video = output_root / input_video.name
        print(f"[{index}/{len(rows)}] 正在生成 {output_video.name}", flush=True)
        result = _render_one(
            input_video=input_video,
            output_video=output_video,
            event=event,
            identity=identity,
            references=references,
            material_start=row["start_seconds"],
            material_end=row["end_seconds"],
            ffmpeg_binary=ffmpeg_binary,
            overwrite=overwrite,
        )
        result.update(
            {
                "material_id": row["material_id"],
                "event_id": event_id,
                "event": event.get("event"),
                "identity_status": identity.get("status") if identity else None,
                "input_video": str(input_video),
                "track_references": [value.reference for value in references],
            }
        )
        results.append(result)

    report = {
        "schema_version": "basketevent_material_review_visualization.v1",
        "job_root": str(job_root),
        "output_directory": str(output_root),
        "material_count": len(results),
        "materials": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析最终素材复核可视化参数。"""
    parser = argparse.ArgumentParser(description="为最终事件素材批量生成复核标注视频。")
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="只处理前N条，用于快速检查。")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行批量可视化并打印汇总报告。"""
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit必须为正整数")
    report = visualize_job(
        job_root=args.job_root,
        output_directory=args.output_directory,
        ffmpeg_binary=args.ffmpeg_binary,
        overwrite=args.overwrite,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
