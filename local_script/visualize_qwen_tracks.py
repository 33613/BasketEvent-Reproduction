"""Render SAM3 trajectories with Qwen recognition results over a video.

The raw SAM3 document contains every player and ball candidate, while the
clean document produced by ``recognize.py`` contains only candidates accepted
by Qwen. This script draws both sets so false rejections remain visible:

* green boxes are player tracks retained by Qwen and include the recognized
  player name, jersey number, and jersey color;
* orange boxes are SAM3 player tracks that Qwen did not retain;
* yellow boxes identify the basketball track selected by Qwen;
* gray boxes are unselected SAM3 basketball candidates.

The clean format historically renumbered accepted players. When a
``source_track_id`` is unavailable, this script recovers the raw ID by exact
trajectory matching and records any ambiguous match in a JSON sidecar report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_ACCEPTED_PLAYER_COLOR = (60, 210, 60)
_REJECTED_PLAYER_COLOR = (0, 165, 255)
_SELECTED_BALL_COLOR = (0, 255, 255)
_OTHER_BALL_COLOR = (150, 150, 150)
_TEXT_COLOR = (255, 255, 255)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for trajectory rendering.

    Args:
        argv: Optional argument sequence used by tests. ``None`` reads the
            process command line.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize raw SAM3 candidates and Qwen-cleaned tracks."
    )
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--raw_json_path", type=Path, required=True)
    parser.add_argument("--clean_json_path", type=Path, required=True)
    parser.add_argument("--output_video_path", type=Path, required=True)
    parser.add_argument(
        "--report_json_path",
        type=Path,
        default=None,
        help=(
            "Optional JSON audit path. Defaults to the output video path with "
            "a .json suffix."
        ),
    )
    parser.add_argument(
        "--show-rejected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw raw player candidates that Qwen did not retain.",
    )
    parser.add_argument(
        "--show-ball-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw selected and unselected SAM3 basketball candidates.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Render only the first N frames for a quick diagnostic preview.",
    )
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.55)
    return parser.parse_args(argv)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON document whose root must be an object.

    Args:
        path: JSON file to read.
        label: Human-readable input name used in errors.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the requested path does not exist.
        ValueError: If the JSON root is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return value


def trajectory_signature(payload: Mapping[str, Any]) -> str | None:
    """Return a stable signature for one non-empty trajectory.

    Args:
        payload: Track payload expected to contain a trajectory list.

    Returns:
        Canonical JSON trajectory text, or ``None`` for an invalid trajectory.
    """
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return None
    return json.dumps(trajectory, ensure_ascii=True, separators=(",", ":"))


def match_clean_tracks_to_raw(
    raw_tracks: Mapping[str, Any],
    clean_tracks: Mapping[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Match Qwen-retained player IDs back to raw SAM3 player IDs.

    ``recognize.py`` now records ``source_track_id``. Older clean files are
    supported by comparing their copied trajectories with raw trajectories.

    Args:
        raw_tracks: Raw SAM3 trajectory document.
        clean_tracks: Qwen-cleaned trajectory document.

    Returns:
        A ``clean_id -> raw_id`` mapping and a list of audit diagnostics.
    """
    raw_player_ids = {
        str(track_id)
        for track_id, payload in raw_tracks.items()
        if str(track_id).startswith("player_") and isinstance(payload, Mapping)
    }
    signature_index: dict[str, list[str]] = {}
    for raw_id in sorted(raw_player_ids):
        signature = trajectory_signature(raw_tracks[raw_id])
        if signature is not None:
            signature_index.setdefault(signature, []).append(raw_id)

    matches: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []
    claimed_raw_ids: set[str] = set()

    for clean_id, payload in sorted(clean_tracks.items()):
        clean_id = str(clean_id)
        if not clean_id.startswith("player_") or not isinstance(payload, Mapping):
            continue

        source_track_id = payload.get("source_track_id")
        if source_track_id is not None:
            candidates = [str(source_track_id)]
            method = "source_track_id"
        else:
            signature = trajectory_signature(payload)
            candidates = signature_index.get(signature, []) if signature else []
            method = "trajectory_signature"

        valid_candidates = [item for item in candidates if item in raw_player_ids]
        available_candidates = [
            item for item in valid_candidates if item not in claimed_raw_ids
        ]
        if len(available_candidates) == 1:
            raw_id = available_candidates[0]
            matches[clean_id] = raw_id
            claimed_raw_ids.add(raw_id)
            diagnostics.append(
                {
                    "status": "matched",
                    "clean_track_id": clean_id,
                    "raw_track_id": raw_id,
                    "method": method,
                }
            )
            continue

        diagnostics.append(
            {
                "status": "unmatched" if not valid_candidates else "ambiguous",
                "clean_track_id": clean_id,
                "method": method,
                "candidate_raw_track_ids": valid_candidates,
            }
        )

    return matches, diagnostics


def match_clean_ball_to_raw(
    raw_tracks: Mapping[str, Any], clean_tracks: Mapping[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    """Match Qwen's selected basketball trajectory to one raw candidate.

    Args:
        raw_tracks: Raw SAM3 trajectory document.
        clean_tracks: Qwen-cleaned trajectory document.

    Returns:
        Selected raw basketball ID, if unique, and an audit diagnostic.
    """
    clean_ball = clean_tracks.get("ball")
    if not isinstance(clean_ball, Mapping):
        return None, {"status": "missing_clean_ball"}

    source_track_id = clean_ball.get("source_track_id")
    if source_track_id is not None and str(source_track_id) in raw_tracks:
        return str(source_track_id), {
            "status": "matched",
            "raw_track_id": str(source_track_id),
            "method": "source_track_id",
        }

    clean_signature = trajectory_signature(clean_ball)
    candidates = []
    for raw_id, payload in raw_tracks.items():
        if not str(raw_id).startswith("ball_") or not isinstance(payload, Mapping):
            continue
        if trajectory_signature(payload) == clean_signature:
            candidates.append(str(raw_id))

    if len(candidates) == 1:
        return candidates[0], {
            "status": "matched",
            "raw_track_id": candidates[0],
            "method": "trajectory_signature",
        }
    return None, {
        "status": "unmatched" if not candidates else "ambiguous",
        "candidate_raw_track_ids": candidates,
        "method": "trajectory_signature",
    }


def build_player_labels(
    raw_tracks: Mapping[str, Any],
    clean_tracks: Mapping[str, Any],
    clean_to_raw: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build display labels and Qwen acceptance states for raw tracks.

    Args:
        raw_tracks: Raw SAM3 trajectory document.
        clean_tracks: Qwen-cleaned trajectory document.
        clean_to_raw: Mapping produced by ``match_clean_tracks_to_raw``.

    Returns:
        Raw player IDs mapped to display metadata.
    """
    raw_to_clean = {raw_id: clean_id for clean_id, raw_id in clean_to_raw.items()}
    labels: dict[str, dict[str, Any]] = {}
    for raw_id in sorted(raw_tracks):
        if not str(raw_id).startswith("player_"):
            continue
        clean_id = raw_to_clean.get(str(raw_id))
        if clean_id is None:
            labels[str(raw_id)] = {
                "accepted": False,
                "text": f"{raw_id} | Qwen not retained",
            }
            continue

        payload = clean_tracks[clean_id]
        name = payload.get("player_name") or "unknown name"
        number = payload.get("jersey_number") or "?"
        color = payload.get("jersey_color") or "unknown color"
        labels[str(raw_id)] = {
            "accepted": True,
            "clean_track_id": clean_id,
            "text": f"{raw_id} | {name} | #{number} {color}",
        }
    return labels


def bbox_at_frame(payload: Any, frame_index: int) -> list[float] | None:
    """Read and validate one ``xywh`` bounding box from a trajectory.

    Args:
        payload: Track payload containing a trajectory list.
        frame_index: Zero-based video frame index.

    Returns:
        Four floating-point ``xywh`` values, or ``None`` when absent.
    """
    if not isinstance(payload, Mapping):
        return None
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or frame_index >= len(trajectory):
        return None
    bbox = trajectory[frame_index]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values


def draw_labeled_box(
    cv2: Any,
    frame: Any,
    bbox: Sequence[float],
    text: str,
    color: tuple[int, int, int],
    line_thickness: int,
    font_scale: float,
) -> None:
    """Draw one clipped box and a readable text label on an OpenCV frame.

    Args:
        cv2: Imported OpenCV module.
        frame: Mutable BGR image array.
        bbox: ``xywh`` box in source-video pixels.
        text: ASCII label rendered above or inside the box.
        color: BGR box and label-background color.
        line_thickness: Rectangle line thickness in pixels.
        font_scale: OpenCV font scale.
    """
    height, width = frame.shape[:2]
    x, y, box_width, box_height = bbox
    x1 = max(0, min(width - 1, int(round(x))))
    y1 = max(0, min(height - 1, int(round(y))))
    x2 = max(0, min(width - 1, int(round(x + box_width))))
    y2 = max(0, min(height - 1, int(round(y + box_height))))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, 1)
    text_width, text_height = text_size
    label_top = max(0, y1 - text_height - baseline - 6)
    label_bottom = min(height - 1, label_top + text_height + baseline + 6)
    label_right = min(width - 1, x1 + text_width + 8)
    cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
    text_y = min(label_bottom - baseline - 3, height - 2)
    cv2.putText(
        frame,
        text,
        (x1 + 4, text_y),
        font,
        font_scale,
        _TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def render_overlay(
    video_path: Path,
    raw_tracks: Mapping[str, Any],
    clean_tracks: Mapping[str, Any],
    output_video_path: Path,
    show_rejected: bool,
    show_ball_candidates: bool,
    max_frames: int | None,
    line_thickness: int,
    font_scale: float,
) -> dict[str, Any]:
    """Render Qwen acceptance and identities over raw SAM3 trajectories.

    Args:
        video_path: Source MP4 path.
        raw_tracks: Raw SAM3 trajectory document.
        clean_tracks: Qwen-cleaned trajectory document.
        output_video_path: Destination MP4 path.
        show_rejected: Whether to draw Qwen-rejected player candidates.
        show_ball_candidates: Whether to draw SAM3 basketball candidates.
        max_frames: Optional maximum number of frames to render.
        line_thickness: Rectangle line thickness in pixels.
        font_scale: OpenCV font scale.

    Returns:
        JSON-serializable rendering and track-matching report.

    Raises:
        RuntimeError: If OpenCV is unavailable or the video cannot be opened.
        ValueError: If a numeric rendering option is invalid.
    """
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")
    if line_thickness <= 0:
        raise ValueError("--line-thickness must be positive")
    if font_scale <= 0:
        raise ValueError("--font-scale must be positive")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required. Install opencv-python or opencv-python-headless."
        ) from error

    clean_to_raw, player_diagnostics = match_clean_tracks_to_raw(
        raw_tracks, clean_tracks
    )
    player_labels = build_player_labels(raw_tracks, clean_tracks, clean_to_raw)
    selected_ball_id, ball_diagnostic = match_clean_ball_to_raw(
        raw_tracks, clean_tracks
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Input video has invalid dimensions: {video_path}")
    if fps <= 0:
        fps = 30.0

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            for raw_id, metadata in player_labels.items():
                if not metadata["accepted"] and not show_rejected:
                    continue
                bbox = bbox_at_frame(raw_tracks.get(raw_id), frame_index)
                if bbox is None:
                    continue
                color = (
                    _ACCEPTED_PLAYER_COLOR
                    if metadata["accepted"]
                    else _REJECTED_PLAYER_COLOR
                )
                draw_labeled_box(
                    cv2,
                    frame,
                    bbox,
                    metadata["text"],
                    color,
                    line_thickness,
                    font_scale,
                )

            if show_ball_candidates:
                for raw_id, payload in raw_tracks.items():
                    raw_id = str(raw_id)
                    if not raw_id.startswith("ball_"):
                        continue
                    bbox = bbox_at_frame(payload, frame_index)
                    if bbox is None:
                        continue
                    is_selected = raw_id == selected_ball_id
                    text = (
                        f"{raw_id} | Qwen selected ball"
                        if is_selected
                        else f"{raw_id} | ball candidate"
                    )
                    draw_labeled_box(
                        cv2,
                        frame,
                        bbox,
                        text,
                        _SELECTED_BALL_COLOR if is_selected else _OTHER_BALL_COLOR,
                        line_thickness,
                        font_scale,
                    )

            banner = (
                f"Qwen accepted players: {len(clean_to_raw)}/{len(player_labels)} | "
                "green=accepted orange=not retained"
            )
            cv2.rectangle(frame, (0, 0), (min(width - 1, 760), 30), (20, 20, 20), -1)
            cv2.putText(
                frame,
                banner,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                _TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    raw_not_retained = sorted(
        raw_id for raw_id, metadata in player_labels.items() if not metadata["accepted"]
    )
    accepted_players = []
    for clean_id, raw_id in sorted(clean_to_raw.items()):
        payload = clean_tracks[clean_id]
        accepted_players.append(
            {
                "clean_track_id": clean_id,
                "raw_track_id": raw_id,
                "player_name": payload.get("player_name"),
                "jersey_number": payload.get("jersey_number"),
                "jersey_color": payload.get("jersey_color"),
            }
        )

    return {
        "schema_version": "basketevent_qwen_visualization.v1",
        "video": str(video_path),
        "output_video": str(output_video_path),
        "source_frame_count": source_frame_count,
        "rendered_frame_count": frame_index,
        "width": width,
        "height": height,
        "fps": fps,
        "accepted_players": accepted_players,
        "raw_player_tracks_not_retained": raw_not_retained,
        "player_match_diagnostics": player_diagnostics,
        "selected_raw_ball_track_id": selected_ball_id,
        "ball_match_diagnostic": ball_diagnostic,
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one UTF-8 JSON object and create its parent directory.

    Args:
        path: Destination JSON path.
        value: JSON-serializable mapping to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    """Load trajectories, render the overlay, and save an audit report."""
    args = parse_args(argv)
    raw_tracks = load_json_object(args.raw_json_path, "Raw SAM3 JSON")
    clean_tracks = load_json_object(args.clean_json_path, "Qwen clean JSON")
    report = render_overlay(
        video_path=args.video_path,
        raw_tracks=raw_tracks,
        clean_tracks=clean_tracks,
        output_video_path=args.output_video_path,
        show_rejected=args.show_rejected,
        show_ball_candidates=args.show_ball_candidates,
        max_frames=args.max_frames,
        line_thickness=args.line_thickness,
        font_scale=args.font_scale,
    )
    report_path = args.report_json_path or args.output_video_path.with_suffix(".json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
