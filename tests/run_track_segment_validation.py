"""Validate PlayNet on one manually audited segment of a mixed SAM3 track.

This runtime experiment does not modify the identity module or the original raw
and clean trajectory files.  It copies one selected source track, masks every
bounding box outside an inclusive stable frame range, attaches an explicitly
provided jersey identity, and preserves the Qwen-selected ball trajectory.
The resulting temporary clean JSON can then be passed to the event-recognition
inference module.

The script records the manual boundary assumption, generated command, model
output, and expected-event comparison in a dedicated run directory.  A
successful expected-event match is evidence about this one controlled input;
it does not establish an automatic trajectory-splitting algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class SegmentSpec:
    """Describe one manually audited player segment.

    Attributes:
        raw_track_id: Source SAM3 track key in the raw JSON.
        output_track_id: Player key written to the temporary clean JSON.
        start_frame: First retained source-video frame, inclusive.
        end_frame: Last retained source-video frame, inclusive.
        jersey_color: Manually verified jersey color.
        jersey_number: Manually verified jersey number.
        player_name: Optional roster name used only as metadata.
        evidence_note: Human-readable explanation of the manual boundary.
    """

    raw_track_id: str
    output_track_id: str
    start_frame: int
    end_frame: int
    jersey_color: str
    jersey_number: str
    player_name: str | None
    evidence_note: str


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    """Read a UTF-8 JSON object from disk.

    Args:
        path: Input JSON path.
        description: Human-readable name used in error messages.

    Returns:
        Parsed top-level JSON object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON root is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{description} root must be a JSON object: {path}")
    return document


def is_valid_bbox(value: Any) -> bool:
    """Return whether a value is a positive finite ``xywh`` bounding box."""
    if not isinstance(value, list) or len(value) != 4:
        return False
    if any(isinstance(item, bool) for item in value):
        return False
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in (x, y, width, height))
        and width > 1
        and height > 1
    )


def build_segment_document(
    raw_tracks: Mapping[str, Any],
    clean_tracks: Mapping[str, Any],
    spec: SegmentSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a one-player clean trajectory document for controlled inference.

    Args:
        raw_tracks: Original SAM3 raw-trajectory document.
        clean_tracks: Existing Qwen-cleaned document used only for its selected
            basketball trajectory.
        spec: Manual segment and identity definition.

    Returns:
        A pair containing the temporary clean JSON and audit statistics.

    Raises:
        KeyError: If the selected player or cleaned ball is unavailable.
        ValueError: If the frame interval or trajectory is invalid.
    """
    if not spec.output_track_id.startswith("player_"):
        raise ValueError("output_track_id must start with 'player_'")
    if spec.start_frame < 0 or spec.end_frame < spec.start_frame:
        raise ValueError("segment frame range must satisfy 0 <= start <= end")
    if not spec.jersey_color.strip():
        raise ValueError("jersey_color must not be empty")
    if re.fullmatch(r"\d{1,2}", spec.jersey_number) is None:
        raise ValueError("jersey_number must be a one- or two-digit string")

    raw_payload = raw_tracks.get(spec.raw_track_id)
    if not isinstance(raw_payload, Mapping):
        raise KeyError(f"raw track does not exist: {spec.raw_track_id}")
    trajectory = raw_payload.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"raw track has no trajectory: {spec.raw_track_id}")
    if spec.end_frame >= len(trajectory):
        raise ValueError(
            f"end_frame {spec.end_frame} exceeds trajectory index "
            f"{len(trajectory) - 1}"
        )

    segment_trajectory = [
        (
            bbox
            if spec.start_frame <= index <= spec.end_frame and is_valid_bbox(bbox)
            else None
        )
        for index, bbox in enumerate(trajectory)
    ]
    retained_frames = [
        index for index, bbox in enumerate(segment_trajectory) if bbox is not None
    ]
    if not retained_frames:
        raise ValueError("selected frame range contains no valid source boxes")

    ball_payload = clean_tracks.get("ball")
    if not isinstance(ball_payload, Mapping):
        raise KeyError("clean trajectory JSON does not contain the selected ball")
    ball_trajectory = ball_payload.get("trajectory")
    if not isinstance(ball_trajectory, list) or not ball_trajectory:
        raise ValueError("cleaned ball trajectory is empty or invalid")

    player_payload = {
        "source_track_id": spec.raw_track_id,
        "jersey_color": spec.jersey_color,
        "jersey_number": spec.jersey_number,
        "player_name": spec.player_name,
        "trajectory": segment_trajectory,
        "segment_validation": {
            "method": "manual_stable_interval",
            "start_frame_inclusive": spec.start_frame,
            "end_frame_inclusive": spec.end_frame,
            "evidence_note": spec.evidence_note,
        },
    }
    document = {
        spec.output_track_id: player_payload,
        "ball": dict(ball_payload),
    }
    statistics = {
        "source_trajectory_length": len(trajectory),
        "source_valid_bbox_count": sum(is_valid_bbox(item) for item in trajectory),
        "retained_valid_bbox_count": len(retained_frames),
        "first_retained_frame": retained_frames[0],
        "last_retained_frame": retained_frames[-1],
        "ball_trajectory_length": len(ball_trajectory),
    }
    return document, statistics


def normalize_event_name(value: str) -> str:
    """Return the canonical BasketEvent label for a user-facing event name."""
    normalized = " ".join(value.strip().lower().split())
    aliases = {
        "assist": "ast",
        "assisted": "ast",
        "ast": "ast",
    }
    return aliases.get(normalized, value.strip())


def evaluate_prediction_report(
    report: Mapping[str, Any], output_track_id: str, expected_event: str
) -> dict[str, Any]:
    """Compare one player prediction with the expected event.

    Args:
        report: Prediction JSON exported by the event-recognition module.
        output_track_id: Temporary player key to locate.
        expected_event: Canonical label or human-readable alias such as
            ``Assist``.

    Returns:
        JSON-compatible comparison containing actual label and confidence.
    """
    canonical_expected = normalize_event_name(expected_event)
    predictions = report.get("player_predictions", [])
    prediction = next(
        (
            item
            for item in predictions
            if isinstance(item, Mapping)
            and str(item.get("player_id")) == output_track_id
        ),
        None,
    )
    if prediction is None:
        return {
            "status": "missing_player_prediction",
            "expected_event": canonical_expected,
            "actual_event": None,
            "confidence": None,
            "matched": False,
        }
    actual_event = str(prediction.get("event"))
    return {
        "status": "completed",
        "expected_event": canonical_expected,
        "actual_event": actual_event,
        "confidence": prediction.get("confidence"),
        "matched": actual_event == canonical_expected,
    }


def write_json(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 JSON and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_inference_command(
    args: argparse.Namespace,
    trajectory_path: Path,
    prediction_path: Path,
) -> list[str]:
    """构造事件识别模块的受控实验命令。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "src.modules.event_recognition.inference",
        "--video",
        str(args.video),
        "--traj_json",
        str(trajectory_path),
        "--checkpoint",
        str(args.checkpoint),
        "--timesformer_model",
        str(args.timesformer_model),
        "--gpu_id",
        str(args.gpu_id),
        "--player_ids",
        args.output_track_id,
        "--bag_clips",
        str(args.bag_clips),
        "--clip_len",
        str(args.clip_len),
        "--fps_in",
        str(args.fps_in),
        "--fps_out",
        str(args.fps_out),
        "--img_size",
        str(args.img_size),
        "--topk",
        str(args.topk),
        "--prediction_json_path",
        str(prediction_path),
        "--timeline_topk",
        str(args.timeline_topk),
    ]


def run_logged_command(command: Sequence[str], log_file: TextIO) -> int:
    """Run a subprocess while mirroring merged output to terminal and log."""
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
        log_file.flush()
    return process.wait()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one stable-segment experiment."""
    from src.core.config import SETTINGS

    parser = argparse.ArgumentParser(
        description=(
            "Mask one manually audited SAM3 track interval and test its "
            "PlayNet event prediction."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--clean-json", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "tests" / "track_segment_runtime",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--raw-track-id", default="player_8")
    parser.add_argument("--output-track-id", default="player_20_segment")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--jersey-color", required=True)
    parser.add_argument("--jersey-number", required=True)
    parser.add_argument("--player-name", default=None)
    parser.add_argument(
        "--evidence-note",
        default="Manually audited stable identity interval.",
    )
    parser.add_argument("--expected-event", default="Assist")
    parser.add_argument("--checkpoint", type=Path, default=SETTINGS.event_checkpoint)
    parser.add_argument(
        "--timesformer-model", type=Path, default=SETTINGS.timesformer_model
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--bag-clips", type=int, default=12)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--fps-in", type=int, default=60)
    parser.add_argument("--fps-out", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeline-topk", type=int, default=2)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the controlled trajectory and manifest without PlayNet.",
    )
    return parser.parse_args(argv)


def _require_file(path: Path, description: str) -> Path:
    """Return a verified file path or raise ``FileNotFoundError``."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _require_directory(path: Path, description: str) -> Path:
    """Return a verified directory path or raise ``NotADirectoryError``."""
    if not path.is_dir():
        raise NotADirectoryError(f"{description} not found: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare the stable segment, optionally run PlayNet, and save evidence."""
    args = parse_args(argv)
    _require_file(args.video, "input video")
    raw_tracks = read_json_object(args.raw_json, "raw trajectory JSON")
    clean_tracks = read_json_object(args.clean_json, "clean trajectory JSON")

    run_directory = args.output_root.expanduser() / args.run_name
    if run_directory.exists():
        raise FileExistsError(
            f"validation run already exists; choose another --run-name: {run_directory}"
        )
    run_directory.mkdir(parents=True)
    trajectory_path = run_directory / "segment_tracks.json"
    prediction_path = run_directory / "prediction.json"
    manifest_path = run_directory / "manifest.json"
    log_path = run_directory / "inference.log"

    spec = SegmentSpec(
        raw_track_id=args.raw_track_id,
        output_track_id=args.output_track_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        jersey_color=args.jersey_color.strip().lower(),
        jersey_number=args.jersey_number.strip(),
        player_name=args.player_name,
        evidence_note=args.evidence_note,
    )
    segment_document, statistics = build_segment_document(
        raw_tracks, clean_tracks, spec
    )
    write_json(trajectory_path, segment_document)

    manifest: dict[str, Any] = {
        "schema_version": "basketevent_track_segment_validation.v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "prepared",
        "video": str(args.video.resolve()),
        "raw_json": str(args.raw_json.resolve()),
        "clean_json": str(args.clean_json.resolve()),
        "trajectory_json": str(trajectory_path.resolve()),
        "prediction_json": str(prediction_path.resolve()),
        "segment": asdict(spec),
        "statistics": statistics,
        "expected_event": normalize_event_name(args.expected_event),
        "boundary_is_manual_assumption": True,
    }
    write_json(manifest_path, manifest)

    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0

    _require_file(args.checkpoint, "PlayNet checkpoint")
    _require_directory(args.timesformer_model, "TimeSformer model")
    command = build_inference_command(args, trajectory_path, prediction_path)
    manifest["inference_command"] = command
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    with log_path.open("w", encoding="utf-8") as log_file:
        return_code = run_logged_command(command, log_file)
    manifest["inference_return_code"] = return_code
    if return_code != 0:
        manifest["status"] = "inference_failed"
        write_json(manifest_path, manifest)
        return return_code
    if not prediction_path.is_file():
        manifest["status"] = "prediction_missing"
        write_json(manifest_path, manifest)
        return 1

    prediction_report = read_json_object(prediction_path, "prediction report")
    evaluation = evaluate_prediction_report(
        prediction_report, spec.output_track_id, args.expected_event
    )
    manifest["status"] = "completed"
    manifest["evaluation"] = evaluation
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
