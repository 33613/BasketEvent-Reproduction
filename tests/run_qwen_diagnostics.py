"""Diagnose Qwen player filtering and jersey recognition on SAM3 tracks.

This script is intentionally separate from ``recognize.py``.  It runs the
current one-pass prompt, a decomposed multi-image prompt, and an independent
per-timestamp prompt on audited player crops, then records every input and output below
``tests/qwen_tests_runtime``.  The resulting audit bundle makes it possible to
distinguish on-court validation, jersey recognition, output-contract failures,
deterministic roster lookup, and temporal identity switches.

Runtime images and model responses are local experiment artifacts and are not
committed to Git.  The directory layout and command examples are documented in
``tests/qwen_tests_runtime/README.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Direct execution sets sys.path[0] to tests/, so expose repository modules
    # such as settings.py and recognize.py without relying on PYTHONPATH.
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CropRecord:
    """Describe one crop supplied to Qwen.

    Attributes:
        image_index: One-based image position used in the prompt.
        frame_index: Zero-based source-video frame index.
        bbox_xywh: SAM3 bounding box in source-video coordinates.
        image_path: Full-body crop path relative to the track directory.
        torso_image_path: Upper-body crop used to inspect the jersey.
        paired_image_path: Side-by-side full-body and jersey-region image used
            by the temporal diagnostic prompt.
        width: Saved crop width in pixels.
        height: Saved crop height in pixels.
        brightness_mean: Mean grayscale intensity in the range 0--255.
        sharpness_laplacian_variance: Variance of the grayscale Laplacian.
            Larger values generally indicate sharper crops, but the metric is
            diagnostic rather than a correctness threshold.
        quality_score: Diagnostic score combining crop area and sharpness.  It
            is used only to choose one observation inside each temporal bin.
    """

    image_index: int
    frame_index: int
    bbox_xywh: list[float]
    image_path: str
    torso_image_path: str
    paired_image_path: str
    width: int
    height: int
    brightness_mean: float
    sharpness_laplacian_variance: float
    quality_score: float


@dataclass(frozen=True)
class RosterIndex:
    """Store deterministic jersey-color and jersey-number roster mappings."""

    allowed_colors: tuple[str, ...]
    identity_to_names: Mapping[tuple[str, str], tuple[str, ...]]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "RosterIndex":
        """Build an index from a BasketEvent recognition-roster document.

        Args:
            document: Parsed roster JSON containing ``jersey_color`` and
                ``players`` fields.

        Returns:
            Normalized roster index.  Jersey ``0`` and ``00`` remain distinct.
        """
        team_colors = {
            str(team): normalize_color(color)
            for team, color in document.get("jersey_color", {}).items()
        }
        grouped: dict[tuple[str, str], list[str]] = {}
        for player in document.get("players", []):
            if not isinstance(player, Mapping):
                continue
            color = team_colors.get(str(player.get("team_name")))
            number = normalize_jersey_number(player.get("jersey"))
            name = str(player.get("name", "")).strip()
            if color is None or number is None or not name:
                continue
            grouped.setdefault((color, number), []).append(name)
        stable = {
            key: tuple(sorted(set(names))) for key, names in sorted(grouped.items())
        }
        colors = tuple(sorted({key[0] for key in stable}))
        return cls(allowed_colors=colors, identity_to_names=stable)

    def lookup(self, color: Any, number: Any) -> dict[str, Any]:
        """Look up an identity without asking Qwen to guess a player name.

        Args:
            color: Qwen jersey-color prediction.
            number: Qwen jersey-number prediction.

        Returns:
            JSON-compatible lookup result with a stable status.
        """
        normalized_color = normalize_color(color)
        normalized_number = normalize_jersey_number(number)
        result: dict[str, Any] = {
            "jersey_color": normalized_color,
            "jersey_number": normalized_number,
            "status": "invalid_identity",
            "player_name": None,
            "candidate_names": [],
        }
        if normalized_color is None:
            result["reason"] = "jersey color is missing or invalid"
            return result
        if normalized_number is None:
            result["reason"] = "jersey number is missing or invalid"
            return result

        names = list(
            self.identity_to_names.get((normalized_color, normalized_number), ())
        )
        result["candidate_names"] = names
        if not names:
            result["status"] = "no_match"
            result["reason"] = "no roster row matches the exact color-number key"
        elif len(names) > 1:
            result["status"] = "ambiguous"
            result["reason"] = "multiple roster rows share the color-number key"
        else:
            result["status"] = "matched"
            result["player_name"] = names[0]
            result["reason"] = None
        return result


def normalize_color(value: Any) -> str | None:
    """Return a lowercase whitespace-normalized color or ``None``."""
    if value is None:
        return None
    normalized = " ".join(str(value).strip().lower().split())
    return normalized or None


def normalize_jersey_number(value: Any) -> str | None:
    """Return a one- or two-digit jersey string while preserving ``0``/``00``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, float) and value.is_integer():
        normalized = str(int(value))
    else:
        normalized = str(value).strip()
    return normalized if re.fullmatch(r"\d{1,2}", normalized) else None


def parse_json_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a Qwen JSON object while tolerating Markdown code fences.

    Args:
        text: Decoded model response.

    Returns:
        A pair containing the parsed object and an error string.  Exactly one
        element of the pair is non-``None``.
    """
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end >= start:
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as error:
        return None, f"{error.msg} at line {error.lineno} column {error.colno}"
    if not isinstance(parsed, dict):
        return None, "Qwen response is valid JSON but is not an object"
    return parsed, None


def build_decomposed_prompt(allowed_colors: Sequence[str]) -> str:
    """Build a prompt that separates validity from jersey readability.

    Args:
        allowed_colors: Jersey-color vocabulary derived from the game roster.

    Returns:
        Prompt requesting independent validity and identity observations.
    """
    color_text = ", ".join(allowed_colors) if allowed_colors else "unknown"
    return f"""You are given chronological crops from ONE SAM3 track in a basketball broadcast.

Diagnose the track in separate stages. Do not reject a real on-court player merely because the jersey number is unreadable.

Stage 1 - on-court validity:
- Decide whether the track mostly follows a real player currently participating on the court.
- Referees, spectators, coaches, staff, bench-only people, empty boxes, and tracks that switch between different people are invalid.
- Jersey-number readability is NOT evidence that a real player is invalid.

Stage 2 - visual jersey identity:
- For every crop, independently record whether a number is readable.
- Aggregate only mutually consistent visual evidence across crops.
- jersey_color must be one of: {color_text}; otherwise use null.
- jersey_number must be a visible one- or two-digit string. Preserve "0" and "00" as different values.
- If evidence conflicts or is insufficient, return null instead of guessing.
- Do not output or infer a player name. Python will perform roster lookup later.

Return STRICT JSON only:
{{
  "is_on_court_player": true,
  "validity_confidence": 0.0,
  "validity_reason": "short explanation independent of number readability",
  "same_identity_across_crops": true,
  "crop_observations": [
    {{
      "image_index": 1,
      "shows_on_court_player": true,
      "number_readable": false,
      "jersey_number": null,
      "jersey_color": "white",
      "note": "short visual observation"
    }}
  ],
  "jersey_number": null,
  "jersey_color": "white",
  "identity_confidence": 0.0,
  "identity_reason": "which crop indices support the final color and number"
}}

Use confidence values between 0 and 1. Include one crop_observations entry for every supplied image.
"""


def build_temporal_observation_prompt(allowed_colors: Sequence[str]) -> str:
    """Build the strict prompt used to inspect one timestamp independently.

    Args:
        allowed_colors: Jersey-color vocabulary derived from the game roster.

    Returns:
        Prompt for a paired image whose left side is the full player crop and
        whose right side is an enlarged upper-body jersey crop.
    """
    color_text = ", ".join(allowed_colors) if allowed_colors else "unknown"
    return f"""Inspect ONE timestamp from ONE SAM3 basketball-player track.

The supplied image is a diagnostic pair:
- LEFT: the complete SAM3 player crop, used to decide whether this is an on-court player;
- RIGHT: an enlarged upper-body crop, used only for jersey color and number reading.

Report only what is visually supported at this timestamp. Do not infer a player name. Do not use nearby players' numbers. If the number is not readable, use null. Allowed jersey colors: {color_text}.

Return STRICT JSON only, with no note, explanation, or Markdown:
{{
  "is_on_court_player": true,
  "on_court_confidence": 0.0,
  "jersey_color_candidate": "white",
  "color_confidence": 0.0,
  "number_readable": true,
  "jersey_number_candidate": "13",
  "number_confidence": 0.0
}}

Confidence values must be between 0 and 1. A readable number must be a visible one- or two-digit string; preserve "0" and "00" as different values. If number_readable is false, jersey_number_candidate must be null and number_confidence must be 0.
"""


def _confidence(value: Any) -> float | None:
    """Normalize a model confidence value to the inclusive range 0--1."""
    if isinstance(value, bool):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


def find_decomposed_output_inconsistencies(
    parsed: Mapping[str, Any] | None,
) -> list[str]:
    """Find contradictions in the original multi-image decomposed response.

    Args:
        parsed: Parsed response from :func:`build_decomposed_prompt`.

    Returns:
        Stable machine-readable contradiction descriptions.  In particular,
        this catches the observed case where free text says ``number 13`` but
        the structured number field is null and ``number_readable`` is false.
    """
    if parsed is None:
        return []
    inconsistencies: list[str] = []
    for observation in parsed.get("crop_observations", []):
        if not isinstance(observation, Mapping):
            continue
        image_index = observation.get("image_index")
        number = normalize_jersey_number(observation.get("jersey_number"))
        readable = observation.get("number_readable")
        note = str(observation.get("note", ""))
        note_numbers = re.findall(r"\b(?:number|#)\s*(\d{1,2})\b", note, re.I)
        if readable is False and number is not None:
            inconsistencies.append(
                f"image {image_index}: unreadable flag conflicts with number {number}"
            )
        if note_numbers and number is None:
            inconsistencies.append(
                f"image {image_index}: note mentions number {note_numbers[0]} but structured number is null"
            )
    return inconsistencies


def validate_temporal_observation(
    parsed: Mapping[str, Any] | None,
    parse_error: str | None,
    allowed_colors: Sequence[str],
) -> dict[str, Any]:
    """Validate and normalize one timestamp-level Qwen response.

    Args:
        parsed: Parsed strict-JSON response, or ``None`` after a parse error.
        parse_error: JSON parsing error, if any.
        allowed_colors: Roster-derived allowed jersey colors.

    Returns:
        A normalized observation with ``status`` set to ``valid``,
        ``unresolved``, ``model_output_inconsistency``, or ``json_parse``.
    """
    normalized: dict[str, Any] = {
        "status": "json_parse" if parse_error else "unresolved",
        "issues": [],
        "is_on_court_player": None,
        "jersey_color": None,
        "jersey_number": None,
        "number_confidence": None,
    }
    if parse_error is not None or parsed is None:
        normalized["issues"] = [parse_error or "missing parsed response"]
        return normalized

    on_court = parsed.get("is_on_court_player")
    readable = parsed.get("number_readable")
    color = normalize_color(parsed.get("jersey_color_candidate"))
    number = normalize_jersey_number(parsed.get("jersey_number_candidate"))
    number_confidence = _confidence(parsed.get("number_confidence"))
    issues: list[str] = []
    if not isinstance(on_court, bool):
        issues.append("is_on_court_player must be boolean")
    if not isinstance(readable, bool):
        issues.append("number_readable must be boolean")
    if color is not None and color not in set(allowed_colors):
        issues.append(f"jersey color {color!r} is outside the roster vocabulary")
        color = None
    if number_confidence is None:
        issues.append("number_confidence must be between 0 and 1")
    if readable is False and number is not None:
        issues.append("number_readable=false conflicts with a number candidate")
    if readable is True and number is None:
        issues.append("number_readable=true requires a valid number candidate")
    if readable is False and number_confidence not in (None, 0.0):
        issues.append("an unreadable number must have zero confidence")

    normalized.update(
        {
            "is_on_court_player": on_court,
            "jersey_color": color,
            "jersey_number": number,
            "number_confidence": number_confidence,
            "issues": issues,
        }
    )
    if issues:
        normalized["status"] = "model_output_inconsistency"
    elif on_court is True and readable is True and number is not None:
        normalized["status"] = "valid"
    else:
        normalized["status"] = "unresolved"
    return normalized


def aggregate_temporal_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    minimum_confidence: float = 0.60,
) -> dict[str, Any]:
    """Aggregate independent number observations without hiding conflicts.

    Args:
        observations: Chronological timestamp results.  Each item contains a
            ``frame_index`` and normalized ``validation`` mapping.
        minimum_confidence: Minimum number confidence included as evidence.

    Returns:
        Track-level classification.  Multiple well-supported number candidates
        produce ``mixed_identity`` rather than a majority-vote identity.  This
        preserves evidence of a SAM3 identity switch such as number 13 changing
        to number 20.
    """
    evidence: list[dict[str, Any]] = []
    inconsistency_count = 0
    for item in sorted(observations, key=lambda value: int(value["frame_index"])):
        validation = item.get("validation")
        if not isinstance(validation, Mapping):
            continue
        if validation.get("status") == "model_output_inconsistency":
            inconsistency_count += 1
        confidence = _confidence(validation.get("number_confidence"))
        number = normalize_jersey_number(validation.get("jersey_number"))
        color = normalize_color(validation.get("jersey_color"))
        if (
            validation.get("status") == "valid"
            and confidence is not None
            and confidence >= minimum_confidence
            and number is not None
        ):
            evidence.append(
                {
                    "frame_index": int(item["frame_index"]),
                    "jersey_color": color,
                    "jersey_number": number,
                    "confidence": confidence,
                }
            )

    grouped: dict[tuple[str | None, str], list[dict[str, Any]]] = {}
    for item in evidence:
        key = (item["jersey_color"], item["jersey_number"])
        grouped.setdefault(key, []).append(item)
    candidates = [
        {
            "jersey_color": color,
            "jersey_number": number,
            "support_count": len(items),
            "confidence_sum": round(sum(item["confidence"] for item in items), 4),
            "frames": [item["frame_index"] for item in items],
        }
        for (color, number), items in grouped.items()
    ]
    candidates.sort(key=lambda item: (-item["confidence_sum"], item["jersey_number"]))

    segments: list[dict[str, Any]] = []
    for item in evidence:
        identity = (item["jersey_color"], item["jersey_number"])
        if segments and segments[-1]["identity"] == identity:
            segments[-1]["end_frame"] = item["frame_index"]
            segments[-1]["observation_frames"].append(item["frame_index"])
        else:
            segments.append(
                {
                    "identity": identity,
                    "start_frame": item["frame_index"],
                    "end_frame": item["frame_index"],
                    "observation_frames": [item["frame_index"]],
                }
            )
    serialized_segments = [
        {
            **{key: value for key, value in segment.items() if key != "identity"},
            "jersey_color": segment["identity"][0],
            "jersey_number": segment["identity"][1],
        }
        for segment in segments
    ]

    if len(candidates) >= 2:
        status = "mixed_identity"
        selected_identity = None
    elif len(candidates) == 1:
        status = "stable_identity"
        selected_identity = {
            "jersey_color": candidates[0]["jersey_color"],
            "jersey_number": candidates[0]["jersey_number"],
        }
    elif inconsistency_count:
        status = "model_output_inconsistency"
        selected_identity = None
    else:
        status = "unresolved"
        selected_identity = None
    return {
        "status": status,
        "minimum_confidence": minimum_confidence,
        "selected_identity": selected_identity,
        "identity_candidates": candidates,
        "proposed_observation_segments": serialized_segments,
        "valid_evidence_count": len(evidence),
        "model_output_inconsistency_count": inconsistency_count,
    }


def classify_failure_stage(
    parsed: Mapping[str, Any] | None,
    parse_error: str | None,
    roster_lookup: Mapping[str, Any] | None,
) -> str:
    """Return the first failed stage of the decomposed diagnostic pipeline."""
    if parse_error is not None or parsed is None:
        return "json_parse"
    if find_decomposed_output_inconsistencies(parsed):
        return "model_output_inconsistency"
    if parsed.get("is_on_court_player") is not True:
        return "on_court_validation"
    if normalize_color(parsed.get("jersey_color")) is None:
        return "jersey_color"
    if normalize_jersey_number(parsed.get("jersey_number")) is None:
        return "jersey_number"
    if roster_lookup is None or roster_lookup.get("status") != "matched":
        return "roster_lookup"
    return "complete"


def _uniform_indices(length: int, count: int) -> list[int]:
    """Return unique rounded uniform indices without requiring NumPy."""
    if length <= 0 or count <= 0:
        return []
    if length == 1 or count == 1:
        return [0]
    effective_count = min(length, count)
    raw = [
        round(position * (length - 1) / (effective_count - 1))
        for position in range(effective_count)
    ]
    return list(dict.fromkeys(raw))


def _make_paired_image(full_image: Any, torso_image: Any) -> Any:
    """Create one labeled full-body/jersey-region diagnostic image."""
    from PIL import Image, ImageDraw, ImageOps

    panel_width = 320
    panel_height = 420
    header_height = 34
    pair = Image.new(
        "RGB", (panel_width * 2, panel_height + header_height), "white"
    )
    draw = ImageDraw.Draw(pair)
    draw.text((8, 9), "FULL PLAYER", fill="black")
    draw.text((panel_width + 8, 9), "JERSEY REGION", fill="black")
    for column, image in enumerate((full_image, torso_image)):
        contained = ImageOps.contain(image, (panel_width - 12, panel_height - 12))
        left = column * panel_width + (panel_width - contained.width) // 2
        top = header_height + (panel_height - contained.height) // 2
        pair.paste(contained, (left, top))
    draw.line(
        (panel_width, 0, panel_width, panel_height + header_height),
        fill="gray",
        width=2,
    )
    return pair


def extract_track_crops(
    video_path: Path,
    trajectory: Sequence[Any],
    track_directory: Path,
    *,
    crop_count: int,
    pad_ratio: float,
) -> tuple[list[Any], list[Any], list[CropRecord]]:
    """Extract unique quality-aware crops distributed across the trajectory.

    Args:
        video_path: Source broadcast clip.
        trajectory: SAM3 ``xywh`` trajectory indexed by source frame.
        track_directory: Per-track diagnostic output directory.
        crop_count: Maximum number of chronological observations supplied to
            Qwen. Fewer are returned when the track has fewer valid frames;
            no image is repeated to fill the requested count.
        pad_ratio: Fractional context padding around each bounding box.

    Returns:
        Full-player images, paired full-player/jersey images, and metadata.

    Raises:
        RuntimeError: If the video cannot be opened or no valid crop exists.
    """
    import cv2
    import numpy as np
    from PIL import Image

    crops_directory = track_directory / "crops"
    torsos_directory = track_directory / "torso_crops"
    pairs_directory = track_directory / "paired_observations"
    for directory in (crops_directory, torsos_directory, pairs_directory):
        directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    valid_frame_indices = [
        frame_index
        for frame_index, bbox in enumerate(trajectory)
        if isinstance(bbox, Sequence)
        and len(bbox) == 4
        and float(bbox[2]) > 1
        and float(bbox[3]) > 1
    ]
    candidate_positions = _uniform_indices(
        len(valid_frame_indices), min(len(valid_frame_indices), crop_count * 4)
    )
    candidate_frame_indices = [
        valid_frame_indices[position] for position in candidate_positions
    ]
    candidates: list[dict[str, Any]] = []
    for frame_index in candidate_frame_indices:
        bbox = trajectory[frame_index]
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        x, y, width, height = (float(value) for value in bbox)
        if width <= 1 or height <= 1:
            continue
        frame_height, frame_width = frame.shape[:2]
        pad_width = width * pad_ratio
        pad_height = height * pad_ratio
        x1 = max(0, int(x - pad_width))
        y1 = max(0, int(y - pad_height))
        x2 = min(frame_width, int(x + width + pad_width))
        y2 = min(frame_height, int(y + height + pad_height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        full_image = Image.fromarray(crop_rgb).convert("RGB")
        crop_width, crop_height = full_image.size
        torso_box = (
            int(crop_width * 0.08),
            int(crop_height * 0.02),
            max(int(crop_width * 0.92), int(crop_width * 0.08) + 1),
            max(int(crop_height * 0.70), int(crop_height * 0.02) + 1),
        )
        torso_image = full_image.crop(torso_box)
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        quality_score = math.sqrt(max(1, crop_width * crop_height)) * math.log1p(
            min(sharpness, 5000.0)
        )
        candidates.append(
            {
                "frame_index": int(frame_index),
                "bbox_xywh": [x, y, width, height],
                "full_image": full_image,
                "torso_image": torso_image,
                "width": crop_width,
                "height": crop_height,
                "brightness_mean": float(np.mean(grayscale)),
                "sharpness": sharpness,
                "quality_score": quality_score,
            }
        )
    capture.release()

    if not candidates:
        raise RuntimeError("No valid crops could be extracted from the trajectory")

    selected: list[dict[str, Any]] = []
    bin_count = min(crop_count, len(candidates))
    for bin_index in range(bin_count):
        start = bin_index * len(candidates) // bin_count
        end = (bin_index + 1) * len(candidates) // bin_count
        bin_candidates = candidates[start:end]
        selected.append(
            max(bin_candidates, key=lambda item: float(item["quality_score"]))
        )
    selected.sort(key=lambda item: int(item["frame_index"]))

    full_images: list[Any] = []
    paired_images: list[Any] = []
    records: list[CropRecord] = []
    for image_index, candidate in enumerate(selected, start=1):
        frame_index = int(candidate["frame_index"])
        filename = f"image_{image_index:02d}_frame_{frame_index:06d}.jpg"
        torso_filename = (
            f"torso_{image_index:02d}_frame_{frame_index:06d}.jpg"
        )
        pair_filename = f"pair_{image_index:02d}_frame_{frame_index:06d}.jpg"
        full_image = candidate["full_image"]
        torso_image = candidate["torso_image"]
        paired_image = _make_paired_image(full_image, torso_image)
        full_image.save(crops_directory / filename, quality=95)
        torso_image.save(torsos_directory / torso_filename, quality=95)
        paired_image.save(pairs_directory / pair_filename, quality=95)
        full_images.append(full_image)
        paired_images.append(paired_image)
        records.append(
            CropRecord(
                image_index=image_index,
                frame_index=frame_index,
                bbox_xywh=list(candidate["bbox_xywh"]),
                image_path=str(Path("crops") / filename),
                torso_image_path=str(Path("torso_crops") / torso_filename),
                paired_image_path=str(Path("paired_observations") / pair_filename),
                width=int(candidate["width"]),
                height=int(candidate["height"]),
                brightness_mean=float(candidate["brightness_mean"]),
                sharpness_laplacian_variance=float(candidate["sharpness"]),
                quality_score=float(candidate["quality_score"]),
            )
        )
    return full_images, paired_images, records


def save_contact_sheet(
    images: Sequence[Any], records: Sequence[CropRecord], output_path: Path
) -> None:
    """Save a labeled overview of every Qwen input crop."""
    from PIL import Image, ImageDraw, ImageOps

    columns = 5
    cell_width = 240
    cell_height = 220
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for offset, (image, record) in enumerate(zip(images, records)):
        column = offset % columns
        row = offset // columns
        thumbnail = ImageOps.contain(image, (cell_width - 12, cell_height - 42))
        left = column * cell_width + (cell_width - thumbnail.width) // 2
        top = row * cell_height + 28
        sheet.paste(thumbnail, (left, top))
        label = (
            f"image {record.image_index} | frame {record.frame_index} | "
            f"sharp {record.sharpness_laplacian_variance:.1f}"
        )
        draw.text((column * cell_width + 6, row * cell_height + 6), label, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def load_qwen_model(model_path: Path, device: str) -> tuple[Any, Any]:
    """Load the same local 4-bit Qwen model configuration as ``recognize.py``."""
    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen runtime diagnostics require a CUDA GPU")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=quantization,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True
    )
    return model, processor


def generate_response(
    model: Any,
    processor: Any,
    images: Sequence[Any],
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> str:
    """Run one deterministic Qwen generation over a chronological crop set."""
    import torch
    from qwen_vl_utils import process_vision_info

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def write_json(path: Path, value: Any) -> None:
    """Write stable UTF-8 JSON and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one diagnostic run."""
    from settings import SETTINGS

    parser = argparse.ArgumentParser(
        description="Audit Qwen validity, jersey OCR, and roster lookup per SAM3 track."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--bbox-json", type=Path, required=True)
    parser.add_argument("--roster-json", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, default=SETTINGS.qwen_model)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "qwen_tests_runtime",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional stable output directory name; defaults to clip plus UTC time.",
    )
    parser.add_argument(
        "--track-id",
        action="append",
        default=[],
        help="Raw SAM3 player ID to inspect; repeat this option or omit for all.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "legacy", "decomposed", "temporal", "all"),
        default="both",
        help=(
            "both runs the two historical prompts; temporal runs independent "
            "per-timestamp OCR; all runs every diagnostic."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-crops", type=int, default=10)
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--minimum-number-confidence",
        type=float,
        default=0.60,
        help="Minimum timestamp-level confidence used as identity evidence.",
    )
    return parser.parse_args(argv)


def _require_runtime_inputs(args: argparse.Namespace) -> None:
    """Validate input paths and numeric options before loading Qwen."""
    for description, path in (
        ("video", args.video),
        ("SAM3 trajectory JSON", args.bbox_json),
        ("roster JSON", args.roster_json),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")
    if not args.qwen_model.is_dir():
        raise NotADirectoryError(f"Qwen model directory not found: {args.qwen_model}")
    if args.num_crops <= 0:
        raise ValueError("--num-crops must be positive")
    if args.pad_ratio < 0:
        raise ValueError("--pad-ratio must be non-negative")
    if not 0.0 <= args.minimum_number_confidence <= 1.0:
        raise ValueError("--minimum-number-confidence must be between 0 and 1")


def main(argv: Sequence[str] | None = None) -> int:
    """Run per-track legacy/decomposed Qwen diagnostics and save an audit bundle."""
    args = parse_args(argv)
    _require_runtime_inputs(args)

    raw_tracks = json.loads(args.bbox_json.read_text(encoding="utf-8"))
    roster_document = json.loads(args.roster_json.read_text(encoding="utf-8"))
    if not isinstance(raw_tracks, dict):
        raise ValueError("SAM3 trajectory JSON root must be an object")
    if not isinstance(roster_document, dict):
        raise ValueError("Roster JSON root must be an object")
    roster_index = RosterIndex.from_document(roster_document)

    available_tracks = sorted(
        track_id
        for track_id, payload in raw_tracks.items()
        if str(track_id).startswith("player_") and isinstance(payload, Mapping)
    )
    selected_tracks = list(dict.fromkeys(args.track_id)) or available_tracks
    missing_tracks = sorted(set(selected_tracks) - set(available_tracks))
    if missing_tracks:
        raise KeyError(f"Requested SAM3 tracks do not exist: {missing_tracks}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.video.stem}_{timestamp}"
    run_directory = args.output_root / run_name
    if run_directory.exists():
        raise FileExistsError(
            f"Diagnostic run already exists; choose another --run-name: {run_directory}"
        )
    run_directory.mkdir(parents=True)

    manifest = {
        "schema_version": "basketevent_qwen_diagnostic.v1",
        "created_utc": timestamp,
        "video": str(args.video.resolve()),
        "bbox_json": str(args.bbox_json.resolve()),
        "roster_json": str(args.roster_json.resolve()),
        "qwen_model": str(args.qwen_model.resolve()),
        "mode": args.mode,
        "device": args.device,
        "num_crops": args.num_crops,
        "pad_ratio": args.pad_ratio,
        "minimum_number_confidence": args.minimum_number_confidence,
        "selected_track_ids": selected_tracks,
        "available_track_ids": available_tracks,
        "allowed_jersey_colors": list(roster_index.allowed_colors),
    }
    write_json(run_directory / "manifest.json", manifest)

    model, processor = load_qwen_model(args.qwen_model, args.device)
    legacy_prompt: str | None = None
    if args.mode in {"both", "legacy", "all"}:
        from recognize import build_onepass_prompt, build_roster_text

        legacy_prompt = build_onepass_prompt(build_roster_text(str(args.roster_json)))
    decomposed_prompt = build_decomposed_prompt(roster_index.allowed_colors)
    temporal_prompt = build_temporal_observation_prompt(
        roster_index.allowed_colors
    )

    summaries: list[dict[str, Any]] = []
    for track_id in selected_tracks:
        print(f"[Qwen diagnostic] {track_id}", flush=True)
        track_directory = run_directory / track_id
        track_directory.mkdir()
        trajectory = raw_tracks[track_id].get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            summaries.append(
                {"track_id": track_id, "status": "missing_trajectory"}
            )
            continue

        images, paired_images, crop_records = extract_track_crops(
            args.video,
            trajectory,
            track_directory,
            crop_count=args.num_crops,
            pad_ratio=args.pad_ratio,
        )
        write_json(
            track_directory / "crop_manifest.json",
            {"track_id": track_id, "crops": [asdict(item) for item in crop_records]},
        )
        save_contact_sheet(
            images, crop_records, track_directory / "contact_sheet.jpg"
        )

        result: dict[str, Any] = {"track_id": track_id}
        if legacy_prompt is not None:
            (track_directory / "legacy_prompt.txt").write_text(
                legacy_prompt, encoding="utf-8"
            )
            raw_output = generate_response(
                model,
                processor,
                images,
                legacy_prompt,
                args.device,
                args.max_new_tokens,
            )
            (track_directory / "legacy_raw_output.txt").write_text(
                raw_output, encoding="utf-8"
            )
            parsed, parse_error = parse_json_response(raw_output)
            lookup = None
            if parsed is not None:
                lookup = roster_index.lookup(
                    parsed.get("jersey_color"), parsed.get("jersey_number")
                )
            result["legacy"] = {
                "parsed": parsed,
                "parse_error": parse_error,
                "roster_lookup": lookup,
                "retained_by_current_code": bool(
                    parsed is not None and parsed.get("is_valid_player") is True
                ),
            }

        if args.mode in {"both", "decomposed", "all"}:
            (track_directory / "decomposed_prompt.txt").write_text(
                decomposed_prompt, encoding="utf-8"
            )
            raw_output = generate_response(
                model,
                processor,
                images,
                decomposed_prompt,
                args.device,
                args.max_new_tokens,
            )
            (track_directory / "decomposed_raw_output.txt").write_text(
                raw_output, encoding="utf-8"
            )
            parsed, parse_error = parse_json_response(raw_output)
            lookup = None
            if parsed is not None:
                lookup = roster_index.lookup(
                    parsed.get("jersey_color"), parsed.get("jersey_number")
                )
            result["decomposed"] = {
                "parsed": parsed,
                "parse_error": parse_error,
                "roster_lookup": lookup,
                "output_inconsistencies": (
                    find_decomposed_output_inconsistencies(parsed)
                ),
                "failure_stage": classify_failure_stage(
                    parsed, parse_error, lookup
                ),
            }

        if args.mode in {"temporal", "all"}:
            (track_directory / "temporal_prompt.txt").write_text(
                temporal_prompt, encoding="utf-8"
            )
            temporal_directory = track_directory / "temporal_outputs"
            temporal_directory.mkdir()
            temporal_observations: list[dict[str, Any]] = []
            for paired_image, crop_record in zip(paired_images, crop_records):
                raw_output = generate_response(
                    model,
                    processor,
                    [paired_image],
                    temporal_prompt,
                    args.device,
                    min(args.max_new_tokens, 256),
                )
                output_stem = (
                    f"frame_{crop_record.frame_index:06d}_"
                    f"image_{crop_record.image_index:02d}"
                )
                (temporal_directory / f"{output_stem}_raw.txt").write_text(
                    raw_output, encoding="utf-8"
                )
                parsed, parse_error = parse_json_response(raw_output)
                validation = validate_temporal_observation(
                    parsed, parse_error, roster_index.allowed_colors
                )
                roster_lookup = None
                if validation["status"] == "valid":
                    roster_lookup = roster_index.lookup(
                        validation.get("jersey_color"),
                        validation.get("jersey_number"),
                    )
                observation_result = {
                    "image_index": crop_record.image_index,
                    "frame_index": crop_record.frame_index,
                    "paired_image_path": crop_record.paired_image_path,
                    "parsed": parsed,
                    "parse_error": parse_error,
                    "validation": validation,
                    "roster_lookup": roster_lookup,
                }
                write_json(
                    temporal_directory / f"{output_stem}_result.json",
                    observation_result,
                )
                temporal_observations.append(observation_result)
            temporal_aggregate = aggregate_temporal_observations(
                temporal_observations,
                minimum_confidence=args.minimum_number_confidence,
            )
            result["temporal"] = {
                "observations": temporal_observations,
                "aggregate": temporal_aggregate,
            }

        write_json(track_directory / "result.json", result)
        summary: dict[str, Any] = {"track_id": track_id, "status": "completed"}
        if "legacy" in result:
            summary["legacy_retained"] = result["legacy"][
                "retained_by_current_code"
            ]
        if "decomposed" in result:
            summary["decomposed_failure_stage"] = result["decomposed"][
                "failure_stage"
            ]
        if "temporal" in result:
            summary["temporal_status"] = result["temporal"]["aggregate"][
                "status"
            ]
            summary["temporal_identity_candidates"] = result["temporal"][
                "aggregate"
            ]["identity_candidates"]
        summaries.append(summary)

    summary_document = {
        "schema_version": "basketevent_qwen_diagnostic_summary.v1",
        "run_directory": str(run_directory.resolve()),
        "track_count": len(selected_tracks),
        "tracks": summaries,
    }
    write_json(run_directory / "summary.json", summary_document)
    print(json.dumps(summary_document, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
