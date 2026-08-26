"""Create model-sized clips from a long user video.

The first product version uses deterministic overlapping windows. This is a
deliberately simple baseline: future scoreboard, replay, audio, or shot-boundary
strategies can implement the same ``plan`` contract without changing the
application layer.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from src.modules.ingestion import VideoAsset


@dataclass(frozen=True)
class VideoSegment:
    """Describe one model-sized interval from a source video."""

    segment_id: str
    source_video_id: str
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    output_filename: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable segment metadata."""
        return asdict(self)


class LongVideoSegmenter:
    """Plan and optionally export overlapping fixed-duration video segments."""

    def __init__(
        self,
        window_seconds: float = 12.0,
        overlap_seconds: float = 2.0,
        minimum_tail_seconds: float = 2.0,
    ) -> None:
        """Configure fixed-window segmentation.

        Args:
            window_seconds: Maximum duration of a generated segment.
            overlap_seconds: Context shared by adjacent segments.
            minimum_tail_seconds: A shorter final segment is merged into the
                previous interval instead of being emitted independently.

        Raises:
            ValueError: If the durations cannot produce forward progress.
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if overlap_seconds < 0 or overlap_seconds >= window_seconds:
            raise ValueError("overlap_seconds must be in [0, window_seconds)")
        if minimum_tail_seconds < 0:
            raise ValueError("minimum_tail_seconds cannot be negative")
        if minimum_tail_seconds > window_seconds:
            raise ValueError("minimum_tail_seconds cannot exceed window_seconds")
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.minimum_tail_seconds = float(minimum_tail_seconds)

    def plan(self, video: VideoAsset) -> list[VideoSegment]:
        """Return chronological model-sized intervals for a video.

        Args:
            video: Metadata produced by the ingestion module.

        Returns:
            Non-empty ordered segments that retain source-video timestamps.
        """
        duration = float(video.duration_seconds)
        if duration <= 0:
            raise ValueError("Video duration must be positive")

        step = self.window_seconds - self.overlap_seconds
        segments: list[VideoSegment] = []
        start = 0.0
        while start < duration:
            end = min(start + self.window_seconds, duration)
            segment_duration = end - start
            if segments and segment_duration < self.minimum_tail_seconds:
                previous = segments[-1]
                segments[-1] = replace(
                    previous,
                    end_seconds=duration,
                    duration_seconds=duration - previous.start_seconds,
                )
                break

            index = len(segments)
            segment_id = f"{video.video_id}_{index:05d}"
            segments.append(
                VideoSegment(
                    segment_id=segment_id,
                    source_video_id=video.video_id,
                    index=index,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=segment_duration,
                    output_filename=f"{segment_id}.mp4",
                )
            )
            if end >= duration:
                break
            start += step
        return segments

    @staticmethod
    def write_manifest(
        path: str | Path,
        video: VideoAsset,
        segments: Sequence[VideoSegment],
    ) -> Path:
        """Persist source metadata and segment lineage as JSON."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": "basketevent_segments.v1",
            "video": video.to_dict(),
            "segments": [segment.to_dict() for segment in segments],
        }
        destination.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @staticmethod
    def export(
        video: VideoAsset,
        segments: Sequence[VideoSegment],
        output_directory: str | Path,
        ffmpeg_binary: str = "ffmpeg",
        overwrite: bool = False,
    ) -> list[Path]:
        """Materialize planned intervals as independently decodable MP4 files.

        This concrete FFmpeg call is intentionally kept beside the baseline
        segmenter for now. It can later move behind an infrastructure adapter
        without changing ``plan`` or the application layer.
        """
        output_root = Path(output_directory).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for segment in segments:
            destination = output_root / segment.output_filename
            if (
                not overwrite
                and destination.is_file()
                and destination.stat().st_size > 0
            ):
                outputs.append(destination)
                continue
            command = [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y" if overwrite else "-n",
                "-ss",
                f"{segment.start_seconds:.6f}",
                "-i",
                str(video.source_path),
                "-t",
                f"{segment.duration_seconds:.6f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(destination),
            ]
            subprocess.run(command, check=True)
            outputs.append(destination)
        return outputs
