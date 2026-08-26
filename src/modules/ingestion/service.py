"""Validate a user video and convert it into BasketEvent metadata.

Ingestion is the boundary between an uploaded file and the internal processing
pipeline. It does not run a neural network. The service validates that the
video is readable, probes its basic media properties, and assigns a stable ID
that later stages use instead of relying on a user-provided filename.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoAsset:
    """Describe one source video accepted by the application.

    Attributes:
        video_id: Stable identifier derived from the local file metadata.
        source_path: Resolved path to the original user video.
        filename: Original filename retained for display.
        file_size_bytes: Size of the source file.
        width: Encoded frame width in pixels.
        height: Encoded frame height in pixels.
        fps: Reported source frame rate.
        frame_count: Reported number of frames.
        duration_seconds: Approximate duration derived from frames and FPS.
    """

    video_id: str
    source_path: Path
    filename: str
    file_size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable video metadata."""
        value = asdict(self)
        value["source_path"] = str(self.source_path)
        return value


class VideoIngestionService:
    """Validate and probe user-supplied video files.

    The service deliberately leaves the source file in place. A future storage
    adapter can copy an upload to object storage without changing applications
    that consume :class:`VideoAsset`.
    """

    SUPPORTED_SUFFIXES = frozenset({".avi", ".mkv", ".mov", ".mp4", ".webm"})

    def ingest(self, source_path: str | Path) -> VideoAsset:
        """Validate a video and return its internal metadata.

        Args:
            source_path: User video available on the local filesystem.

        Returns:
            Probed metadata used by segmentation and later applications.

        Raises:
            FileNotFoundError: If the input does not exist.
            ValueError: If the extension is unsupported or the video cannot be
                decoded well enough to obtain valid dimensions and duration.
        """
        import cv2

        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported video extension: {path.suffix}")

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open video: {path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                "Video metadata is incomplete: "
                f"fps={fps}, frames={frame_count}, size={width}x{height}"
            )

        stat = path.stat()
        fingerprint = f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        video_id = hashlib.sha256(fingerprint).hexdigest()[:16]
        return VideoAsset(
            video_id=video_id,
            source_path=path,
            filename=path.name,
            file_size_bytes=stat.st_size,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / fps,
        )
