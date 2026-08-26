"""Application use case for ingesting and segmenting a long user video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.modules.ingestion import VideoAsset
from src.modules.segmentation import VideoSegment


class VideoIngestion(Protocol):
    """Application-facing contract for accepting a user video."""

    def ingest(self, source_path: str | Path) -> VideoAsset:
        """Return validated metadata for a source video."""
        ...


class VideoSegmentation(Protocol):
    """Application-facing contract for planning and exporting clips."""

    def plan(self, video: VideoAsset) -> list[VideoSegment]:
        """Return model-sized source-video intervals."""
        ...

    def write_manifest(
        self,
        path: str | Path,
        video: VideoAsset,
        segments: Sequence[VideoSegment],
    ) -> Path:
        """Persist segment lineage."""
        ...

    def export(
        self,
        video: VideoAsset,
        segments: Sequence[VideoSegment],
        output_directory: str | Path,
        ffmpeg_binary: str = "ffmpeg",
        overwrite: bool = False,
    ) -> list[Path]:
        """Materialize planned intervals as video clips."""
        ...


@dataclass(frozen=True)
class LongVideoProcessingConfig:
    """Configure the baseline long-video preparation workflow."""

    output_root: Path
    export_clips: bool = False
    ffmpeg_binary: str = "ffmpeg"
    overwrite: bool = False


class LongVideoProcessingApplication:
    """Coordinate ingestion and segmentation through replaceable modules.

    Applications depend only on the public ``ingest`` and ``plan`` contracts.
    A future segmentation strategy can therefore replace fixed windows without
    changing this orchestration class.
    """

    def __init__(
        self,
        ingestion: VideoIngestion,
        segmenter: VideoSegmentation,
        config: LongVideoProcessingConfig,
    ) -> None:
        """Store injected modules without opening files or loading models."""
        self.ingestion = ingestion
        self.segmenter = segmenter
        self.config = config

    def run(self, input_video: str | Path) -> dict[str, Any]:
        """Register a user video, plan segments, and optionally export clips."""
        video = self.ingestion.ingest(input_video)
        segments = self.segmenter.plan(video)
        run_root = self.config.output_root.expanduser() / video.video_id
        manifest = self.segmenter.write_manifest(
            run_root / "segments.json",
            video,
            segments,
        )
        exported: list[Path] = []
        if self.config.export_clips:
            exported = self.segmenter.export(
                video=video,
                segments=segments,
                output_directory=run_root / "clips",
                ffmpeg_binary=self.config.ffmpeg_binary,
                overwrite=self.config.overwrite,
            )
        return {
            "schema_version": "basketevent_long_video_application.v1",
            "video": video.to_dict(),
            "segment_count": len(segments),
            "manifest": str(manifest),
            "exported_clips": [str(path) for path in exported],
        }
