"""Tests for product-oriented ingestion, segmentation, and material modules."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.process_video import (
    LongVideoProcessingApplication,
    LongVideoProcessingConfig,
)
from src.modules.ingestion import VideoAsset
from src.modules.catalog import MaterialStatisticsService
from src.modules.segmentation import LongVideoSegmenter


def make_asset(duration_seconds: float = 30.0) -> VideoAsset:
    """Return deterministic metadata without requiring a video codec."""
    return VideoAsset(
        video_id="video-test",
        source_path=Path("input.mp4"),
        filename="input.mp4",
        file_size_bytes=100,
        width=1280,
        height=720,
        fps=30.0,
        frame_count=int(duration_seconds * 30),
        duration_seconds=duration_seconds,
    )


class FakeIngestionService:
    """Return fixed metadata to isolate application orchestration."""

    def __init__(self, asset: VideoAsset) -> None:
        """Store the asset returned by every ingest call."""
        self.asset = asset
        self.inputs: list[Path] = []

    def ingest(self, source_path: str | Path) -> VideoAsset:
        """Record the source path and return the configured asset."""
        self.inputs.append(Path(source_path))
        return self.asset


class LongVideoSegmenterTest(unittest.TestCase):
    """Verify the deterministic baseline segmentation contract."""

    def test_overlapping_windows_retain_source_timestamps(self) -> None:
        """A 30-second video should produce three chronological windows."""
        segments = LongVideoSegmenter(
            window_seconds=12,
            overlap_seconds=2,
        ).plan(make_asset())

        self.assertEqual(len(segments), 3)
        self.assertEqual(
            [(item.start_seconds, item.end_seconds) for item in segments],
            [(0.0, 12.0), (10.0, 22.0), (20.0, 30.0)],
        )
        self.assertEqual(segments[1].source_video_id, "video-test")

    def test_short_tail_is_merged_into_previous_window(self) -> None:
        """A tiny final tail should extend the prior segment instead."""
        segments = LongVideoSegmenter(
            window_seconds=10,
            overlap_seconds=0,
            minimum_tail_seconds=3,
        ).plan(make_asset(duration_seconds=21.0))

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1].start_seconds, 10.0)
        self.assertEqual(segments[-1].end_seconds, 21.0)


class LongVideoProcessingApplicationTest(unittest.TestCase):
    """Verify applications coordinate modules without implementing them."""

    def test_application_writes_manifest_without_exporting_media(self) -> None:
        """The default preparation pass should only create segment metadata."""
        with tempfile.TemporaryDirectory() as directory:
            asset = make_asset()
            ingestion = FakeIngestionService(asset)
            application = LongVideoProcessingApplication(
                ingestion=ingestion,
                segmenter=LongVideoSegmenter(12, 2),
                config=LongVideoProcessingConfig(output_root=Path(directory)),
            )

            report = application.run("uploaded-game.mp4")

            manifest = Path(report["manifest"])
            self.assertTrue(manifest.is_file())
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(document["segments"]), 3)
            self.assertEqual(report["exported_clips"], [])
            self.assertEqual(ingestion.inputs, [Path("uploaded-game.mp4")])


class MaterialStatisticsServiceTest(unittest.TestCase):
    """Verify processed clips become simple catalog statistics."""

    def test_events_and_jersey_numbers_are_counted(self) -> None:
        """Statistics should work without mapping numbers to player names."""
        reports = [
            {
                "player_predictions": [
                    {
                        "player_id": "player_0",
                        "jersey_color": "white",
                        "jersey_number": "20",
                        "event": "ast",
                        "confidence": 0.9,
                    },
                    {
                        "player_id": "player_1",
                        "jersey_color": "black",
                        "jersey_number": "17",
                        "event": "Made Shot",
                        "confidence": 0.8,
                    },
                ],
                "temporal_events": [{}, {}],
            }
        ]

        result = MaterialStatisticsService().summarize(reports)

        self.assertEqual(result.clip_count, 1)
        self.assertEqual(result.non_background_prediction_count, 2)
        self.assertEqual(result.event_counts, {"Made Shot": 1, "ast": 1})
        self.assertEqual(result.participant_counts["white #20"], 1)
        self.assertAlmostEqual(result.mean_confidence or 0.0, 0.85)


if __name__ == "__main__":
    unittest.main()
