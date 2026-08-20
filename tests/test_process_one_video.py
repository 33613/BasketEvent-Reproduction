"""Tests for the resumable single-video BasketEvent pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from local_script.process_one_video import (
    PipelineConfig,
    SingleVideoPaths,
    SingleVideoPipeline,
)


class RecordingPipeline(SingleVideoPipeline):
    """Replace model subprocesses with deterministic test output creation."""

    def __init__(self, paths, config):
        """Initialize the pipeline and an executed-stage record."""
        super().__init__(paths, config)
        self.executed_stages: list[str] = []

    def _run_command(self, stage: str, command: Sequence[str]) -> None:
        """Record a stage and create the output expected by the orchestrator."""
        self.executed_stages.append(stage)
        self.report["stages"][stage] = {"status": "completed"}
        if stage == "visualize":
            self.paths.visualization.write_bytes(b"video")
            self.paths.visualization_report.write_text("{}", encoding="utf-8")


class SingleVideoPipelineTest(unittest.TestCase):
    """Verify path mapping, commands, and Qwen-empty control flow."""

    def _make_paths(self, root: Path) -> SingleVideoPaths:
        """Create a complete path object rooted in a temporary directory."""
        project = root / "project"
        project.mkdir()
        for relative in (
            "track_one_video.py",
            "recognize.py",
            "inference.py",
            "local_script/visualize_qwen_tracks.py",
        ):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        game = "bkn-vs-det-0022400861"
        data = root / "data"
        video = data / game / "video" / "130.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        roster = root / "artifacts" / game / "metadata" / "recognize_roster.json"
        roster.parent.mkdir(parents=True)
        roster.write_text("{}", encoding="utf-8")
        return SingleVideoPaths(
            project_root=project,
            data_root=data,
            artifacts_root=root / "artifacts",
            game_id=game,
            clip_id="130",
            roster=roster,
            sam3_checkpoint=root / "models" / "sam3.pt",
            sam3_bpe=root / "models" / "bpe.txt.gz",
            qwen_model=root / "models" / "qwen",
            event_checkpoint=root / "models" / "playnet.pt",
            timesformer_model=root / "models" / "timesformer",
        )

    def test_paths_follow_game_artifact_layout(self):
        """All per-clip outputs should remain grouped under the game ID."""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))

            self.assertEqual(paths.video.name, "130.mp4")
            self.assertEqual(paths.raw_tracks.parts[-3:], ("tracks", "raw", "130.json"))
            self.assertEqual(paths.clean_tracks.name, "130.json")
            self.assertEqual(paths.prediction.name, "130_events.json")
            self.assertEqual(paths.visualization.name, "130_overlay.mp4")

    def test_zero_qwen_players_skips_playnet_and_renders_diagnostics(self):
        """An empty Qwen player set should be reported without pipeline failure."""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))
            paths.create_output_directories()
            paths.raw_tracks.write_text(
                json.dumps({"player_0": {"trajectory": []}}), encoding="utf-8"
            )
            paths.clean_tracks.write_text(
                json.dumps({"ball": {"trajectory": []}}), encoding="utf-8"
            )
            pipeline = RecordingPipeline(paths, PipelineConfig(resume=True))

            report = pipeline.run()

            self.assertEqual(report["accepted_player_count"], 0)
            self.assertEqual(report["status"], "completed_with_warning")
            self.assertEqual(report["stages"]["playnet"]["status"], "skipped")
            self.assertEqual(pipeline.executed_stages, ["visualize"])
            self.assertTrue(paths.pipeline_report.is_file())

    def test_commands_include_server_memory_and_local_model_options(self):
        """Generated commands should preserve the verified TITAN settings."""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))
            pipeline = SingleVideoPipeline(paths, PipelineConfig())

            sam3_command = pipeline._sam3_command()
            qwen_command = pipeline._qwen_command()
            playnet_command = pipeline._playnet_command()

            self.assertIn("--offload-video-to-cpu", sam3_command)
            self.assertIn("--offload-state-to-cpu", sam3_command)
            self.assertIn(str(paths.qwen_model), qwen_command)
            self.assertIn(str(paths.event_checkpoint), playnet_command)
            self.assertIn(str(paths.prediction), playnet_command)


if __name__ == "__main__":
    unittest.main()
