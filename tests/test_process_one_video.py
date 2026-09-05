"""测试可续跑的单窗口处理流程。"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Sequence

from src.application.process_clip import (
    PipelineConfig,
    SingleVideoPaths,
    SingleVideoPipeline,
)


class RecordingPipeline(SingleVideoPipeline):
    """用确定性文件替代模型子进程。"""

    def __init__(self, paths, config):
        """初始化流程并记录执行阶段。"""
        super().__init__(paths, config)
        self.executed_stages: list[str] = []
        self.executed_commands: list[list[str]] = []

    def _run_command(self, stage: str, command: Sequence[str]) -> None:
        """记录命令并生成编排器期待的最小输出。"""
        self.executed_stages.append(stage)
        self.executed_commands.append([str(part) for part in command])
        self.report["stages"][stage] = {"status": "completed"}
        if stage == "sam3":
            self.paths.raw_tracks.write_text(
                json.dumps({"player_0": {"trajectory": [[0, 0, 10, 20]]}}),
                encoding="utf-8",
            )
        elif stage == "prepare":
            self.paths.model_tracks.write_text(
                json.dumps({"player_0": {"trajectory": [[0, 0, 10, 20]]}}),
                encoding="utf-8",
            )
            self.paths.track_preparation_report.write_text("{}", encoding="utf-8")
        elif stage == "playnet":
            self.paths.prediction.write_text("{}", encoding="utf-8")
        elif stage == "visualize":
            self.paths.visualization.write_bytes(b"video")
            self.paths.visualization_report.write_text("{}", encoding="utf-8")


class SingleVideoPipelineTest(unittest.TestCase):
    """验证路径、命令和无有效人物轨迹时的控制流。"""

    def _make_paths(self, root: Path) -> SingleVideoPaths:
        """在临时目录中创建完整路径对象。"""
        project = root / "project"
        project.mkdir()
        for relative in (
            "src/modules/tracking/sam3_tracker.py",
            "src/modules/tracking/preparation.py",
            "src/modules/event_recognition/inference.py",
            "src/modules/materials/visualization.py",
        ):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        game = "bkn-vs-det-0022400861"
        data = root / "data"
        video = data / game / "video" / "130.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        return SingleVideoPaths(
            project_root=project,
            data_root=data,
            artifacts_root=root / "artifacts",
            game_id=game,
            clip_id="130",
            sam3_checkpoint=root / "models" / "sam3.pt",
            sam3_bpe=root / "models" / "bpe.txt.gz",
            event_checkpoint=root / "models" / "playnet.pt",
            timesformer_model=root / "models" / "timesformer",
        )

    def test_paths_follow_game_artifact_layout(self):
        """每个窗口的输出应按比赛编号集中存放。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))

            self.assertEqual(paths.video.name, "130.mp4")
            self.assertEqual(paths.raw_tracks.parts[-3:], ("tracks", "raw", "130.json"))
            self.assertEqual(
                paths.model_tracks.parts[-3:],
                ("tracks", "model_input", "130.json"),
            )
            self.assertEqual(paths.prediction.name, "130_events.json")
            self.assertEqual(paths.visualization.name, "130_overlay.mp4")

    def test_zero_model_players_skips_playnet(self):
        """没有有效人物轨迹时应跳过 PlayNet，但仍生成追踪可视化。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))
            paths.create_output_directories()
            paths.raw_tracks.write_text(
                json.dumps({"player_0": {"trajectory": []}}), encoding="utf-8"
            )
            paths.model_tracks.write_text(
                json.dumps({"ball": {"trajectory": []}}), encoding="utf-8"
            )
            pipeline = RecordingPipeline(paths, PipelineConfig(resume=True))

            report = pipeline.run()

            self.assertEqual(report["model_player_count"], 0)
            self.assertEqual(report["status"], "completed_with_warning")
            self.assertEqual(report["stages"]["playnet"]["status"], "skipped")
            self.assertEqual(pipeline.executed_stages, ["visualize"])
            self.assertTrue(paths.pipeline_report.is_file())

    def test_commands_use_unfiltered_model_tracks(self):
        """轨迹准备和 PlayNet 命令不得依赖 Qwen 输出。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))
            pipeline = SingleVideoPipeline(paths, PipelineConfig())

            sam3_command = pipeline._sam3_command()
            prepare_command = pipeline._prepare_command()
            playnet_command = pipeline._playnet_command()

            self.assertIn("--offload-video-to-cpu", sam3_command)
            self.assertIn("--offload-state-to-cpu", sam3_command)
            self.assertIn(str(paths.raw_tracks), prepare_command)
            self.assertIn(str(paths.model_tracks), prepare_command)
            self.assertIn(str(paths.model_tracks), playnet_command)
            self.assertNotIn("qwen", " ".join(prepare_command).lower())

    def test_legacy_identity_stage_maps_to_prepare(self):
        """旧阶段名只作为兼容入口，实际执行轨迹准备。"""
        self.assertEqual(PipelineConfig(start_at="qwen").start_at, "prepare")
        self.assertEqual(PipelineConfig(start_at="identity").start_at, "prepare")

    def test_visualize_only_uses_existing_tracks_without_playnet(self):
        """已有轨迹在无预测结果时仍应可以可视化。"""
        with tempfile.TemporaryDirectory() as directory:
            paths = self._make_paths(Path(directory))
            paths.create_output_directories()
            paths.raw_tracks.write_text(
                json.dumps({"player_0": {"trajectory": []}}), encoding="utf-8"
            )
            paths.model_tracks.write_text(
                json.dumps({"player_0": {"trajectory": []}}), encoding="utf-8"
            )
            pipeline = RecordingPipeline(
                paths,
                PipelineConfig(visualize_only=True, resume=False),
            )

            report = pipeline.run()

            self.assertEqual(pipeline.executed_stages, ["visualize"])
            self.assertEqual(report["stages"]["playnet"]["status"], "skipped")
            self.assertEqual(report["visualization_mode"], "tracks_only")
            self.assertNotIn("--prediction_json_path", pipeline.executed_commands[0])


if __name__ == "__main__":
    unittest.main()
