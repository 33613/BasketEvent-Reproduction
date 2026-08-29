"""测试长视频调度、失败重试、断点续跑和最终入库。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.application.process_clip import PipelineConfig, SingleVideoPaths
from src.application.process_long_video import (
    LongVideoScheduler,
    LongVideoSchedulerConfig,
    parse_args,
)
from src.core.config import Settings
from src.modules.database import ProductDatabase
from src.modules.ingestion import VideoAsset
from src.modules.materials import ExportedMaterial
from src.modules.segmentation import VideoSegment


class FakeIngestion:
    """返回固定视频元数据，避免测试依赖实际解码器。"""

    def __init__(self, source_path: Path) -> None:
        """保存测试输入路径。"""
        self.asset = VideoAsset(
            video_id="video-test",
            source_path=source_path,
            filename=source_path.name,
            file_size_bytes=10,
            width=1280,
            height=720,
            fps=30.0,
            frame_count=660,
            duration_seconds=22.0,
        )

    def ingest(self, source_path: str | Path) -> VideoAsset:
        """返回同一个稳定视频对象。"""
        return self.asset


class FakeSegmenter:
    """生成两个重叠窗口并写出最小占位 MP4。"""

    def plan(self, video: VideoAsset) -> list[VideoSegment]:
        """返回 0～12 秒和 10～22 秒两个窗口。"""
        return [
            VideoSegment(
                segment_id="video-test_00000",
                source_video_id=video.video_id,
                index=0,
                start_seconds=0.0,
                end_seconds=12.0,
                source_start_frame=0,
                source_end_frame=360,
                duration_seconds=12.0,
                output_filename="video-test_00000.mp4",
            ),
            VideoSegment(
                segment_id="video-test_00001",
                source_video_id=video.video_id,
                index=1,
                start_seconds=10.0,
                end_seconds=22.0,
                source_start_frame=300,
                source_end_frame=660,
                duration_seconds=12.0,
                output_filename="video-test_00001.mp4",
            ),
        ]

    def write_manifest(self, path, video, segments):
        """写出调度器可审计的窗口清单。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "video": video.to_dict(),
                    "segments": [segment.to_dict() for segment in segments],
                }
            ),
            encoding="utf-8",
        )
        return destination

    def export(self, video, segments, output_directory, **kwargs):
        """创建单窗口入口所需的文件布局。"""
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        outputs = []
        for segment in segments:
            output = root / segment.output_filename
            output.write_bytes(b"window")
            outputs.append(output)
        return outputs


class RetryWindowProcessor:
    """让第一个窗口首次失败，以验证自动重试。"""

    def __init__(self) -> None:
        """初始化每个窗口的调用次数。"""
        self.calls: dict[str, int] = {}

    def run(self, paths: SingleVideoPaths, config: PipelineConfig):
        """第二次调用后写出可合并的 PlayNet 预测。"""
        count = self.calls.get(paths.clip_id, 0) + 1
        self.calls[paths.clip_id] = count
        if paths.clip_id.endswith("00000") and count == 1:
            raise RuntimeError("模拟一次临时模型失败")

        paths.pipeline_report.parent.mkdir(parents=True, exist_ok=True)
        paths.pipeline_report.write_text('{"status":"completed"}', encoding="utf-8")
        paths.prediction.parent.mkdir(parents=True, exist_ok=True)
        local_start, local_end = (
            (8.0, 10.0) if paths.clip_id.endswith("00000") else (0.0, 2.0)
        )
        paths.prediction.write_text(
            json.dumps(
                {
                    "temporal_events": [
                        {
                            "player_id": "player_0",
                            "event": "Made Shot",
                            "confidence": 0.9,
                            "temporal_score": 0.8,
                            "start_time": local_start,
                            "end_time": local_end,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {"status": "completed"}


class FakeMaterialExporter:
    """用小文件替代 FFmpeg，但保留素材时间和编号。"""

    def export(self, *, source_video, material_drafts, output_directory, **kwargs):
        """为每个待剪范围创建一段占位素材。"""
        root = Path(output_directory)
        root.mkdir(parents=True, exist_ok=True)
        results = []
        for index, draft in enumerate(material_drafts):
            output = root / f"{index:05d}.mp4"
            output.write_bytes(b"material")
            results.append(
                ExportedMaterial(
                    material_id=str(draft["material_id"]),
                    video_path=output,
                    start_seconds=float(draft["start_seconds"]),
                    end_seconds=float(draft["end_seconds"]),
                    event_ids=tuple(draft["event_ids"]),
                )
            )
        return tuple(results)


class FakeIdentityProcessor:
    """为最终素材生成一条确定的球衣身份。"""

    def __init__(self) -> None:
        """记录实际处理次数。"""
        self.call_count = 0

    def run(self, *, source_video_id, material, output_directory):
        """写出 CrossMaterialIdentityAssociator 可读取的报告。"""
        self.call_count += 1
        output_directory.mkdir(parents=True, exist_ok=True)
        report = output_directory / "identity.json"
        report.write_text(
            json.dumps(
                {
                    "resolutions": [
                        {
                            "track_id": "player_0",
                            "status": "identified",
                            "jersey_color": "white",
                            "jersey_number": "13",
                            "player_name": None,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return report


class LongVideoSchedulerTest(unittest.TestCase):
    """验证一条命令背后的完整应用编排。"""

    def test_retry_resume_timeline_identity_and_database(self) -> None:
        """临时失败应重试，重复运行应复用成功窗口和身份结果。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.write_bytes(b"input")
            settings = Settings(
                project_root=root,
                data_root=root / "data",
                artifacts_root=root / "artifacts",
                runtime_root=root / "runtime",
                product_data_root=root / "product",
                model_root=root / "models",
            )
            window_processor = RetryWindowProcessor()
            identity_processor = FakeIdentityProcessor()
            scheduler = LongVideoScheduler(
                settings=settings,
                scheduler_config=LongVideoSchedulerConfig(
                    runtime_root=root / "long_video_runtime",
                    with_identity=True,
                    max_attempts_per_run=2,
                ),
                pipeline_config=PipelineConfig(),
                ingestion=FakeIngestion(source),
                segmenter=FakeSegmenter(),
                window_processor=window_processor,
                material_exporter=FakeMaterialExporter(),
                identity_processor=identity_processor,
            )

            first = scheduler.run(source)
            second = scheduler.run(source)

            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["material_count"], 1)
            self.assertEqual(first["windows"]["video-test_00000"]["attempt_count"], 2)
            self.assertEqual(window_processor.calls["video-test_00000"], 2)
            self.assertEqual(window_processor.calls["video-test_00001"], 1)
            self.assertEqual(identity_processor.call_count, 1)
            self.assertEqual(second["status"], "completed")

            job_root = root / "long_video_runtime" / "video-test"
            timeline = json.loads(
                (job_root / "event_timeline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(timeline["events"]), 1)
            self.assertEqual(timeline["events"][0]["event"], "Made Shot")
            materials = ProductDatabase.open(job_root / "product_data").list_materials()
            self.assertEqual(len(materials), 1)
            self.assertEqual(materials[0].participants[0].jersey_number, "13")

    def test_cli_defaults_to_repository_test_runtime(self) -> None:
        """默认运行数据必须位于 tests/long_video_runtime。"""
        args = parse_args(["input.mp4"])
        self.assertEqual(
            args.runtime_root,
            Path(__file__).resolve().parents[1] / "tests" / "long_video_runtime",
        )

    def test_full_restart_requires_overwriting_windows(self) -> None:
        """关闭续跑时必须明确允许重建固定窗口。"""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "必须同时覆盖固定窗口"):
                LongVideoSchedulerConfig(
                    runtime_root=Path(directory),
                    resume=False,
                    overwrite_windows=False,
                )

    def test_resume_rejects_changed_pipeline_configuration(self) -> None:
        """同一任务目录不能混用不同 PlayNet 或采样参数。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.write_bytes(b"input")
            settings = Settings(
                project_root=root,
                data_root=root / "data",
                artifacts_root=root / "artifacts",
                runtime_root=root / "runtime",
                product_data_root=root / "product",
                model_root=root / "models",
            )
            common_arguments = {
                "settings": settings,
                "scheduler_config": LongVideoSchedulerConfig(
                    runtime_root=root / "long_video_runtime",
                    max_attempts_per_run=2,
                ),
                "ingestion": FakeIngestion(source),
                "segmenter": FakeSegmenter(),
                "material_exporter": FakeMaterialExporter(),
            }
            LongVideoScheduler(
                **common_arguments,
                pipeline_config=PipelineConfig(topk=5),
                window_processor=RetryWindowProcessor(),
            ).run(source)

            with self.assertRaisesRegex(ValueError, "模型配置与本次不同"):
                LongVideoScheduler(
                    **common_arguments,
                    pipeline_config=PipelineConfig(topk=3),
                    window_processor=RetryWindowProcessor(),
                ).run(source)

    def test_resume_allows_changed_gpu_assignment(self) -> None:
        """服务器资源变化后应允许把未完成工作切换到另一张 GPU。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.mp4"
            source.write_bytes(b"input")
            settings = Settings(
                project_root=root,
                data_root=root / "data",
                artifacts_root=root / "artifacts",
                runtime_root=root / "runtime",
                product_data_root=root / "product",
                model_root=root / "models",
            )
            common_arguments = {
                "settings": settings,
                "scheduler_config": LongVideoSchedulerConfig(
                    runtime_root=root / "long_video_runtime",
                    max_attempts_per_run=2,
                ),
                "ingestion": FakeIngestion(source),
                "segmenter": FakeSegmenter(),
                "material_exporter": FakeMaterialExporter(),
            }
            first_processor = RetryWindowProcessor()
            LongVideoScheduler(
                **common_arguments,
                pipeline_config=PipelineConfig(sam3_gpus="0,1", playnet_gpu=0),
                window_processor=first_processor,
            ).run(source)

            second_processor = RetryWindowProcessor()
            result = LongVideoScheduler(
                **common_arguments,
                pipeline_config=PipelineConfig(sam3_gpus="1", playnet_gpu=1),
                window_processor=second_processor,
            ).run(source)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(second_processor.calls, {})
            state = json.loads(
                (
                    root / "long_video_runtime" / "video-test" / "job_state.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(state["configuration"]["pipeline"]["sam3_gpus"], "1")
            self.assertEqual(state["configuration"]["pipeline"]["playnet_gpu"], 1)


if __name__ == "__main__":
    unittest.main()
