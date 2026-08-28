"""把一个长视频自动处理成事件素材并登记到产品数据库。

该应用只负责编排已经存在的模块：视频接入、固定窗口、单窗口模型链路、
全局事件时间线、最终素材导出和可选身份后处理。每个窗口的状态都会持久化，
因此 SSH 中断后可以直接重跑同一命令继续处理。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.application.finalize_materials import MaterialFinalizationApplication
from src.application.process_clip import (
    PipelineConfig,
    SingleVideoPaths,
    SingleVideoPipeline,
)
from src.core.config import SETTINGS, Settings
from src.modules.catalog import CatalogService
from src.modules.database import ProductDatabase
from src.modules.event_recognition import EventTimelineService
from src.modules.identity import CrossMaterialIdentityAssociator
from src.modules.ingestion import VideoAsset, VideoIngestionService
from src.modules.materials import ExportedMaterial, MaterialExporter
from src.modules.segmentation import LongVideoSegmenter, VideoSegment


def _utc_now() -> str:
    """返回便于写入 JSON 的 UTC 时间。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_component(value: str) -> str:
    """把业务编号转换为 Windows 和 Linux 都可用的目录名。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return normalized.strip("._") or "item"


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象并拒绝不完整的根结构。"""
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """先写临时文件再替换，避免中断留下半个状态文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class WindowProcessor(Protocol):
    """约束单个固定窗口处理器。"""

    def run(self, paths: SingleVideoPaths, config: PipelineConfig) -> Mapping[str, Any]:
        """运行一个窗口并返回流水线报告。"""
        ...


class MaterialIdentityProcessor(Protocol):
    """约束最终素材身份处理器。"""

    def run(
        self,
        *,
        source_video_id: str,
        material: ExportedMaterial,
        output_directory: Path,
    ) -> Path:
        """处理一段最终素材并返回身份报告路径。"""
        ...


class DefaultWindowProcessor:
    """调用现有单窗口流水线，不复制任何模型实现。"""

    def run(self, paths: SingleVideoPaths, config: PipelineConfig) -> Mapping[str, Any]:
        """运行 SAM3、轨迹准备、PlayNet 和可视化。"""
        return SingleVideoPipeline(paths, config).run()


@dataclass(frozen=True)
class LongVideoJobPaths:
    """集中保存一次长视频任务的所有运行路径。"""

    runtime_root: Path
    video_id: str

    @property
    def job_root(self) -> Path:
        """返回本次任务根目录。"""
        return self.runtime_root / self.video_id

    @property
    def state(self) -> Path:
        """返回可恢复任务状态文件。"""
        return self.job_root / "job_state.json"

    @property
    def segment_manifest(self) -> Path:
        """返回固定窗口清单。"""
        return self.job_root / "segments.json"

    @property
    def window_data_root(self) -> Path:
        """返回符合单窗口入口约定的窗口视频根目录。"""
        return self.job_root / "windows"

    @property
    def window_video_directory(self) -> Path:
        """返回实际窗口 MP4 目录。"""
        return self.window_data_root / self.video_id / "video"

    @property
    def window_artifacts_root(self) -> Path:
        """返回窗口模型中间结果目录。"""
        return self.job_root / "window_artifacts"

    @property
    def timeline(self) -> Path:
        """返回源视频全局事件时间线。"""
        return self.job_root / "event_timeline.json"

    @property
    def final_materials(self) -> Path:
        """返回最终事件素材目录。"""
        return self.job_root / "final_materials"

    @property
    def final_identity(self) -> Path:
        """返回最终素材身份中间结果目录。"""
        return self.job_root / "final_identity"

    @property
    def identity_index(self) -> Path:
        """返回素材编号到身份报告的索引。"""
        return self.job_root / "identity_index.json"

    @property
    def product_data(self) -> Path:
        """返回本次测试独立的 SQLite 和媒体目录。"""
        return self.job_root / "product_data"

    @property
    def finalization_report(self) -> Path:
        """返回最终导出与入库报告。"""
        return self.job_root / "finalization_report.json"


@dataclass(frozen=True)
class LongVideoSchedulerConfig:
    """配置长视频调度、重试和可选身份阶段。"""

    runtime_root: Path
    window_seconds: float = 12.0
    overlap_seconds: float = 2.0
    minimum_tail_seconds: float = 2.0
    max_attempts_per_run: int = 2
    allow_partial: bool = False
    with_identity: bool = False
    resume: bool = True
    overwrite_windows: bool = False
    ffmpeg_binary: str = "ffmpeg"

    def __post_init__(self) -> None:
        """在启动昂贵模型前检查调度参数。"""
        if self.max_attempts_per_run <= 0:
            raise ValueError("max_attempts_per_run 必须为正数")
        if not self.resume and not self.overwrite_windows:
            raise ValueError("关闭断点续跑时必须同时覆盖固定窗口")


class SubprocessMaterialIdentityProcessor:
    """通过已有命令行模块处理最终素材的轨迹和身份。"""

    def __init__(
        self,
        *,
        settings: Settings,
        pipeline_config: PipelineConfig,
        identity_gpus: str = "0",
        roster_json: Path | None = None,
        resume: bool = True,
    ) -> None:
        """保存模型路径、GPU、名单和断点续跑策略。"""
        self.settings = settings
        self.pipeline_config = pipeline_config
        self.identity_gpus = identity_gpus
        self.roster_json = roster_json
        self.resume = resume

    @staticmethod
    def _nonempty(path: Path) -> bool:
        """判断一个阶段产物是否存在且非空。"""
        return path.is_file() and path.stat().st_size > 0

    @staticmethod
    def _environment() -> dict[str, str]:
        """为模型子进程构造隔离环境。"""
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        return environment

    def _run_command(self, command: Sequence[str], log_path: Path) -> None:
        """实时显示子进程输出并保存日志。"""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(" ".join(str(value) for value in command), flush=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [str(value) for value in command],
                cwd=self.settings.project_root,
                env=self._environment(),
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
            return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    def run(
        self,
        *,
        source_video_id: str,
        material: ExportedMaterial,
        output_directory: Path,
    ) -> Path:
        """依次运行 SAM3 和身份服务，并返回身份审计报告。"""
        output_directory.mkdir(parents=True, exist_ok=True)
        raw_tracks = output_directory / "raw_tracks.json"
        identity_tracks = output_directory / "identity_tracks.json"
        identity_report = output_directory / "identity_tracks_identity.json"

        if not (self.resume and self._nonempty(raw_tracks)):
            sam3_command = [
                sys.executable,
                "-u",
                "-m",
                "src.modules.tracking.sam3_tracker",
                "--video_path",
                str(material.video_path),
                "--json_save_path",
                str(raw_tracks),
                "--gpus_to_use",
                self.pipeline_config.sam3_gpus,
                "--sam3_checkpoint",
                str(self.settings.sam3_checkpoint),
                "--sam3_bpe",
                str(self.settings.sam3_bpe),
                "--max-num-objects",
                str(self.pipeline_config.max_num_objects),
                "--max-ball-objects",
                str(self.pipeline_config.max_ball_objects),
                "--sam3-num-maskmem",
                str(self.pipeline_config.sam3_num_maskmem),
                "--sam3-max-cond-frames",
                str(self.pipeline_config.sam3_max_cond_frames),
            ]
            if self.pipeline_config.offload_video_to_cpu:
                sam3_command.append("--offload-video-to-cpu")
            if self.pipeline_config.offload_state_to_cpu:
                sam3_command.append("--offload-state-to-cpu")
            self._run_command(sam3_command, output_directory / "sam3.log")

        if not (self.resume and self._nonempty(identity_report)):
            identity_command = [
                sys.executable,
                "-u",
                "-m",
                "src.modules.identity.service",
                "--video_path",
                str(material.video_path),
                "--bbox_json_path",
                str(raw_tracks),
                "--json_save_path",
                str(identity_tracks),
                "--game_id",
                source_video_id,
                "--gpus_to_use",
                self.identity_gpus,
                "--qwen_model",
                str(self.settings.qwen_model),
            ]
            if self.roster_json is not None:
                identity_command.extend(["--roster_json", str(self.roster_json)])
            self._run_command(identity_command, output_directory / "identity.log")

        if not self._nonempty(identity_report):
            raise RuntimeError(f"身份服务没有生成报告：{identity_report}")
        return identity_report


class LongVideoScheduler:
    """执行可恢复的长视频端到端处理任务。"""

    def __init__(
        self,
        *,
        settings: Settings,
        scheduler_config: LongVideoSchedulerConfig,
        pipeline_config: PipelineConfig,
        ingestion: VideoIngestionService | None = None,
        segmenter: LongVideoSegmenter | None = None,
        window_processor: WindowProcessor | None = None,
        timeline_service: EventTimelineService | None = None,
        material_exporter: MaterialExporter | None = None,
        identity_processor: MaterialIdentityProcessor | None = None,
    ) -> None:
        """注入可替换模块；默认组合项目现有实现。"""
        self.settings = settings
        self.scheduler_config = scheduler_config
        self.pipeline_config = pipeline_config
        self.ingestion = ingestion or VideoIngestionService()
        self.segmenter = segmenter or LongVideoSegmenter(
            window_seconds=scheduler_config.window_seconds,
            overlap_seconds=scheduler_config.overlap_seconds,
            minimum_tail_seconds=scheduler_config.minimum_tail_seconds,
        )
        self.window_processor = window_processor or DefaultWindowProcessor()
        self.timeline_service = timeline_service or EventTimelineService()
        self.material_exporter = material_exporter or MaterialExporter(
            ffmpeg_binary=scheduler_config.ffmpeg_binary,
            overwrite=False,
        )
        self.identity_processor = identity_processor

    def _new_state(
        self, video: VideoAsset, segments: Sequence[VideoSegment]
    ) -> dict[str, Any]:
        """创建或补齐任务状态，不清除既有成功记录。"""
        paths = LongVideoJobPaths(self.scheduler_config.runtime_root, video.video_id)
        state = (
            _read_json_object(paths.state)
            if self.scheduler_config.resume and paths.state.is_file()
            else {}
        )
        window_configuration = {
            "window_seconds": self.scheduler_config.window_seconds,
            "overlap_seconds": self.scheduler_config.overlap_seconds,
            "minimum_tail_seconds": self.scheduler_config.minimum_tail_seconds,
        }
        pipeline_configuration = asdict(self.pipeline_config)
        previous_configuration = state.get("configuration", {})
        if state and (
            previous_configuration.get("windows") != window_configuration
            or previous_configuration.get("pipeline") != pipeline_configuration
        ):
            raise ValueError(
                "任务目录中的切窗或模型配置与本次不同。若要全部重算，请同时使用 "
                "--no-resume --overwrite-windows。"
            )
        state.update(
            {
                "schema_version": "basketevent_long_video_job.v1",
                "video": video.to_dict(),
                "video_id": video.video_id,
                "status": "running",
                "updated_utc": _utc_now(),
                "configuration": {
                    "windows": window_configuration,
                    "pipeline": pipeline_configuration,
                    "max_attempts_per_run": self.scheduler_config.max_attempts_per_run,
                    "allow_partial": self.scheduler_config.allow_partial,
                    "with_identity": self.scheduler_config.with_identity,
                },
            }
        )
        previous_windows = state.get("windows", {})
        if not isinstance(previous_windows, Mapping):
            previous_windows = {}
        state["windows"] = {
            segment.segment_id: dict(previous_windows.get(segment.segment_id, {}))
            for segment in segments
        }
        for segment in segments:
            state["windows"][segment.segment_id].setdefault("status", "pending")
            state["windows"][segment.segment_id].setdefault("attempt_count", 0)
        if not self.scheduler_config.with_identity:
            state.pop("identities", None)
        return state

    def _window_paths(
        self, job_paths: LongVideoJobPaths, segment: VideoSegment
    ) -> SingleVideoPaths:
        """把导出的固定窗口映射到单窗口流水线目录结构。"""
        return SingleVideoPaths(
            project_root=self.settings.project_root,
            data_root=job_paths.window_data_root,
            artifacts_root=job_paths.window_artifacts_root,
            game_id=job_paths.video_id,
            clip_id=segment.segment_id,
            sam3_checkpoint=self.settings.sam3_checkpoint,
            sam3_bpe=self.settings.sam3_bpe,
            event_checkpoint=self.settings.event_checkpoint,
            timesformer_model=self.settings.timesformer_model,
        )

    def _process_windows(
        self,
        *,
        job_paths: LongVideoJobPaths,
        segments: Sequence[VideoSegment],
        state: dict[str, Any],
    ) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
        """逐窗执行模型链路，保存每次尝试并返回预测报告。"""
        predictions: dict[str, Mapping[str, Any]] = {}
        failed: list[str] = []
        for segment in segments:
            window_state = state["windows"][segment.segment_id]
            paths = self._window_paths(job_paths, segment)
            if (
                self.scheduler_config.resume
                and window_state.get("status") == "succeeded"
                and paths.pipeline_report.is_file()
            ):
                if paths.prediction.is_file() and paths.prediction.stat().st_size > 0:
                    predictions[segment.segment_id] = _read_json_object(
                        paths.prediction
                    )
                continue

            succeeded = False
            for _ in range(self.scheduler_config.max_attempts_per_run):
                window_state["status"] = "running"
                window_state["attempt_count"] = int(window_state["attempt_count"]) + 1
                window_state["started_utc"] = _utc_now()
                window_state["video"] = str(paths.video)
                window_state["pipeline_report"] = str(paths.pipeline_report)
                state["updated_utc"] = _utc_now()
                _write_json_atomic(job_paths.state, state)
                try:
                    report = self.window_processor.run(paths, self.pipeline_config)
                    window_state["status"] = "succeeded"
                    window_state["pipeline_status"] = report.get("status")
                    window_state["finished_utc"] = _utc_now()
                    window_state.pop("error", None)
                    if (
                        paths.prediction.is_file()
                        and paths.prediction.stat().st_size > 0
                    ):
                        predictions[segment.segment_id] = _read_json_object(
                            paths.prediction
                        )
                        window_state["prediction"] = str(paths.prediction)
                    else:
                        window_state["prediction"] = None
                    succeeded = True
                    break
                except Exception as error:
                    window_state["status"] = "failed"
                    window_state["finished_utc"] = _utc_now()
                    window_state["error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                    _write_json_atomic(job_paths.state, state)
            if not succeeded:
                failed.append(segment.segment_id)
            state["updated_utc"] = _utc_now()
            _write_json_atomic(job_paths.state, state)
        return predictions, failed

    def _process_identities(
        self,
        *,
        job_paths: LongVideoJobPaths,
        materials: Sequence[ExportedMaterial],
        state: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """批量处理最终素材身份；单段失败只产生警告。"""
        if self.identity_processor is None:
            raise RuntimeError("with_identity=True 但没有配置身份处理器")
        identity_state = state.setdefault("identities", {})
        reports: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        for material in materials:
            item_state = identity_state.setdefault(
                material.material_id, {"status": "pending", "attempt_count": 0}
            )
            report_value = item_state.get("report")
            report_path = Path(str(report_value)) if report_value else None
            if (
                self.scheduler_config.resume
                and item_state.get("status") == "succeeded"
                and report_path is not None
                and report_path.is_file()
            ):
                reports[material.material_id] = _read_json_object(report_path)
                continue

            succeeded = False
            output = job_paths.final_identity / _safe_component(material.material_id)
            for _ in range(self.scheduler_config.max_attempts_per_run):
                item_state["status"] = "running"
                item_state["attempt_count"] = int(item_state["attempt_count"]) + 1
                item_state["started_utc"] = _utc_now()
                _write_json_atomic(job_paths.state, state)
                try:
                    report_path = self.identity_processor.run(
                        source_video_id=job_paths.video_id,
                        material=material,
                        output_directory=output,
                    )
                    reports[material.material_id] = _read_json_object(report_path)
                    item_state.update(
                        {
                            "status": "succeeded",
                            "report": str(report_path),
                            "finished_utc": _utc_now(),
                        }
                    )
                    item_state.pop("error", None)
                    succeeded = True
                    break
                except Exception as error:
                    item_state.update(
                        {
                            "status": "failed",
                            "finished_utc": _utc_now(),
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )
                    _write_json_atomic(job_paths.state, state)
            if not succeeded:
                failed.append(material.material_id)
            _write_json_atomic(job_paths.state, state)
        _write_json_atomic(
            job_paths.identity_index,
            {
                material_id: item["report"]
                for material_id, item in identity_state.items()
                if item.get("status") == "succeeded"
            },
        )
        return reports, failed

    def run(self, input_video: str | Path) -> dict[str, Any]:
        """运行或恢复整条长视频产品链路。"""
        video = self.ingestion.ingest(input_video)
        segments = self.segmenter.plan(video)
        job_paths = LongVideoJobPaths(
            self.scheduler_config.runtime_root, video.video_id
        )
        job_paths.job_root.mkdir(parents=True, exist_ok=True)
        state = self._new_state(video, segments)
        state["segmentation"] = {"status": "running", "started_utc": _utc_now()}
        _write_json_atomic(job_paths.state, state)
        try:
            self.segmenter.write_manifest(job_paths.segment_manifest, video, segments)
            exported_windows = self.segmenter.export(
                video,
                segments,
                job_paths.window_video_directory,
                ffmpeg_binary=self.scheduler_config.ffmpeg_binary,
                overwrite=self.scheduler_config.overwrite_windows,
            )
            state["segmentation"] = {
                "status": "succeeded",
                "finished_utc": _utc_now(),
                "manifest": str(job_paths.segment_manifest),
                "window_count": len(exported_windows),
            }
            _write_json_atomic(job_paths.state, state)
        except Exception as error:
            state["status"] = "failed"
            state["segmentation"] = {
                "status": "failed",
                "finished_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            _write_json_atomic(job_paths.state, state)
            raise

        predictions, failed_windows = self._process_windows(
            job_paths=job_paths,
            segments=segments,
            state=state,
        )
        if failed_windows and not self.scheduler_config.allow_partial:
            state.update(
                {
                    "status": "failed",
                    "failed_windows": failed_windows,
                    "updated_utc": _utc_now(),
                }
            )
            _write_json_atomic(job_paths.state, state)
            raise RuntimeError(
                "存在处理失败的固定窗口；重跑相同命令会只重试失败项："
                + ", ".join(failed_windows)
            )

        successful_segments = [
            segment for segment in segments if segment.segment_id in predictions
        ]
        timeline = self.timeline_service.build_timeline(
            successful_segments,
            predictions,
            source_duration_seconds=video.duration_seconds,
        )
        self.timeline_service.write_report(job_paths.timeline, timeline)
        materials = self.material_exporter.export(
            source_video=video.source_path,
            material_drafts=timeline["material_drafts"],
            output_directory=job_paths.final_materials,
        )

        identity_reports: dict[str, dict[str, Any]] = {}
        failed_identities: list[str] = []
        if self.scheduler_config.with_identity:
            identity_reports, failed_identities = self._process_identities(
                job_paths=job_paths,
                materials=materials,
                state=state,
            )

        database = ProductDatabase.open(job_paths.product_data)
        finalizer = MaterialFinalizationApplication(
            exporter=self.material_exporter,
            catalog=CatalogService(),
            database=database,
            associator=CrossMaterialIdentityAssociator(),
        )
        finalization = finalizer.run(
            source_video_id=video.video_id,
            source_video_path=video.source_path,
            timeline_report=timeline,
            output_directory=job_paths.final_materials,
            identity_reports=identity_reports,
            replace_database_records=True,
        )
        _write_json_atomic(job_paths.finalization_report, finalization)

        warning_count = len(failed_windows) + len(failed_identities)
        state.update(
            {
                "status": "completed_with_warnings" if warning_count else "completed",
                "failed_windows": failed_windows,
                "failed_identities": failed_identities,
                "timeline": str(job_paths.timeline),
                "finalization_report": str(job_paths.finalization_report),
                "database": str(database.path),
                "material_count": len(materials),
                "updated_utc": _utc_now(),
            }
        )
        _write_json_atomic(job_paths.state, state)
        return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析长视频端到端处理命令。"""
    parser = argparse.ArgumentParser(
        description="切分长视频、逐窗识别事件、导出最终素材并登记 SQLite。"
    )
    parser.add_argument("video", type=Path, help="待处理长视频路径")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=SETTINGS.project_root / "tests" / "long_video_runtime",
        help="测试运行数据根目录",
    )
    parser.add_argument("--window-seconds", type=float, default=12.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-tail-seconds", type=float, default=2.0)
    parser.add_argument("--max-attempts-per-run", type=int, default=2)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--with-identity", action="store_true")
    parser.add_argument("--identity-gpus", default="0")
    parser.add_argument("--roster-json", type=Path, default=None)
    parser.add_argument("--sam3-gpus", default="0,1")
    parser.add_argument("--playnet-gpu", type=int, default=0)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-windows", action="store_true")
    parser.add_argument("--bag-clips", type=int, default=12)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--fps-in", type=int, default=60)
    parser.add_argument("--fps-out", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeline-topk", type=int, default=2)
    parser.add_argument("--max-num-objects", type=int, default=10)
    parser.add_argument("--max-ball-objects", type=int, default=2)
    parser.add_argument("--sam3-num-maskmem", type=int, default=3)
    parser.add_argument("--sam3-max-cond-frames", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """构造默认模块并运行长视频调度器。"""
    args = parse_args(argv)
    pipeline_config = PipelineConfig(
        sam3_gpus=args.sam3_gpus,
        playnet_gpu=args.playnet_gpu,
        max_num_objects=args.max_num_objects,
        max_ball_objects=args.max_ball_objects,
        sam3_num_maskmem=args.sam3_num_maskmem,
        sam3_max_cond_frames=args.sam3_max_cond_frames,
        bag_clips=args.bag_clips,
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        image_size=args.img_size,
        topk=args.topk,
        timeline_topk=args.timeline_topk,
        resume=not args.no_resume,
    )
    scheduler_config = LongVideoSchedulerConfig(
        runtime_root=args.runtime_root,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        minimum_tail_seconds=args.minimum_tail_seconds,
        max_attempts_per_run=args.max_attempts_per_run,
        allow_partial=args.allow_partial,
        with_identity=args.with_identity,
        resume=not args.no_resume,
        overwrite_windows=args.overwrite_windows,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    identity_processor = (
        SubprocessMaterialIdentityProcessor(
            settings=SETTINGS,
            pipeline_config=pipeline_config,
            identity_gpus=args.identity_gpus,
            roster_json=args.roster_json,
            resume=not args.no_resume,
        )
        if args.with_identity
        else None
    )
    state = LongVideoScheduler(
        settings=SETTINGS,
        scheduler_config=scheduler_config,
        pipeline_config=pipeline_config,
        identity_processor=identity_processor,
    ).run(args.video)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
