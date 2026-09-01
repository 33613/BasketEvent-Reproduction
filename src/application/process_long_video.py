"""把一个长视频自动处理成事件素材并登记到产品数据库。

该应用只负责编排已经存在的模块：视频接入、固定窗口、单窗口模型链路、
全局事件时间线、最终素材导出和可选身份后处理。每个窗口的状态都会持久化，
因此 SSH 中断后可以直接重跑同一命令继续处理。
"""

from __future__ import annotations

import argparse
import json
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
from src.modules.identity import (
    CrossMaterialIdentityAssociator,
    EventActorIdentityService,
)
from src.modules.identity.resolver import IdentityResolver, RosterLookup
from src.modules.ingestion import VideoAsset, VideoIngestionService
from src.modules.materials import MaterialExporter
from src.modules.segmentation import LongVideoSegmenter, VideoSegment


def _utc_now() -> str:
    """返回便于写入 JSON 的 UTC 时间。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _result_affecting_pipeline_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """返回影响模型结果的配置，排除可在续跑时调整的 GPU 分配。

    ``sam3_gpus`` 和 ``playnet_gpu`` 只决定任务在哪张物理显卡上运行，
    不改变模型、采样或推理参数。因此服务器资源变化后允许改用空闲 GPU，
    同时仍拒绝在同一任务目录内混用 ``topk`` 等结果相关参数。
    """
    result_configuration = dict(configuration)
    result_configuration.pop("sam3_gpus", None)
    result_configuration.pop("playnet_gpu", None)
    return result_configuration


class WindowProcessor(Protocol):
    """约束单个固定窗口处理器。"""

    def run(self, paths: SingleVideoPaths, config: PipelineConfig) -> Mapping[str, Any]:
        """运行一个窗口并返回流水线报告。"""
        ...


class EventIdentityProcessor(Protocol):
    """约束基于窗口缓存的事件主体身份处理器。"""

    def run(
        self,
        *,
        source_video_id: str,
        timeline_report: Mapping[str, Any],
        window_video_directory: Path,
        raw_tracks_directory: Path,
        cache_directory: Path,
        output_path: Path,
    ) -> Path:
        """处理全部非空事件主体，并返回一份身份报告路径。"""
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
    def raw_tracks_directory(self) -> Path:
        """返回15个固定窗口已经缓存的SAM3原始轨迹目录。"""
        return self.window_artifacts_root / self.video_id / "tracks" / "raw"

    @property
    def timeline(self) -> Path:
        """返回源视频全局事件时间线。"""
        return self.job_root / "event_timeline.json"

    @property
    def final_materials(self) -> Path:
        """返回最终事件素材目录。"""
        return self.job_root / "final_materials"

    @property
    def event_identity(self) -> Path:
        """返回事件主体身份汇总报告。"""
        return self.job_root / "event_identity.json"

    @property
    def event_identity_tracks(self) -> Path:
        """返回唯一窗口轨迹的Qwen逐帧证据缓存目录。"""
        return self.job_root / "event_identity_tracks"

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
    identity_num_crops: int = 10
    identity_pad_ratio: float = 0.0
    resume: bool = True
    overwrite_windows: bool = False
    ffmpeg_binary: str = "ffmpeg"

    def __post_init__(self) -> None:
        """在启动昂贵模型前检查调度参数。"""
        if self.max_attempts_per_run <= 0:
            raise ValueError("max_attempts_per_run 必须为正数")
        if self.identity_num_crops <= 0:
            raise ValueError("identity_num_crops 必须为正数")
        if self.identity_pad_ratio < 0:
            raise ValueError("identity_pad_ratio 不能为负数")
        if not self.resume and not self.overwrite_windows:
            raise ValueError("关闭断点续跑时必须同时覆盖固定窗口")


class CachedWindowEventIdentityProcessor:
    """复用窗口SAM3 bbox，并在一次模型加载中识别全部事件主体。"""

    def __init__(
        self,
        *,
        settings: Settings,
        identity_gpus: str = "0",
        roster_json: Path | None = None,
        sample_count: int = 10,
        pad_ratio: float = 0.0,
    ) -> None:
        """保存Qwen设备、可选名单和事件轨迹取样参数。"""
        self.settings = settings
        self.identity_gpus = identity_gpus
        self.roster_json = roster_json
        self.sample_count = sample_count
        self.pad_ratio = pad_ratio

    def run(
        self,
        *,
        source_video_id: str,
        timeline_report: Mapping[str, Any],
        window_video_directory: Path,
        raw_tracks_directory: Path,
        cache_directory: Path,
        output_path: Path,
    ) -> Path:
        """加载一次Qwen，依次处理时间线引用的唯一人物轨迹。"""
        import torch
        from src.modules.identity.qwen_observer import QwenTrackObserver
        from src.modules.identity.sampling import TrackSampler

        model_path = self.settings.require_directory(
            self.settings.qwen_model, "Qwen模型"
        )
        device = (
            f"cuda:{str(self.identity_gpus).split(',')[0]}"
            if torch.cuda.is_available()
            else "cpu"
        )
        observer = QwenTrackObserver.from_pretrained(
            str(model_path),
            device=device,
            local_files_only=self.settings.hf_local_files_only,
        )
        roster = RosterLookup.from_file(self.roster_json)
        service = EventActorIdentityService(
            sampler=TrackSampler(
                sample_count=self.sample_count,
                pad_ratio=self.pad_ratio,
            ),
            observer=observer,
            resolver=IdentityResolver(roster),
        )
        report = service.process(
            source_video_id=source_video_id,
            timeline_report=timeline_report,
            window_video_directory=window_video_directory,
            raw_tracks_directory=raw_tracks_directory,
            cache_directory=cache_directory,
        )
        _write_json_atomic(output_path, report)
        return output_path


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
        identity_processor: EventIdentityProcessor | None = None,
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
        previous_pipeline_configuration = previous_configuration.get("pipeline", {})
        if not isinstance(previous_pipeline_configuration, Mapping):
            previous_pipeline_configuration = {}
        result_configuration = _result_affecting_pipeline_configuration(
            pipeline_configuration
        )
        previous_result_configuration = _result_affecting_pipeline_configuration(
            previous_pipeline_configuration
        )
        if state and (
            previous_configuration.get("windows") != window_configuration
            or previous_result_configuration != result_configuration
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
                    "identity": {
                        "sample_count": self.scheduler_config.identity_num_crops,
                        "pad_ratio": self.scheduler_config.identity_pad_ratio,
                    },
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
            state.pop("identity", None)
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

    def _process_event_identity(
        self,
        *,
        job_paths: LongVideoJobPaths,
        timeline_report: Mapping[str, Any],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """复用窗口轨迹处理所有事件主体；失败时保留无身份事件素材。"""
        if self.identity_processor is None:
            raise RuntimeError("with_identity=True 但没有配置身份处理器")
        identity_state = state.setdefault(
            "identity", {"status": "pending", "attempt_count": 0}
        )
        report_value = identity_state.get("report")
        report_path = Path(str(report_value)) if report_value else None
        if (
            self.scheduler_config.resume
            and identity_state.get("status") in {"succeeded", "completed_with_warnings"}
            and report_path is not None
            and report_path.is_file()
        ):
            existing_report = _read_json_object(report_path)
            if (
                existing_report.get("sample_count_per_track_reference")
                == self.scheduler_config.identity_num_crops
                and float(existing_report.get("pad_ratio", -1.0))
                == self.scheduler_config.identity_pad_ratio
                and int(existing_report.get("failed_track_reference_count") or 0) == 0
            ):
                return existing_report, []

        for _ in range(self.scheduler_config.max_attempts_per_run):
            identity_state["status"] = "running"
            identity_state["attempt_count"] = (
                int(identity_state.get("attempt_count", 0)) + 1
            )
            identity_state["started_utc"] = _utc_now()
            _write_json_atomic(job_paths.state, state)
            try:
                report_path = self.identity_processor.run(
                    source_video_id=job_paths.video_id,
                    timeline_report=timeline_report,
                    window_video_directory=job_paths.window_video_directory,
                    raw_tracks_directory=job_paths.raw_tracks_directory,
                    cache_directory=job_paths.event_identity_tracks,
                    output_path=job_paths.event_identity,
                )
                report = _read_json_object(report_path)
                failed_references = [
                    str(reference)
                    for reference, value in report.get("track_evidence", {}).items()
                    if isinstance(value, Mapping) and value.get("status") != "completed"
                ]
                identity_state.update(
                    {
                        "status": (
                            "completed_with_warnings"
                            if failed_references
                            else "succeeded"
                        ),
                        "report": str(report_path),
                        "finished_utc": _utc_now(),
                    }
                )
                identity_state.pop("error", None)
                _write_json_atomic(job_paths.state, state)
                return report, failed_references
            except Exception as error:
                identity_state.update(
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
        return None, ["event_actor_identity"]

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

        event_identity_report: dict[str, Any] | None = None
        failed_identities: list[str] = []
        if self.scheduler_config.with_identity:
            event_identity_report, failed_identities = self._process_event_identity(
                job_paths=job_paths,
                timeline_report=timeline,
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
            event_identity_report=event_identity_report,
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
    parser.add_argument("--identity-num-crops", type=int, default=10)
    parser.add_argument("--identity-pad-ratio", type=float, default=0.0)
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
        identity_num_crops=args.identity_num_crops,
        identity_pad_ratio=args.identity_pad_ratio,
        resume=not args.no_resume,
        overwrite_windows=args.overwrite_windows,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    identity_processor = (
        CachedWindowEventIdentityProcessor(
            settings=SETTINGS,
            identity_gpus=args.identity_gpus,
            roster_json=args.roster_json,
            sample_count=args.identity_num_crops,
            pad_ratio=args.identity_pad_ratio,
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
