"""Run the complete BasketEvent inference pipeline for one BARD video clip.

This entry point centralizes the server paths, environment variables, and
stage ordering that were previously repeated in shell commands. It executes
SAM3 tracking, Qwen trajectory filtering, PlayNet event inference, and the
diagnostic overlay in order. Existing non-empty stage outputs are reused by
default so an interrupted run can resume without repeating expensive work.

Qwen may reject every SAM3 player candidate. That is a valid diagnostic result
rather than a filesystem failure: PlayNet is skipped, a tracking-only overlay
is rendered, and the pipeline report records ``completed_with_warning``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import SETTINGS, Settings


_STAGE_ORDER = ("sam3", "qwen", "playnet", "visualize")


@dataclass(frozen=True)
class SingleVideoPaths:
    """Collect every input and output path for one BARD video clip.

    Attributes:
        project_root: Root of the BasketEvent source checkout.
        data_root: Root containing ``{game}/video/{clip}.mp4``.
        artifacts_root: Root for reusable intermediate and final outputs.
        game_id: BARD game directory name.
        clip_id: Video stem, for example ``130``.
        roster: Game-level roster JSON consumed by Qwen.
        sam3_checkpoint: Local SAM3 checkpoint path.
        sam3_bpe: SAM3 BPE vocabulary path.
        qwen_model: Local Qwen model directory.
        event_checkpoint: Author-provided PlayNet checkpoint.
        timesformer_model: Local TimeSformer directory.
    """

    project_root: Path
    data_root: Path
    artifacts_root: Path
    game_id: str
    clip_id: str
    roster: Path
    sam3_checkpoint: Path
    sam3_bpe: Path
    qwen_model: Path
    event_checkpoint: Path
    timesformer_model: Path

    @classmethod
    def from_settings(
        cls,
        game_id: str,
        clip_id: str,
        settings: Settings = SETTINGS,
        roster: Path | None = None,
    ) -> "SingleVideoPaths":
        """Build conventional server paths from central project settings.

        Args:
            game_id: BARD game directory name.
            clip_id: Source video stem without ``.mp4``.
            settings: Central path configuration.
            roster: Optional roster override. The generated game roster is
                used when omitted.

        Returns:
            Fully resolved path collection for the requested clip.
        """
        return cls(
            project_root=settings.project_root,
            data_root=settings.data_root,
            artifacts_root=settings.artifacts_root,
            game_id=game_id,
            clip_id=clip_id,
            roster=roster
            or settings.game_metadata_dir(game_id) / "recognize_roster.json",
            sam3_checkpoint=settings.sam3_checkpoint,
            sam3_bpe=settings.sam3_bpe,
            qwen_model=settings.qwen_model,
            event_checkpoint=settings.event_checkpoint,
            timesformer_model=settings.timesformer_model,
        )

    @property
    def video(self) -> Path:
        """Return the input BARD MP4 path."""
        return self.data_root / self.game_id / "video" / f"{self.clip_id}.mp4"

    @property
    def game_artifacts(self) -> Path:
        """Return the artifact directory for this game."""
        return self.artifacts_root / self.game_id

    @property
    def raw_tracks(self) -> Path:
        """Return the raw SAM3 trajectory JSON path."""
        return self.game_artifacts / "tracks" / "raw" / f"{self.clip_id}.json"

    @property
    def clean_tracks(self) -> Path:
        """Return the Qwen-filtered trajectory JSON path."""
        return self.game_artifacts / "tracks" / "clean" / f"{self.clip_id}.json"

    @property
    def prediction(self) -> Path:
        """Return the PlayNet event prediction JSON path."""
        return self.game_artifacts / "predictions" / f"{self.clip_id}_events.json"

    @property
    def visualization(self) -> Path:
        """Return the annotated MP4 path."""
        return self.game_artifacts / "visualizations" / f"{self.clip_id}_overlay.mp4"

    @property
    def visualization_report(self) -> Path:
        """Return the visualization audit JSON path."""
        return self.game_artifacts / "visualizations" / f"{self.clip_id}_overlay.json"

    @property
    def pipeline_report(self) -> Path:
        """Return the stage-level pipeline report JSON path."""
        return self.game_artifacts / "reports" / f"{self.clip_id}_pipeline.json"

    def stage_log(self, stage: str) -> Path:
        """Return the log path for one pipeline stage.

        Args:
            stage: Pipeline stage name.

        Returns:
            UTF-8 text log path below the game artifact directory.
        """
        return self.game_artifacts / "logs" / f"{self.clip_id}_{stage}.log"

    def create_output_directories(self) -> None:
        """Create all artifact directories used by the pipeline."""
        for path in (
            self.raw_tracks,
            self.clean_tracks,
            self.prediction,
            self.visualization,
            self.visualization_report,
            self.pipeline_report,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Store runtime parameters for one-video inference.

    Attributes:
        sam3_gpus: Comma-separated GPU IDs used by SAM3.
        qwen_gpus: Comma-separated GPU IDs exposed to Qwen.
        playnet_gpu: GPU ID used by TimeSformer and PlayNet.
        offload_video_to_cpu: Whether SAM3 stores decoded frames in CPU RAM.
        offload_state_to_cpu: Whether SAM3 stores reusable state in CPU RAM.
        max_num_objects: Maximum simultaneous SAM3 player tracks.
        max_ball_objects: Maximum simultaneous SAM3 ball tracks.
        sam3_num_maskmem: Number of SAM3 mask-memory frames.
        sam3_max_cond_frames: Maximum conditioning frames in memory attention.
        bag_clips: Number of sampled PlayNet clips.
        clip_len: Frames per sampled PlayNet clip.
        fps_in: Nominal source-video frame rate used by sampling.
        fps_out: Effective model sampling frame rate.
        image_size: Spatial TimeSformer input size.
        topk: Number of printed PlayNet predictions.
        timeline_topk: Evidence windows exported per predicted player-event.
        resume: Whether existing non-empty stage outputs may be reused.
        start_at: First stage to execute; earlier outputs must already exist.
        visualize_only: Whether to render existing outputs without running any
            model stage.
        dry_run: Whether commands are printed without execution.
    """

    sam3_gpus: str = "0,1"
    qwen_gpus: str = "0"
    playnet_gpu: int = 0
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    max_num_objects: int = 10
    max_ball_objects: int = 2
    sam3_num_maskmem: int = 3
    sam3_max_cond_frames: int = 2
    bag_clips: int = 12
    clip_len: int = 8
    fps_in: int = 60
    fps_out: int = 4
    image_size: int = 224
    topk: int = 5
    timeline_topk: int = 2
    resume: bool = True
    start_at: str = "sam3"
    visualize_only: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        """Validate parameters before expensive models are constructed."""
        if self.start_at not in _STAGE_ORDER:
            raise ValueError(f"Unknown start stage: {self.start_at}")
        positive_values = {
            "max_num_objects": self.max_num_objects,
            "max_ball_objects": self.max_ball_objects,
            "sam3_num_maskmem": self.sam3_num_maskmem,
            "sam3_max_cond_frames": self.sam3_max_cond_frames,
            "bag_clips": self.bag_clips,
            "clip_len": self.clip_len,
            "fps_in": self.fps_in,
            "fps_out": self.fps_out,
            "image_size": self.image_size,
            "topk": self.topk,
            "timeline_topk": self.timeline_topk,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")


class SingleVideoPipeline:
    """Coordinate all inference stages for one video without loading models.

    Each model remains owned by its original entry-point process. This keeps
    GPU memory lifetimes isolated: when SAM3 exits, its memory is released
    before Qwen starts, and likewise before PlayNet. The class is responsible
    only for paths, commands, resumption, validation, and audit reporting.
    """

    def __init__(self, paths: SingleVideoPaths, config: PipelineConfig) -> None:
        """Initialize a pipeline without performing filesystem or GPU work.

        Args:
            paths: Input, model, and artifact paths for the selected clip.
            config: Runtime and resumption parameters.
        """
        self.paths = paths
        self.config = config
        self.report: dict[str, Any] = {
            "schema_version": "basketevent_single_video_pipeline.v1",
            "game_id": paths.game_id,
            "clip_id": paths.clip_id,
            "video": str(paths.video),
            "status": "running",
            "accepted_player_count": None,
            "stages": {},
            "outputs": {
                "raw_tracks": str(paths.raw_tracks),
                "clean_tracks": str(paths.clean_tracks),
                "prediction": str(paths.prediction),
                "visualization": str(paths.visualization),
                "visualization_report": str(paths.visualization_report),
            },
        }

    @staticmethod
    def _is_nonempty_file(path: Path) -> bool:
        """Return whether a path is a regular file with at least one byte."""
        return path.is_file() and path.stat().st_size > 0

    def _stage_enabled(self, stage: str) -> bool:
        """Return whether a stage is at or after the requested start stage."""
        if self.config.visualize_only:
            return stage == "visualize"
        return _STAGE_ORDER.index(stage) >= _STAGE_ORDER.index(self.config.start_at)

    def _child_environment(self) -> dict[str, str]:
        """Build the clean environment inherited by every model subprocess."""
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        return environment

    def _validate_common_inputs(self) -> None:
        """Validate the source video and project entry points."""
        required_files = {
            "input video": self.paths.video,
            "SAM3 tracking module": self.paths.project_root
            / "src"
            / "modules"
            / "tracking"
            / "sam3_tracker.py",
            "Qwen recognition module": self.paths.project_root
            / "src"
            / "modules"
            / "identity"
            / "resolver.py",
            "event-recognition module": self.paths.project_root
            / "src"
            / "modules"
            / "event_recognition"
            / "inference.py",
            "visualization module": self.paths.project_root
            / "src"
            / "modules"
            / "materials"
            / "visualization.py",
        }
        for label, path in required_files.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

    def _run_command(self, stage: str, command: Sequence[str]) -> None:
        """Execute one command while streaming and persisting merged output.

        Args:
            stage: Human-readable pipeline stage name.
            command: Argument vector executed without a shell.

        Raises:
            subprocess.CalledProcessError: If the child process exits nonzero.
        """
        display_command = " ".join(str(part) for part in command)
        print(f"\n[{stage}] {display_command}", flush=True)
        self.report["stages"][stage] = {
            "status": "dry_run" if self.config.dry_run else "running",
            "command": [str(part) for part in command],
            "log": str(self.paths.stage_log(stage)),
        }
        if self.config.dry_run:
            return

        log_path = self.paths.stage_log(stage)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [str(part) for part in command],
                cwd=self.paths.project_root,
                env=self._child_environment(),
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
            self.report["stages"][stage]["status"] = "failed"
            self.report["stages"][stage]["return_code"] = return_code
            self._write_report()
            raise subprocess.CalledProcessError(return_code, command)
        self.report["stages"][stage]["status"] = "completed"

    def _reuse_or_run(
        self, stage: str, output_path: Path, command: Sequence[str]
    ) -> None:
        """Reuse a completed stage output or execute the stage command.

        Args:
            stage: Pipeline stage name.
            output_path: File proving that the stage completed.
            command: Command used when the output cannot be reused.
        """
        if not self._stage_enabled(stage):
            if not self._is_nonempty_file(output_path):
                mode = (
                    "--visualize-only"
                    if self.config.visualize_only
                    else f"--start-at {self.config.start_at}"
                )
                raise FileNotFoundError(
                    f"{mode} requires {stage} output: {output_path}"
                )
            self.report["stages"][stage] = {
                "status": "existing_prerequisite",
                "output": str(output_path),
            }
            return
        if self.config.resume and self._is_nonempty_file(output_path):
            self.report["stages"][stage] = {
                "status": "reused",
                "output": str(output_path),
            }
            print(f"[{stage}] Reusing {output_path}", flush=True)
            return
        self._run_command(stage, command)
        if not self.config.dry_run and not self._is_nonempty_file(output_path):
            raise RuntimeError(
                f"{stage} exited successfully but did not create: {output_path}"
            )

    def _sam3_command(self) -> list[str]:
        """Build the SAM3 tracking command."""
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.modules.tracking.sam3_tracker",
            "--video_path",
            str(self.paths.video),
            "--json_save_path",
            str(self.paths.raw_tracks),
            "--gpus_to_use",
            self.config.sam3_gpus,
            "--sam3_checkpoint",
            str(self.paths.sam3_checkpoint),
            "--sam3_bpe",
            str(self.paths.sam3_bpe),
            "--max-num-objects",
            str(self.config.max_num_objects),
            "--max-ball-objects",
            str(self.config.max_ball_objects),
            "--sam3-num-maskmem",
            str(self.config.sam3_num_maskmem),
            "--sam3-max-cond-frames",
            str(self.config.sam3_max_cond_frames),
        ]
        if self.config.offload_video_to_cpu:
            command.append("--offload-video-to-cpu")
        if self.config.offload_state_to_cpu:
            command.append("--offload-state-to-cpu")
        return command

    def _qwen_command(self) -> list[str]:
        """Build the Qwen trajectory-filtering command."""
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.modules.identity.resolver",
            "--video_path",
            str(self.paths.video),
            "--bbox_json_path",
            str(self.paths.raw_tracks),
            "--json_save_path",
            str(self.paths.clean_tracks),
            "--gpus_to_use",
            self.config.qwen_gpus,
            "--qwen_model",
            str(self.paths.qwen_model),
        ]
        if self.paths.roster.is_file():
            command.extend(["--roster_json", str(self.paths.roster)])
        return command

    def _playnet_command(self) -> list[str]:
        """Build the TimeSformer and PlayNet inference command."""
        return [
            sys.executable,
            "-u",
            "-m",
            "src.modules.event_recognition.inference",
            "--video",
            str(self.paths.video),
            "--traj_json",
            str(self.paths.clean_tracks),
            "--checkpoint",
            str(self.paths.event_checkpoint),
            "--timesformer_model",
            str(self.paths.timesformer_model),
            "--gpu_id",
            str(self.config.playnet_gpu),
            "--bag_clips",
            str(self.config.bag_clips),
            "--clip_len",
            str(self.config.clip_len),
            "--fps_in",
            str(self.config.fps_in),
            "--fps_out",
            str(self.config.fps_out),
            "--img_size",
            str(self.config.image_size),
            "--topk",
            str(self.config.topk),
            "--prediction_json_path",
            str(self.paths.prediction),
            "--timeline_topk",
            str(self.config.timeline_topk),
        ]

    def _visualization_command(self, include_prediction: bool) -> list[str]:
        """Build the trajectory and optional event-timeline rendering command.

        Args:
            include_prediction: Whether to load the PlayNet prediction JSON.

        Returns:
            Visualization subprocess argument vector.
        """
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.modules.materials.visualization",
            "--video_path",
            str(self.paths.video),
            "--raw_json_path",
            str(self.paths.raw_tracks),
            "--clean_json_path",
            str(self.paths.clean_tracks),
            "--output_video_path",
            str(self.paths.visualization),
            "--report_json_path",
            str(self.paths.visualization_report),
        ]
        if include_prediction:
            command.extend(["--prediction_json_path", str(self.paths.prediction)])
        return command

    def _accepted_player_count(self) -> int:
        """Count player objects retained in the Qwen-cleaned JSON.

        Returns:
            Number of top-level keys whose names begin with ``player_``.

        Raises:
            ValueError: If the clean JSON root is not an object.
        """
        with self.paths.clean_tracks.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Qwen clean JSON root must be an object: {self.paths.clean_tracks}"
            )
        return sum(str(key).startswith("player_") for key in value)

    def _write_report(self) -> None:
        """Persist the current pipeline state as formatted UTF-8 JSON."""
        self.paths.pipeline_report.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.pipeline_report.open("w", encoding="utf-8") as file:
            json.dump(self.report, file, ensure_ascii=False, indent=2)

    def run(self) -> dict[str, Any]:
        """Execute or resume all stages and return the final audit report.

        Returns:
            JSON-serializable stage and output report.
        """
        self._validate_common_inputs()
        self.paths.create_output_directories()

        try:
            self._reuse_or_run("sam3", self.paths.raw_tracks, self._sam3_command())
            self._reuse_or_run("qwen", self.paths.clean_tracks, self._qwen_command())

            if self.config.dry_run:
                accepted_player_count = None
            else:
                accepted_player_count = self._accepted_player_count()
            self.report["accepted_player_count"] = accepted_player_count

            if self.config.visualize_only:
                include_prediction = (
                    accepted_player_count != 0
                    and self._is_nonempty_file(self.paths.prediction)
                )
                self.report["stages"]["playnet"] = {
                    "status": "skipped",
                    "reason": "Visualization-only mode",
                }
                self._reuse_or_run(
                    "visualize",
                    self.paths.visualization,
                    self._visualization_command(include_prediction),
                )
                self.report["visualization_mode"] = (
                    "tracks_and_events" if include_prediction else "tracks_only"
                )
                self.report["status"] = (
                    "dry_run" if self.config.dry_run else "completed"
                )
            elif accepted_player_count == 0:
                self.report["stages"]["playnet"] = {
                    "status": "skipped",
                    "reason": "Qwen retained no player trajectories",
                }
                self._reuse_or_run(
                    "visualize",
                    self.paths.visualization,
                    self._visualization_command(include_prediction=False),
                )
                self.report["status"] = "completed_with_warning"
                self.report["warning"] = (
                    "Qwen retained no players; PlayNet was skipped and the "
                    "overlay contains SAM3/Qwen diagnostics only."
                )
            else:
                self._reuse_or_run(
                    "playnet", self.paths.prediction, self._playnet_command()
                )
                include_prediction = self.config.dry_run or self._is_nonempty_file(
                    self.paths.prediction
                )
                self._reuse_or_run(
                    "visualize",
                    self.paths.visualization,
                    self._visualization_command(include_prediction),
                )
                self.report["status"] = (
                    "dry_run" if self.config.dry_run else "completed"
                )
        except Exception:
            self.report["status"] = "failed"
            self._write_report()
            raise

        self._write_report()
        print(json.dumps(self.report, ensure_ascii=False, indent=2))
        return self.report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the single-video pipeline command-line interface.

    Args:
        argv: Optional test argument sequence. ``None`` reads process arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run SAM3, Qwen, PlayNet, and visualization for one clip."
    )
    parser.add_argument("--game", required=True, help="BARD game directory name.")
    parser.add_argument("--clip", required=True, help="Video stem without .mp4.")
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--sam3-gpus", default="0,1")
    parser.add_argument("--qwen-gpus", default="0")
    parser.add_argument("--playnet-gpu", type=int, default=0)
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-num-objects", type=int, default=10)
    parser.add_argument("--max-ball-objects", type=int, default=2)
    parser.add_argument("--sam3-num-maskmem", type=int, default=3)
    parser.add_argument("--sam3-max-cond-frames", type=int, default=2)
    parser.add_argument("--bag-clips", type=int, default=12)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--fps-in", type=int, default=60)
    parser.add_argument("--fps-out", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--timeline-topk", type=int, default=2)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing non-empty outputs (default: true).",
    )
    parser.add_argument(
        "--start-at",
        choices=_STAGE_ORDER,
        default="sam3",
        help="Start at a later stage when earlier outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write a report without loading any model.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help=(
            "Render an existing raw/clean result without loading SAM3, Qwen, "
            "or PlayNet. An existing prediction JSON is added automatically."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve paths, build the pipeline, and process one video clip."""
    args = parse_args(argv)
    paths = SingleVideoPaths.from_settings(
        game_id=args.game,
        clip_id=args.clip,
        roster=args.roster,
    )
    config = PipelineConfig(
        sam3_gpus=args.sam3_gpus,
        qwen_gpus=args.qwen_gpus,
        playnet_gpu=args.playnet_gpu,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
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
        resume=args.resume,
        start_at=args.start_at,
        visualize_only=args.visualize_only,
        dry_run=args.dry_run,
    )
    SingleVideoPipeline(paths, config).run()


if __name__ == "__main__":
    main()
