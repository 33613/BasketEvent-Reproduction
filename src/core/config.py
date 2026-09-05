"""Central path and runtime configuration for BasketEvent.

Every path can be overridden with an environment variable.  Entry-point
scripts expose the same values as command-line options, so their precedence is
command line, environment, then the server-oriented defaults in this module.
Importing this module never creates directories or loads model weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _environment_path(name: str, default: Path) -> Path:
    """Return an expanded path from an environment variable or a default.

    Args:
        name: Environment variable name.
        default: Path used when the variable is unset or empty.

    Returns:
        An absolute-looking, user-expanded path.  The path is not required to
        exist because output directories may be created later.
    """
    value = os.getenv(name)
    path = Path(value) if value else default
    return path.expanduser()


def _environment_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean value from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset or empty.

    Returns:
        The parsed boolean value.

    Raises:
        ValueError: If the value is not a recognized boolean spelling.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


@dataclass(frozen=True)
class Settings:
    """Store portable paths and safe runtime defaults for BasketEvent.

    Attributes:
        project_root: Root of the checked-out BasketEvent repository.
        data_root: Human-readable BARD staging dataset.
        artifacts_root: Expensive, reusable SAM3/Qwen intermediate outputs.
        runtime_root: Dataset arranged as videos/train/valid/test for the
            original BasketEvent data loader.
        model_root: Root containing all downloaded model weights.
        gpu_ids: Comma-separated visible GPU indices used by single-stage
            scripts unless a command-line option overrides them.
        hf_local_files_only: Whether Hugging Face loaders must avoid network
            access.  This defaults to true for the offline laboratory server.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    data_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_DATA_ROOT", Path("/home/fangzilin/data/basket")
        )
    )
    artifacts_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_ARTIFACTS_ROOT",
            Path("/home/fangzilin/data/basket_artifacts"),
        )
    )
    runtime_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_RUNTIME_ROOT",
            Path("/home/fangzilin/data/basket_runtime"),
        )
    )
    product_data_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_PRODUCT_DATA_ROOT",
            Path(__file__).resolve().parents[2] / "product_data",
        )
    )
    model_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_MODEL_ROOT", Path("/home/fangzilin/models")
        )
    )
    gpu_ids: str = field(default_factory=lambda: os.getenv("BASKETEVENT_GPU_IDS", "0"))
    hf_local_files_only: bool = field(
        default_factory=lambda: _environment_bool(
            "BASKETEVENT_HF_LOCAL_FILES_ONLY", True
        )
    )

    @property
    def sam3_source_root(self) -> Path:
        """Return the SAM3 Git submodule directory."""
        return self.project_root / "sam3"

    @property
    def sam3_checkpoint(self) -> Path:
        """Return the local SAM3 checkpoint."""
        return self.model_root / "sam3" / "sam3.pt"

    @property
    def sam3_bpe(self) -> Path:
        """Return the tokenizer vocabulary shipped in the SAM3 submodule."""
        return (
            self.sam3_source_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        )

    @property
    def qwen_model(self) -> Path:
        """Return the local Qwen2.5-VL-7B-Instruct model directory."""
        return self.model_root / "Qwen2.5-VL-7B-Instruct"

    @property
    def timesformer_model(self) -> Path:
        """Return the local Kinetics-400 TimeSformer model directory."""
        return self.model_root / "timesformer-base-finetuned-k400"

    @property
    def event_checkpoint(self) -> Path:
        """Return the author-provided BasketEvent recognition checkpoint."""
        return self.model_root / "basketevent" / "playnet.pt"

    @property
    def product_database_path(self) -> Path:
        """返回产品人物库与素材库共用的 SQLite 数据库路径。"""
        return self.product_data_root / "database" / "basketevent.sqlite3"

    @property
    def product_media_root(self) -> Path:
        """返回用户视频及生成素材的产品媒体根目录。"""
        return self.product_data_root / "media"

    @property
    def videos_dir(self) -> Path:
        """Return the runtime video directory consumed by the data loader."""
        return self.runtime_root / "videos"

    @property
    def train_annotations_dir(self) -> Path:
        """Return the runtime training-annotation directory."""
        return self.runtime_root / "train"

    @property
    def valid_annotations_dir(self) -> Path:
        """Return the runtime validation-annotation directory."""
        return self.runtime_root / "valid"

    @property
    def test_annotations_dir(self) -> Path:
        """Return the runtime test-annotation directory."""
        return self.runtime_root / "test"

    @property
    def cache_dir(self) -> Path:
        """Return the disposable dataset-index cache directory."""
        return self.runtime_root / "cache"

    @property
    def trained_checkpoints_dir(self) -> Path:
        """Return the default directory for newly trained checkpoints."""
        return self.model_root / "basketevent-trained"

    @property
    def split_config(self) -> Path:
        """Return the game-level train/validation/test split definition."""
        return self.artifacts_root / "split_config.json"

    @property
    def annotation_summary(self) -> Path:
        """Return the latest BARD annotation-build summary path."""
        return self.artifacts_root / "annotation_build_summary.json"

    @property
    def roster_summary(self) -> Path:
        """Return the latest BARD roster-conversion summary path."""
        return self.artifacts_root / "roster_build_summary.json"

    def game_artifacts_dir(self, bard_game: str) -> Path:
        """Return the reusable artifact directory for one BARD game.

        Args:
            bard_game: BARD folder name, for example
                ``bkn-vs-det-0022400861``.

        Returns:
            Directory that groups metadata, trajectories, annotations, and
            reports belonging to the game.
        """
        return self.artifacts_root / bard_game

    def raw_tracks_dir(self, bard_game: str) -> Path:
        """Return the SAM3 raw-trajectory directory for one BARD game."""
        return self.game_artifacts_dir(bard_game) / "tracks" / "raw"

    def clean_tracks_dir(self, bard_game: str) -> Path:
        """Return the Qwen-cleaned trajectory directory for one BARD game."""
        return self.game_artifacts_dir(bard_game) / "tracks" / "clean"

    def annotations_dir(self, bard_game: str) -> Path:
        """Return accepted BasketEvent annotations for one BARD game."""
        return self.game_artifacts_dir(bard_game) / "annotations"

    def annotation_reports_dir(self, bard_game: str) -> Path:
        """Return per-clip label-mapping and anomaly reports for one game."""
        return self.game_artifacts_dir(bard_game) / "reports"

    def game_metadata_dir(self, bard_game: str) -> Path:
        """Return generated roster and other game-level metadata paths."""
        return self.game_artifacts_dir(bard_game) / "metadata"

    def require_file(self, path: str | Path, description: str) -> Path:
        """Validate and return a required file path.

        Args:
            path: Candidate file path.
            description: Human-readable name used in an error message.

        Returns:
            The expanded path.

        Raises:
            FileNotFoundError: If the path is not a regular file.
        """
        candidate = Path(path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"{description} not found: {candidate}")
        return candidate

    def require_directory(self, path: str | Path, description: str) -> Path:
        """Validate and return a required directory path.

        Args:
            path: Candidate directory path.
            description: Human-readable name used in an error message.

        Returns:
            The expanded path.

        Raises:
            NotADirectoryError: If the path is not a directory.
        """
        candidate = Path(path).expanduser()
        if not candidate.is_dir():
            raise NotADirectoryError(f"{description} not found: {candidate}")
        return candidate

    @staticmethod
    def create_directories(paths: Iterable[str | Path]) -> None:
        """Create explicitly requested output directories.

        Args:
            paths: Directories to create, including missing parents.
        """
        for path in paths:
            Path(path).expanduser().mkdir(parents=True, exist_ok=True)

    @staticmethod
    def create_parent(path: str | Path) -> None:
        """Create the parent directory for an output file.

        Args:
            path: Output file whose parent should exist.
        """
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
