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
        default_factory=lambda: Path(__file__).resolve().parent
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
    model_root: Path = field(
        default_factory=lambda: _environment_path(
            "BASKETEVENT_MODEL_ROOT", Path("/home/fangzilin/models")
        )
    )
    gpu_ids: str = field(
        default_factory=lambda: os.getenv("BASKETEVENT_GPU_IDS", "0")
    )
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
            self.sam3_source_root
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
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
