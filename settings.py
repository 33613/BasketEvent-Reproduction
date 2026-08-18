"""Central runtime settings for BasketEvent experiments."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Stores portable paths used by the BasketEvent pipeline."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    data_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("BASKETEVENT_DATA_ROOT", "/home/fangzilin/data/basket")
        )
    )
    artifacts_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BASKETEVENT_ARTIFACTS_ROOT",
                "/home/fangzilin/data/basket_artifacts",
            )
        )
    )
    runtime_root: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BASKETEVENT_RUNTIME_ROOT",
                "/home/fangzilin/data/basket_runtime",
            )
        )
    )
    model_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("BASKETEVENT_MODEL_ROOT", "/home/fangzilin/models")
        )
    )

    @property
    def sam3_checkpoint(self) -> Path:
        """Returns the local SAM3 checkpoint path."""
        return self.model_root / "sam3" / "sam3.pt"

    @property
    def sam3_bpe(self) -> Path:
        """Returns the SAM3 tokenizer vocabulary path."""
        return (
            self.project_root
            / "sam3"
            / "sam3"
            / "assets"
            / "bpe_simple_vocab_16e6.txt.gz"
        )

    @property
    def qwen_model(self) -> Path:
        """Returns the local Qwen model directory."""
        return self.model_root / "Qwen2.5-VL-7B-Instruct"

    @property
    def timesformer_model(self) -> Path:
        """Returns the local TimeSformer model directory."""
        return self.model_root / "timesformer-base-finetuned-k400"

    @property
    def event_checkpoint(self) -> Path:
        """Returns the author-provided event-recognition checkpoint."""
        return self.model_root / "basketevent" / "playnet.pt"


SETTINGS = Settings()