"""Use cases that coordinate BasketEvent modules without implementing models."""

from src.application.process_clip import (
    PipelineConfig,
    SingleVideoPaths,
    SingleVideoPipeline,
)

__all__ = ["PipelineConfig", "SingleVideoPaths", "SingleVideoPipeline"]
