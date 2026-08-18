"""BARD-to-BasketEvent data preparation utilities.

The package contains deterministic data rules only. It deliberately does not
load SAM3, Qwen, TimeSformer, or any other learned model, which keeps annotation
generation reproducible and independently testable.
"""

from .labeling import BardAnnotationBuilder, BardLabelMapper
from .roster import BardRosterAdapter

__all__ = ["BardAnnotationBuilder", "BardLabelMapper", "BardRosterAdapter"]
