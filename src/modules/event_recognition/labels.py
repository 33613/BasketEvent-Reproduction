"""Canonical BasketEvent event-label vocabulary.

This module has no PyTorch dependency so data preparation, validation, and the
training dataset can share exactly one class-to-index definition.
"""

LABEL_MAP = {
    "blank": 0,
    "Missed Shot": 1,
    "Made Shot": 2,
    "Free Throw": 3,
    "Foul": 4,
    "Turnover": 5,
    "Jump Ball": 6,
    "Rebound": 7,
    "steal": 8,
    "block": 9,
    "ast": 10,
}

SUPPORTED_LABELS = frozenset(LABEL_MAP)
