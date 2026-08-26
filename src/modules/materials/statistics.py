"""Summarize event predictions as searchable clip-material statistics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MaterialStatistics:
    """Aggregate counts from one or more processed clip reports."""

    clip_count: int
    player_prediction_count: int
    non_background_prediction_count: int
    temporal_event_count: int
    mean_confidence: float | None
    event_counts: dict[str, int]
    participant_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable statistics."""
        return asdict(self)


class MaterialStatisticsService:
    """Build deterministic catalog statistics from PlayNet JSON reports."""

    @staticmethod
    def _participant_label(prediction: Mapping[str, Any]) -> str:
        """Build a display label without requiring a player roster."""
        color = str(prediction.get("jersey_color") or "unknown").lower()
        number = prediction.get("jersey_number")
        if number is not None and str(number).strip():
            return f"{color} #{str(number).strip()}"
        return str(prediction.get("player_id") or f"{color} unknown")

    def summarize(
        self,
        reports: Sequence[Mapping[str, Any]],
    ) -> MaterialStatistics:
        """Summarize already-loaded prediction reports.

        Args:
            reports: Documents following the temporal prediction schema emitted
                by the event-recognition module.

        Returns:
            Counts suitable for a future material-catalog page.
        """
        event_counts: Counter[str] = Counter()
        participant_counts: Counter[str] = Counter()
        confidences: list[float] = []
        prediction_count = 0
        non_background_count = 0
        temporal_event_count = 0

        for report in reports:
            predictions = report.get("player_predictions", [])
            if not isinstance(predictions, list):
                raise ValueError("player_predictions must be a list")
            temporal_events = report.get("temporal_events", [])
            if not isinstance(temporal_events, list):
                raise ValueError("temporal_events must be a list")
            temporal_event_count += len(temporal_events)
            for prediction in predictions:
                if not isinstance(prediction, Mapping):
                    raise ValueError("Each player prediction must be an object")
                prediction_count += 1
                event = str(prediction.get("event") or "unknown")
                event_counts[event] += 1
                participant_counts[self._participant_label(prediction)] += 1
                if event != "blank":
                    non_background_count += 1
                confidence = prediction.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))

        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return MaterialStatistics(
            clip_count=len(reports),
            player_prediction_count=prediction_count,
            non_background_prediction_count=non_background_count,
            temporal_event_count=temporal_event_count,
            mean_confidence=mean_confidence,
            event_counts=dict(sorted(event_counts.items())),
            participant_counts=dict(sorted(participant_counts.items())),
        )

    def summarize_files(
        self,
        report_paths: Sequence[str | Path],
    ) -> MaterialStatistics:
        """Load prediction JSON files and summarize them."""
        reports: list[Mapping[str, Any]] = []
        for path_value in report_paths:
            path = Path(path_value).expanduser()
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
            if not isinstance(value, Mapping):
                raise ValueError(f"Prediction report root must be an object: {path}")
            reports.append(value)
        return self.summarize(reports)
