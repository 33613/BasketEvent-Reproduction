"""Build single-label BasketEvent annotations from structured BARD actions.

Label generation is intentionally deterministic. No language or vision model
is used to decide the event class. BARD's structured action fields are mapped
through the explicit rules in :class:`BardLabelMapper`, then joined to cleaned
player trajectories by the exact key ``(jersey_color, jersey_number)``.

Scheme A keeps a clip only when every ground-truth actor can be matched to one
track and each actor has at most one distinct BasketEvent class. Repeated
instances of the same class are collapsed with a warning; conflicting classes
exclude the clip because the original BasketEvent loader accepts one integer
label per player and clip.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.modules.event_recognition.labels import SUPPORTED_LABELS


@dataclass(frozen=True)
class AnnotationAnomaly:
    """Describe one mapping, identity, or source-data anomaly."""

    code: str
    severity: str
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the anomaly as JSON-compatible data."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


@dataclass(frozen=True)
class LabelContribution:
    """Represent one deterministic label assigned to a visible identity."""

    jersey_color: str
    jersey_number: str
    label: str
    source_action_index: int
    role: str

    @property
    def identity_key(self) -> tuple[str, str]:
        """Return the normalized color-and-number identity key."""
        return (self.jersey_color, self.jersey_number)

    def to_dict(self) -> dict[str, Any]:
        """Return the contribution as JSON-compatible data."""
        return {
            "jersey_color": self.jersey_color,
            "jersey_number": self.jersey_number,
            "label": self.label,
            "source_action_index": self.source_action_index,
            "role": self.role,
        }


@dataclass(frozen=True)
class MappingResult:
    """Hold deterministic action mapping results before track association."""

    contributions: tuple[LabelContribution, ...]
    anomalies: tuple[AnnotationAnomaly, ...]

    @property
    def accepted(self) -> bool:
        """Return whether mapping produced no exclusion-level anomaly."""
        return not any(item.severity == "error" for item in self.anomalies)


@dataclass(frozen=True)
class AnnotationResult:
    """Hold a final annotation and its complete audit report."""

    annotation: dict[str, Any] | None
    report: dict[str, Any]

    @property
    def accepted(self) -> bool:
        """Return whether the clip was retained by Scheme A."""
        return self.annotation is not None


def normalize_color(value: Any) -> str | None:
    """Normalize a jersey color without guessing aliases.

    Args:
        value: Color value produced by BARD or visual recognition.

    Returns:
        Lowercase whitespace-normalized color, or ``None`` when missing.
    """
    if value is None:
        return None
    normalized = " ".join(str(value).strip().lower().split())
    return normalized or None


def normalize_jersey_number(value: Any) -> str | None:
    """Normalize a jersey number while preserving the distinction 0/00.

    Args:
        value: Integer or string jersey number.

    Returns:
        One- or two-digit string, or ``None`` for an invalid value.

    Notes:
        ``00`` and ``0`` are different NBA jersey numbers. A model output of
        integer ``0`` is therefore never silently matched to BARD actor ``00``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, float) and value.is_integer():
        normalized = str(int(value))
    else:
        normalized = str(value).strip()
    return normalized if re.fullmatch(r"\d{1,2}", normalized) else None


class BardLabelMapper:
    """Map BARD action objects to the author's 11-class label vocabulary.

    Mapping rules:

    * A 2PT/3PT shot with ``result=true`` becomes ``Made Shot``.
    * A 2PT/3PT shot with ``result=false`` becomes ``Missed Shot``.
    * A made or missed free throw remains ``Free Throw`` because the original
      BasketEvent vocabulary does not distinguish its result.
    * Foul, Turnover, Jump Ball, and Rebound keep their class names.
    * Steal and Block use lowercase ``steal``/``block`` to match
      ``src.labels.LABEL_MAP`` exactly.
    * A shot with ``assisted=true`` creates a second ``ast`` contribution for
      ``other_player``. No other ``other_player`` relationship creates a
      BasketEvent class.
    * Unknown actions are never converted to ``blank``; they exclude the clip
      and are recorded as ``UNMAPPED_ACTION``.
    """

    _DIRECT_LABELS = {
        "free throw": "Free Throw",
        "foul": "Foul",
        "turnover": "Turnover",
        "jump ball": "Jump Ball",
        "rebound": "Rebound",
        "steal": "steal",
        "block": "block",
    }
    _SHOT_ACTIONS = frozenset({"2pt shot", "3pt shot"})

    def map_document(self, document: Mapping[str, Any]) -> MappingResult:
        """Map and validate one BARD structured-action document.

        Args:
            document: Parsed ``description/action/<clip>.json`` document.

        Returns:
            Label contributions and all mapping anomalies.
        """
        anomalies: list[AnnotationAnomaly] = []
        contributions: list[LabelContribution] = []
        actions = document.get("actions")
        if not isinstance(actions, list):
            return MappingResult(
                (),
                (
                    AnnotationAnomaly(
                        code="INVALID_ACTION_DOCUMENT",
                        severity="error",
                        message="The BARD action document must contain an actions list.",
                        context={},
                    ),
                ),
            )

        numerosity = document.get("numerosity")
        if isinstance(numerosity, int) and numerosity != len(actions):
            anomalies.append(
                AnnotationAnomaly(
                    code="NUMEROSITY_MISMATCH",
                    severity="error",
                    message="BARD numerosity does not match the number of actions.",
                    context={"numerosity": numerosity, "action_count": len(actions)},
                )
            )
        if not actions:
            anomalies.append(
                AnnotationAnomaly(
                    code="NO_STRUCTURED_ACTION",
                    severity="error",
                    message="Scheme A requires at least one structured BARD action.",
                    context={},
                )
            )

        for index, raw_action in enumerate(actions):
            if not isinstance(raw_action, Mapping):
                anomalies.append(
                    AnnotationAnomaly(
                        code="INVALID_ACTION",
                        severity="error",
                        message="Skipped an action that is not an object.",
                        context={"source_action_index": index},
                    )
                )
                continue
            mapped, action_anomalies = self._map_action(index, raw_action)
            contributions.extend(mapped)
            anomalies.extend(action_anomalies)

        grouped: dict[tuple[str, str], list[LabelContribution]] = {}
        for contribution in contributions:
            grouped.setdefault(contribution.identity_key, []).append(contribution)

        collapsed: list[LabelContribution] = []
        for identity, items in grouped.items():
            labels = sorted({item.label for item in items})
            if len(labels) > 1:
                anomalies.append(
                    AnnotationAnomaly(
                        code="MULTI_LABEL_ACTOR",
                        severity="error",
                        message=(
                            "Scheme A excludes a clip when one player has more "
                            "than one distinct event class."
                        ),
                        context={
                            "jersey_color": identity[0],
                            "jersey_number": identity[1],
                            "labels": labels,
                            "source_action_indices": [
                                item.source_action_index for item in items
                            ],
                        },
                    )
                )
                continue
            collapsed.append(items[0])
            if len(items) > 1:
                anomalies.append(
                    AnnotationAnomaly(
                        code="DUPLICATE_LABEL_COLLAPSED",
                        severity="warning",
                        message=(
                            "Repeated instances of the same class were collapsed "
                            "to the single label accepted by BasketEvent."
                        ),
                        context={
                            "jersey_color": identity[0],
                            "jersey_number": identity[1],
                            "label": labels[0],
                            "source_action_indices": [
                                item.source_action_index for item in items
                            ],
                        },
                    )
                )
        return MappingResult(tuple(collapsed), tuple(anomalies))

    def _map_action(
        self, index: int, action: Mapping[str, Any]
    ) -> tuple[list[LabelContribution], list[AnnotationAnomaly]]:
        """Map one BARD action and its optional assist relationship."""
        anomalies: list[AnnotationAnomaly] = []
        color = normalize_color(action.get("color"))
        player = normalize_jersey_number(action.get("player"))
        action_name = " ".join(str(action.get("action", "")).strip().split())
        action_key = action_name.casefold()
        if color is None or player is None:
            anomalies.append(
                AnnotationAnomaly(
                    code="INVALID_ACTION_IDENTITY",
                    severity="error",
                    message="An action actor needs a valid color and jersey number.",
                    context={
                        "source_action_index": index,
                        "color": action.get("color"),
                        "player": action.get("player"),
                    },
                )
            )
            return [], anomalies

        if action_key in self._SHOT_ACTIONS:
            result = action.get("result")
            if not isinstance(result, bool):
                anomalies.append(
                    AnnotationAnomaly(
                        code="MISSING_SHOT_RESULT",
                        severity="error",
                        message="A 2PT/3PT shot needs a boolean result field.",
                        context={"source_action_index": index, "result": result},
                    )
                )
                return [], anomalies
            label = "Made Shot" if result else "Missed Shot"
        else:
            label = self._DIRECT_LABELS.get(action_key)
            if label is None:
                anomalies.append(
                    AnnotationAnomaly(
                        code="UNMAPPED_ACTION",
                        severity="error",
                        message=(
                            "The action is outside the fixed BasketEvent mapping "
                            "and is not converted to blank."
                        ),
                        context={
                            "source_action_index": index,
                            "action": action_name,
                        },
                    )
                )
                return [], anomalies

        if label not in SUPPORTED_LABELS:
            raise RuntimeError(
                f"Fixed BARD mapping produced an unknown BasketEvent label: {label}"
            )

        mapped = [
            LabelContribution(
                jersey_color=color,
                jersey_number=player,
                label=label,
                source_action_index=index,
                role="primary_actor",
            )
        ]
        if action_key in self._SHOT_ACTIONS and action.get("assisted") is True:
            assistant = normalize_jersey_number(action.get("other_player"))
            if assistant is None:
                anomalies.append(
                    AnnotationAnomaly(
                        code="MISSING_ASSISTANT_IDENTITY",
                        severity="error",
                        message=(
                            "An assisted shot needs a valid other_player jersey "
                            "number before an ast label can be generated."
                        ),
                        context={
                            "source_action_index": index,
                            "other_player": action.get("other_player"),
                        },
                    )
                )
            else:
                mapped.append(
                    LabelContribution(
                        jersey_color=color,
                        jersey_number=assistant,
                        label="ast",
                        source_action_index=index,
                        role="assistant",
                    )
                )
        return mapped, anomalies


class BardAnnotationBuilder:
    """Join deterministic BARD labels to Qwen-cleaned trajectories."""

    def __init__(self, mapper: BardLabelMapper | None = None) -> None:
        """Initialize the builder with an optional custom deterministic mapper."""
        self._mapper = mapper or BardLabelMapper()

    def build(
        self,
        clean_tracks: Mapping[str, Any],
        action_document: Mapping[str, Any],
        *,
        bard_game: str,
        game_id: str,
        video_name: str,
    ) -> AnnotationResult:
        """Build one author-compatible annotation and an anomaly report.

        Args:
            clean_tracks: Qwen-cleaned trajectory JSON produced by
                ``recognize.py``.
            action_document: Structured BARD action JSON for the same clip.
            bard_game: BARD game-folder name.
            game_id: Numeric NBA game identifier.
            video_name: Clip stem without ``.mp4``.

        Returns:
            Accepted annotation or ``None`` plus a complete audit report.
        """
        mapping = self._mapper.map_document(action_document)
        anomalies = list(mapping.anomalies)
        track_index: dict[tuple[str, str], list[str]] = {}
        resolved_players: dict[str, dict[str, Any]] = {}

        for track_id, raw_payload in clean_tracks.items():
            if track_id == "ball":
                continue
            if not isinstance(raw_payload, Mapping):
                anomalies.append(
                    AnnotationAnomaly(
                        code="INVALID_TRACK_PAYLOAD",
                        severity="warning",
                        message="Omitted a player track whose payload is not an object.",
                        context={"track_id": str(track_id)},
                    )
                )
                continue
            trajectory = raw_payload.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                anomalies.append(
                    AnnotationAnomaly(
                        code="MISSING_PLAYER_TRAJECTORY",
                        severity="warning",
                        message="Omitted a player track without a non-empty trajectory.",
                        context={"track_id": str(track_id)},
                    )
                )
                continue
            color = normalize_color(raw_payload.get("jersey_color"))
            number = normalize_jersey_number(raw_payload.get("jersey_number"))
            if color is None or number is None:
                anomalies.append(
                    AnnotationAnomaly(
                        code="UNRESOLVED_TRACK_IDENTITY",
                        severity="warning",
                        message=(
                            "Omitted a track that cannot safely be labeled blank "
                            "without a color-and-number identity."
                        ),
                        context={
                            "track_id": str(track_id),
                            "jersey_color": raw_payload.get("jersey_color"),
                            "jersey_number": raw_payload.get("jersey_number"),
                        },
                    )
                )
                continue
            key = (color, number)
            track_index.setdefault(key, []).append(str(track_id))
            resolved_players[str(track_id)] = copy.deepcopy(dict(raw_payload))
            resolved_players[str(track_id)]["jersey_color"] = color
            resolved_players[str(track_id)]["jersey_number"] = number
            resolved_players[str(track_id)].pop("event", None)

        duplicate_identities = {
            key: ids for key, ids in track_index.items() if len(ids) > 1
        }
        for (color, number), track_ids in duplicate_identities.items():
            anomalies.append(
                AnnotationAnomaly(
                    code="AMBIGUOUS_TRACK_IDENTITY",
                    severity="error",
                    message=(
                        "Multiple cleaned tracks share one color-and-number key, "
                        "so an event cannot be assigned uniquely."
                    ),
                    context={
                        "jersey_color": color,
                        "jersey_number": number,
                        "track_ids": track_ids,
                    },
                )
            )

        assignments: list[dict[str, Any]] = []
        for contribution in mapping.contributions:
            candidates = track_index.get(contribution.identity_key, [])
            if not candidates:
                anomalies.append(
                    AnnotationAnomaly(
                        code="UNMATCHED_GT_ACTOR",
                        severity="error",
                        message=(
                            "No cleaned track matches a structured BARD action "
                            "actor by exact jersey color and number."
                        ),
                        context=contribution.to_dict(),
                    )
                )
                continue
            if len(candidates) != 1:
                continue
            track_id = candidates[0]
            resolved_players[track_id]["event"] = {"actionType": contribution.label}
            assignments.append({"track_id": track_id, **contribution.to_dict()})

        ball_payload = clean_tracks.get("ball")
        ball_trajectory = (
            ball_payload.get("trajectory")
            if isinstance(ball_payload, Mapping)
            else None
        )
        if not isinstance(ball_trajectory, list) or not ball_trajectory:
            anomalies.append(
                AnnotationAnomaly(
                    code="MISSING_BALL_TRACK",
                    severity="error",
                    message=(
                        "Training uses require_ball=True, so Scheme A excludes a "
                        "clip without a cleaned ball trajectory."
                    ),
                    context={},
                )
            )

        accepted = not any(item.severity == "error" for item in anomalies)
        annotation: dict[str, Any] | None = None
        if accepted:
            annotation = {
                track_id: resolved_players[track_id]
                for track_id in sorted(resolved_players)
            }
            annotation["ball"] = copy.deepcopy(dict(ball_payload))

        report = {
            "schema_version": "basketevent_annotation_report.v1",
            "status": "accepted" if accepted else "excluded",
            "policy": "scheme_a_single_label_per_player",
            "bard_game": bard_game,
            "game_id": game_id,
            "video_name": video_name,
            "label_source": "bard_structured_action_fixed_rules",
            "mapping_rules_version": "bard_to_basketevent.v1",
            "contributions": [
                contribution.to_dict() for contribution in mapping.contributions
            ],
            "assignments": assignments,
            "retained_player_tracks": (sorted(resolved_players) if accepted else []),
            "anomalies": [item.to_dict() for item in anomalies],
        }
        return AnnotationResult(annotation, report)


def count_anomalies(reports: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count anomaly codes across per-clip reports.

    Args:
        reports: Annotation report documents.

    Returns:
        Stable code-to-count mapping sorted by anomaly code.
    """
    counts: dict[str, int] = {}
    for report in reports:
        for anomaly in report.get("anomalies", []):
            if not isinstance(anomaly, Mapping):
                continue
            code = str(anomaly.get("code", "UNKNOWN_ANOMALY"))
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))
