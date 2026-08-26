"""Convert BARD season rosters into the format used by ``recognize.py``.

BARD stores team identifiers and a season-level player list, while
``recognize.py`` expects each team to have an independently supplied jersey
color. This module performs that structural conversion only. It never infers
colors from BARD action labels because doing so would leak ground-truth event
information into visual player recognition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RosterAnomaly:
    """Describe a roster field that could not be converted safely.

    Attributes:
        code: Stable machine-readable anomaly identifier.
        severity: ``error`` prevents roster use; ``warning`` is auditable but
            does not invalidate the converted document.
        message: Human-readable explanation.
        context: Source values that help a reviewer locate the problem.
    """

    code: str
    severity: str
    message: str
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the anomaly."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


@dataclass(frozen=True)
class RosterConversion:
    """Hold a converted recognition roster and its audit information."""

    roster: dict[str, Any] | None
    anomalies: tuple[RosterAnomaly, ...]

    @property
    def accepted(self) -> bool:
        """Return whether the roster is safe to pass to ``recognize.py``."""
        return self.roster is not None and not any(
            item.severity == "error" for item in self.anomalies
        )


class BardRosterAdapter:
    """Adapt one BARD roster using explicit per-team jersey colors.

    The source roster is season-level metadata, so duplicate team/number pairs
    are retained rather than guessed away. ``recognize.py`` is instructed to
    leave a player name unresolved when multiple names share the same visible
    number and color.
    """

    def convert(
        self,
        source: Mapping[str, Any],
        team_colors: Mapping[str, str],
        game_id: str,
    ) -> RosterConversion:
        """Convert one BARD roster into BasketEvent recognition metadata.

        Args:
            source: Parsed ``description/players/roster.json`` document.
            team_colors: Independently verified mapping from BARD team code to
                the jersey color worn in this game.
            game_id: Numeric NBA game identifier used by BasketEvent.

        Returns:
            Converted roster and all conversion anomalies. A missing team
            color is an error because the visual identity key is color plus
            jersey number.
        """
        anomalies: list[RosterAnomaly] = []
        bard_game = str(source.get("game", "")).strip()
        teams = source.get("teams")
        players = source.get("players")
        if (
            not bard_game
            or not isinstance(teams, list)
            or not isinstance(players, list)
        ):
            anomalies.append(
                RosterAnomaly(
                    code="INVALID_BARD_ROSTER",
                    severity="error",
                    message="BARD roster must contain game, teams, and players fields.",
                    context={"game": bard_game},
                )
            )
            return RosterConversion(None, tuple(anomalies))

        normalized_colors = {
            str(team).strip().upper(): str(color).strip().lower()
            for team, color in team_colors.items()
            if str(team).strip() and str(color).strip()
        }
        missing_colors = [
            str(team).strip().upper()
            for team in teams
            if str(team).strip().upper() not in normalized_colors
        ]
        if missing_colors:
            anomalies.append(
                RosterAnomaly(
                    code="MISSING_TEAM_COLOR",
                    severity="error",
                    message=(
                        "Every team needs an independently verified jersey color; "
                        "colors are never inferred from BARD action labels."
                    ),
                    context={"teams": missing_colors},
                )
            )
            return RosterConversion(None, tuple(anomalies))

        converted_players: list[dict[str, str]] = []
        seen_rows: set[tuple[str, str, str]] = set()
        identity_to_names: dict[tuple[str, str], set[str]] = {}
        for index, player in enumerate(players):
            if not isinstance(player, Mapping):
                anomalies.append(
                    RosterAnomaly(
                        code="INVALID_ROSTER_PLAYER",
                        severity="warning",
                        message="Skipped a roster player that is not an object.",
                        context={"player_index": index},
                    )
                )
                continue
            team = str(player.get("team", "")).strip().upper()
            name = str(player.get("name", "")).strip()
            numbers = player.get("jersey_numbers", [])
            if not team or not name or not isinstance(numbers, list):
                anomalies.append(
                    RosterAnomaly(
                        code="INCOMPLETE_ROSTER_PLAYER",
                        severity="warning",
                        message="Skipped a player without team, name, or jersey list.",
                        context={"player_index": index, "player": dict(player)},
                    )
                )
                continue
            if team not in normalized_colors:
                anomalies.append(
                    RosterAnomaly(
                        code="UNKNOWN_PLAYER_TEAM",
                        severity="warning",
                        message="Skipped a player whose team is not in this game.",
                        context={"player_index": index, "team": team, "name": name},
                    )
                )
                continue
            if not numbers:
                anomalies.append(
                    RosterAnomaly(
                        code="PLAYER_WITHOUT_JERSEY",
                        severity="warning",
                        message="Skipped a season-roster player without a jersey number.",
                        context={"player_index": index, "team": team, "name": name},
                    )
                )
                continue
            for raw_number in numbers:
                number = str(raw_number).strip()
                if not number:
                    continue
                row_key = (team, number, name)
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                converted_players.append(
                    {"team_name": team, "jersey": number, "name": name}
                )
                identity_to_names.setdefault(
                    (normalized_colors[team], number), set()
                ).add(name)

        ambiguous = [
            {"jersey_color": color, "jersey": number, "names": sorted(names)}
            for (color, number), names in sorted(identity_to_names.items())
            if len(names) > 1
        ]
        if ambiguous:
            anomalies.append(
                RosterAnomaly(
                    code="AMBIGUOUS_SEASON_ROSTER_IDENTITY",
                    severity="warning",
                    message=(
                        "Multiple season-roster players share a color and jersey "
                        "number. Visual recognition must leave the name unresolved."
                    ),
                    context={"identities": ambiguous},
                )
            )

        roster = {
            "schema_version": "basketevent_recognition_roster.v1",
            "game_id": game_id,
            "source_game": bard_game,
            "jersey_color": {
                str(team).strip().upper(): normalized_colors[str(team).strip().upper()]
                for team in teams
            },
            "players": sorted(
                converted_players,
                key=lambda item: (
                    item["team_name"],
                    self._jersey_sort_key(item["jersey"]),
                    item["name"],
                ),
            ),
            "source_note": (
                "BARD players.csv is season-level metadata and does not prove "
                "that every listed player appeared in this game."
            ),
        }
        return RosterConversion(roster, tuple(anomalies))

    @staticmethod
    def _jersey_sort_key(number: str) -> tuple[int, int | str]:
        """Return a stable numeric-first sort key while preserving ``00``."""
        return (0, int(number)) if number.isdigit() else (1, number)
