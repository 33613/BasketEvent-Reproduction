"""Create reproducible single-label BasketEvent annotations from BARD.

The command has two independent stages:

``rosters``
    Converts BARD season rosters into the input format consumed by the
    identity resolver. Team colors must come from an independently prepared
    game-level configuration; action labels are never used to infer colors.

``labels``
    Reads Qwen-cleaned trajectories and BARD structured actions, applies the
    fixed mapping in ``src.modules.ingestion.bard.labeling``, filters to
    Scheme A, and writes an
    author-compatible annotation plus a separate anomaly report per clip.

Source BARD files are treated as read-only. Generated files live under the
artifact root configured by ``src.core.config``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import SETTINGS  # noqa: E402
from src.modules.ingestion.bard.labeling import (  # noqa: E402
    BardAnnotationBuilder,
    count_anomalies,
)
from src.modules.ingestion.bard.roster import BardRosterAdapter  # noqa: E402


GAME_ID_PATTERN = re.compile(r"(\d{10})$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional arguments used by tests. ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments for the chosen subcommand.
    """
    parser = argparse.ArgumentParser(
        description="Build fixed-rule BARD annotations for BasketEvent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rosters = subparsers.add_parser(
        "rosters",
        help="Convert BARD rosters using independently verified team colors.",
    )
    _add_common_paths(rosters)
    rosters.add_argument(
        "--team-colors",
        type=Path,
        required=True,
        help=(
            "JSON with a games mapping: {bard_game: {TEAM: color}}. "
            "See config/bard_team_colors.example.json."
        ),
    )
    rosters.add_argument("--dry-run", action="store_true")

    labels = subparsers.add_parser(
        "labels",
        help="Generate Scheme-A labels and per-clip anomaly reports.",
    )
    _add_common_paths(labels)
    labels.add_argument(
        "--clips",
        nargs="+",
        help="Optional clip stems to process within each selected game.",
    )
    labels.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rebuild existing reports. An old accepted annotation is removed "
            "when the rebuilt clip is excluded, preventing stale exports."
        ),
    )
    labels.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first malformed JSON instead of recording and continuing.",
    )
    labels.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    """Add portable BARD and artifact-root options to a subparser.

    Args:
        parser: Subparser receiving shared data-selection arguments.
    """
    parser.add_argument(
        "--data-root",
        type=Path,
        default=SETTINGS.data_root,
        help="BARD staging root; defaults to Settings.data_root.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=SETTINGS.artifacts_root,
        help="Generated artifact root; defaults to Settings.artifacts_root.",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        help="Exact BARD game folders; defaults to every staged game.",
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file.

    Args:
        path: Input JSON path.

    Returns:
        Parsed JSON value.
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    """Atomically write formatted UTF-8 JSON.

    Args:
        path: Destination path.
        value: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def discover_games(data_root: Path, requested: Sequence[str] | None) -> list[str]:
    """Discover staged BARD games and validate an optional selection.

    Args:
        data_root: BARD staging root.
        requested: Exact game folder names, or ``None`` for all games.

    Returns:
        Sorted BARD game folder names.

    Raises:
        FileNotFoundError: If the root or a requested game is unavailable.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"BARD data root not found: {data_root}")
    discovered = {
        path.name
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "video").is_dir()
        and (path / "description").is_dir()
    }
    if requested:
        unknown = sorted(set(requested) - discovered)
        if unknown:
            raise FileNotFoundError(
                f"Requested games are not staged under {data_root}: {unknown}"
            )
        return sorted(dict.fromkeys(requested))
    return sorted(discovered)


def extract_game_id(bard_game: str) -> str:
    """Extract the trailing ten-digit NBA game ID from a BARD folder."""
    match = GAME_ID_PATTERN.search(bard_game)
    if match is None:
        raise ValueError(f"Cannot extract a ten-digit game ID from: {bard_game}")
    return match.group(1)


def load_team_colors(path: Path) -> dict[str, dict[str, str]]:
    """Load and validate independently supplied game-level team colors.

    Args:
        path: JSON configuration path.

    Returns:
        BARD-game to team-color mapping. A top-level ``games`` wrapper is
        accepted so the file can carry schema metadata.
    """
    document = read_json(path)
    if not isinstance(document, Mapping):
        raise ValueError("Team-color configuration must be a JSON object")
    games = document.get("games", document)
    if not isinstance(games, Mapping):
        raise ValueError("team-colors.games must be a JSON object")
    result: dict[str, dict[str, str]] = {}
    for game, mapping in games.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Team colors for {game!r} must be an object")
        result[str(game)] = {
            str(team).strip().upper(): str(color).strip().lower()
            for team, color in mapping.items()
        }
    return result


def run_rosters(args: argparse.Namespace) -> dict[str, Any]:
    """Convert selected BARD rosters and write game-level audit reports.

    Args:
        args: Parsed ``rosters`` arguments.

    Returns:
        Batch conversion summary.
    """
    data_root = args.data_root.expanduser().resolve()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    colors_by_game = load_team_colors(args.team_colors.expanduser().resolve())
    games = discover_games(data_root, args.games)
    adapter = BardRosterAdapter()
    results: list[dict[str, Any]] = []

    for game in games:
        source_path = data_root / game / "description" / "players" / "roster.json"
        output_dir = artifacts_root / game / "metadata"
        output_path = output_dir / "recognize_roster.json"
        report_path = output_dir / "roster_report.json"
        if not source_path.is_file():
            report = {
                "schema_version": "basketevent_roster_report.v1",
                "status": "excluded",
                "bard_game": game,
                "anomalies": [
                    {
                        "code": "MISSING_BARD_ROSTER",
                        "severity": "error",
                        "message": "BARD roster.json was not found.",
                        "context": {"path": str(source_path)},
                    }
                ],
            }
        else:
            source = read_json(source_path)
            if not isinstance(source, Mapping):
                raise ValueError(f"Roster root must be an object: {source_path}")
            conversion = adapter.convert(
                source,
                colors_by_game.get(game, {}),
                extract_game_id(game),
            )
            report = {
                "schema_version": "basketevent_roster_report.v1",
                "status": "accepted" if conversion.accepted else "excluded",
                "bard_game": game,
                "source": str(source_path),
                "output": str(output_path) if conversion.accepted else None,
                "anomalies": [item.to_dict() for item in conversion.anomalies],
            }
            if conversion.accepted and not args.dry_run:
                write_json(output_path, conversion.roster)
        if not args.dry_run:
            write_json(report_path, report)
        results.append(report)

    summary = {
        "schema_version": "basketevent_roster_build.v1",
        "data_root": str(data_root),
        "artifacts_root": str(artifacts_root),
        "games": len(results),
        "accepted": sum(item["status"] == "accepted" for item in results),
        "excluded": sum(item["status"] == "excluded" for item in results),
        "results": results,
    }
    if not args.dry_run:
        write_json(artifacts_root / "roster_build_summary.json", summary)
    return summary


def missing_input_report(
    *,
    game: str,
    game_id: str,
    video_name: str,
    code: str,
    message: str,
    path: Path,
) -> dict[str, Any]:
    """Build a standard excluded report for a missing clip input."""
    return {
        "schema_version": "basketevent_annotation_report.v1",
        "status": "excluded",
        "policy": "scheme_a_single_label_per_player",
        "bard_game": game,
        "game_id": game_id,
        "video_name": video_name,
        "label_source": "bard_structured_action_fixed_rules",
        "mapping_rules_version": "bard_to_basketevent.v1",
        "contributions": [],
        "assignments": [],
        "retained_player_tracks": [],
        "anomalies": [
            {
                "code": code,
                "severity": "error",
                "message": message,
                "context": {"path": str(path)},
            }
        ],
    }


def run_labels(args: argparse.Namespace) -> dict[str, Any]:
    """Build labels for selected clips while retaining every anomaly report.

    Args:
        args: Parsed ``labels`` arguments.

    Returns:
        Batch annotation summary.
    """
    data_root = args.data_root.expanduser().resolve()
    artifacts_root = args.artifacts_root.expanduser().resolve()
    games = discover_games(data_root, args.games)
    selected_clips = set(args.clips or [])
    builder = BardAnnotationBuilder()
    reports: list[dict[str, Any]] = []
    skipped_existing = 0

    for game in games:
        game_id = extract_game_id(game)
        game_root = data_root / game
        clean_dir = artifacts_root / game / "tracks" / "clean"
        annotation_dir = artifacts_root / game / "annotations"
        report_dir = artifacts_root / game / "reports"
        videos = sorted((game_root / "video").glob("*.mp4"))
        if selected_clips:
            videos = [video for video in videos if video.stem in selected_clips]

        for video in videos:
            video_name = video.stem
            action_path = game_root / "description" / "action" / f"{video_name}.json"
            clean_path = clean_dir / f"{video_name}.json"
            annotation_path = annotation_dir / f"{video_name}.json"
            report_path = report_dir / f"{video_name}.json"
            if report_path.exists() and not args.overwrite:
                existing_report = read_json(report_path)
                already_complete = (
                    isinstance(existing_report, Mapping)
                    and existing_report.get("status") == "accepted"
                    and annotation_path.is_file()
                )
                if already_complete:
                    skipped_existing += 1
                    continue
                # Excluded reports are retried automatically. This lets a
                # MISSING_CLEAN_TRACK clip become eligible after SAM3/Qwen
                # finishes without requiring a destructive overwrite flag.

            try:
                if not action_path.is_file():
                    report = missing_input_report(
                        game=game,
                        game_id=game_id,
                        video_name=video_name,
                        code="MISSING_ACTION_DOCUMENT",
                        message="Structured BARD action JSON was not found.",
                        path=action_path,
                    )
                    annotation = None
                elif not clean_path.is_file():
                    report = missing_input_report(
                        game=game,
                        game_id=game_id,
                        video_name=video_name,
                        code="MISSING_CLEAN_TRACK",
                        message=(
                            "Run tracking and identity resolution before generating labels "
                            "for this clip."
                        ),
                        path=clean_path,
                    )
                    annotation = None
                else:
                    action_document = read_json(action_path)
                    clean_tracks = read_json(clean_path)
                    if not isinstance(action_document, Mapping) or not isinstance(
                        clean_tracks, Mapping
                    ):
                        raise ValueError("Action and clean-track roots must be objects")
                    result = builder.build(
                        clean_tracks,
                        action_document,
                        bard_game=game,
                        game_id=game_id,
                        video_name=video_name,
                    )
                    annotation = result.annotation
                    report = result.report
                    report["inputs"] = {
                        "video": str(video),
                        "action": str(action_path),
                        "clean_tracks": str(clean_path),
                    }
                    report["output"] = (
                        str(annotation_path) if annotation is not None else None
                    )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                if args.fail_fast:
                    raise
                report = missing_input_report(
                    game=game,
                    game_id=game_id,
                    video_name=video_name,
                    code="PROCESSING_ERROR",
                    message=f"Could not process clip: {type(error).__name__}: {error}",
                    path=clean_path,
                )
                annotation = None

            if not args.dry_run:
                if annotation is not None:
                    write_json(annotation_path, annotation)
                elif args.overwrite and annotation_path.exists():
                    # Overwrite means the current deterministic result is
                    # authoritative. Removing a now-rejected stale annotation
                    # prevents the runtime exporter from silently using it.
                    annotation_path.unlink()
                write_json(report_path, report)
            reports.append(report)

    summary = {
        "schema_version": "basketevent_annotation_build.v1",
        "policy": "scheme_a_single_label_per_player",
        "label_source": "bard_structured_action_fixed_rules",
        "data_root": str(data_root),
        "artifacts_root": str(artifacts_root),
        "processed": len(reports),
        "accepted": sum(item["status"] == "accepted" for item in reports),
        "excluded": sum(item["status"] == "excluded" for item in reports),
        "skipped_existing_reports": skipped_existing,
        "anomaly_counts": count_anomalies(reports),
        "games": games,
    }
    if not args.dry_run:
        write_json(artifacts_root / "annotation_build_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected deterministic BARD annotation stage."""
    args = parse_args(argv)
    if args.command == "rosters":
        summary = run_rosters(args)
    elif args.command == "labels":
        summary = run_labels(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
