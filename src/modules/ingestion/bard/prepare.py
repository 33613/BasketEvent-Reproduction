"""Prepare a small BARD subset and export it for BasketEvent.

The script keeps two data layouts separate:

1. ``prepare`` builds a human-readable BARD staging directory.  It preserves
   the source video, play-by-play (PBP), structured actions, GT caption, and
   roster information.  It does not invent BasketEvent labels.
2. ``repair-manifest`` rebuilds the lightweight clip index after folders are
   renamed or moved.  It scans file names only and never decodes video data.
3. ``export`` converts accepted annotations from
   ``<artifacts>/<game>/annotations`` to the directory layout read by
   ``src.modules.event_recognition.dataset``. Source BARD folders remain
   read-only.

The ``make-split`` command assigns whole games to train/valid/test.  Splitting
by game prevents clips from the same game from leaking across data splits.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import SETTINGS  # noqa: E402


HF_REPO = "GabrieleGiudici/BARD"
DEFAULT_WORKSPACE_ROOT = SETTINGS.data_root
METADATA_FILES = (
    "dataset.csv",
    "dataset_paths.csv",
    "players.csv",
    "captions/caption.json",
)
VALID_SPLITS = ("train", "valid", "test")

# BARD game-folder abbreviations and players.csv do not always use the same
# NBA abbreviation.  Keeping these exceptions explicit avoids fuzzy matching.
TEAM_ALIASES = {
    "bkn": "BRK",
    "cha": "CHO",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence.  ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments for the selected subcommand.
    """
    parser = argparse.ArgumentParser(
        description="Prepare BARD data and export it for BasketEvent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Download selected BARD files and build staging games."
    )
    prepare.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    prepare.add_argument(
        "--source-dir",
        type=Path,
        help="Raw Hugging Face mirror; defaults to <workspace>/_bard_source.",
    )
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument(
        "--games", nargs="+", help="Exact BARD game folders to prepare."
    )
    selection.add_argument(
        "--all-games",
        action="store_true",
        help="Prepare every game. Use only after checking disk space.",
    )
    prepare.add_argument(
        "--game-limit",
        type=int,
        default=2,
        help="Number of metadata-sorted games when --games is omitted (default: 2).",
    )
    prepare.add_argument(
        "--download",
        action="store_true",
        help="Download/resume metadata and exactly the selected final MP4 files.",
    )
    prepare.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Download/load metadata and print the selection without touching "
            "MP4 files."
        ),
    )
    prepare.add_argument("--max-workers", type=int, default=2)
    prepare.add_argument(
        "--download-retries",
        type=int,
        default=5,
        help="Retry the filtered snapshot after transient download failures.",
    )
    prepare.add_argument(
        "--retry-delay",
        type=int,
        default=15,
        help="Base seconds between retries; later retries use a capped backoff.",
    )
    prepare.add_argument(
        "--materialize",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hard links avoid a second local copy when both paths share a drive.",
    )
    prepare.add_argument(
        "--allow-partial",
        action="store_true",
        help="Prepare available clips even when a selected game is incomplete.",
    )
    prepare.add_argument("--dry-run", action="store_true")

    repair_manifest = subparsers.add_parser(
        "repair-manifest",
        help="Rebuild manifest.jsonl files from an existing staging layout.",
    )
    repair_manifest.add_argument(
        "--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT
    )
    repair_manifest.add_argument(
        "--games",
        nargs="+",
        help="Exact game folders; defaults to every discovered staging game.",
    )
    repair_manifest.add_argument("--dry-run", action="store_true")

    make_split = subparsers.add_parser(
        "make-split", help="Create a deterministic game-level split file."
    )
    make_split.add_argument(
        "--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT
    )
    make_split.add_argument(
        "--output",
        type=Path,
        default=SETTINGS.split_config,
        help="Output JSON; defaults to Settings.split_config.",
    )
    make_split.add_argument(
        "--annotations-root",
        type=Path,
        default=SETTINGS.artifacts_root,
        help=(
            "Artifact root containing <game>/annotations/*.json. Only games "
            "with accepted Scheme-A annotations are split."
        ),
    )
    make_split.add_argument("--train-ratio", type=float, default=0.8)
    make_split.add_argument("--valid-ratio", type=float, default=0.1)
    make_split.add_argument("--seed", type=int, default=42)
    make_split.add_argument("--dry-run", action="store_true")

    export = subparsers.add_parser(
        "export",
        help="Export completed games for the event-recognition dataset loader.",
    )
    export.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    export.add_argument(
        "--runtime-root",
        type=Path,
        default=SETTINGS.runtime_root,
        help="Destination containing videos/, train/, valid/, and test/.",
    )
    export.add_argument(
        "--annotations-root",
        type=Path,
        default=SETTINGS.artifacts_root,
        help="Artifact root containing accepted <game>/annotations/*.json.",
    )
    export.add_argument(
        "--split-file",
        type=Path,
        default=SETTINGS.split_config,
        help="Game-level split JSON; defaults to Settings.split_config.",
    )
    export.add_argument(
        "--materialize", choices=("hardlink", "copy"), default="hardlink"
    )
    export.add_argument(
        "--allow-missing-annotations",
        action="store_true",
        help=(
            "Skip and report clips excluded by Scheme A. Without this flag, "
            "the first missing accepted annotation stops export."
        ),
    )
    export.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def read_semicolon_csv(path: Path) -> list[dict[str, str]]:
    """Read a semicolon-delimited UTF-8 CSV file.

    Args:
        path: CSV path.

    Returns:
        Rows represented as dictionaries.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def normalize_video_key(value: str) -> str:
    """Normalize a BARD repository-relative video path.

    Args:
        value: Path read from BARD metadata.

    Returns:
        A forward-slash repository-relative path.
    """
    return Path(value.replace("\\", "/")).as_posix()


def parse_actions(raw_actions: str) -> list[dict[str, Any]]:
    """Safely parse BARD's Python-literal action list.

    Args:
        raw_actions: Serialized list stored in a BARD CSV row.

    Returns:
        Parsed action dictionaries.

    Raises:
        ValueError: If the value is not a list of dictionaries.
    """
    actions = ast.literal_eval(raw_actions)
    if not isinstance(actions, list) or not all(
        isinstance(item, dict) for item in actions
    ):
        raise ValueError("BARD actions must be a list of dictionaries")
    return actions


def load_metadata(
    source_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], list[dict[str, str]]]:
    """Load BARD paths, actions, captions, and season roster metadata.

    Args:
        source_dir: Local mirror root containing the four metadata files.

    Returns:
        A tuple of clip records, video paths grouped by game, and player rows.

    Raises:
        ValueError: If paired BARD CSV files are inconsistent.
        KeyError: If a required column is missing.
    """
    source_rows = read_semicolon_csv(source_dir / "dataset.csv")
    path_rows = read_semicolon_csv(source_dir / "dataset_paths.csv")
    if len(source_rows) != len(path_rows):
        raise ValueError("dataset.csv and dataset_paths.csv row counts differ")

    records: dict[str, dict[str, Any]] = {}
    videos_by_game: dict[str, list[str]] = defaultdict(list)
    for index, (source_row, path_row) in enumerate(
        zip(source_rows, path_rows, strict=True)
    ):
        if source_row["actions"] != path_row["actions"]:
            raise ValueError(f"Action mismatch at metadata row {index}")
        video_key = normalize_video_key(path_row["urls"])
        game = video_key.split("/", maxsplit=1)[0]
        records[video_key] = {
            "nba_event_url": source_row["urls"],
            "actions": parse_actions(path_row["actions"]),
            "numerosity": int(path_row["numerosity"]),
        }
        videos_by_game[game].append(video_key)

    caption_path = source_dir / "captions" / "caption.json"
    with caption_path.open("r", encoding="utf-8") as handle:
        caption_rows = json.load(handle)
    for item in caption_rows:
        video_key = normalize_video_key(item["video"])
        assistant_answers = [
            turn.get("value", "").strip()
            for turn in item.get("conversations", [])
            if turn.get("from") in {"gpt", "assistant"}
        ]
        if video_key in records and assistant_answers:
            records[video_key]["gt_caption"] = assistant_answers[-1]

    players = read_semicolon_csv(source_dir / "players.csv")
    return records, dict(videos_by_game), players


def download_metadata(source_dir: Path, max_workers: int) -> None:
    """Download or resume only the BARD metadata files.

    Args:
        source_dir: Local Hugging Face mirror destination.
        max_workers: Maximum concurrent download workers.

    Raises:
        RuntimeError: If ``huggingface_hub`` is unavailable.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before using --download.") from exc
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=list(METADATA_FILES),
        local_dir=source_dir,
        max_workers=max_workers,
    )


def select_games(
    videos_by_game: dict[str, list[str]],
    requested_games: list[str] | None,
    all_games: bool,
    game_limit: int,
) -> list[str]:
    """Select games explicitly or by a conservative default limit.

    Args:
        videos_by_game: Metadata paths grouped by BARD game folder.
        requested_games: Explicit folder names, if supplied.
        all_games: Whether every game should be selected.
        game_limit: Default count when no explicit selection is supplied.

    Returns:
        Selected game folders in stable order.

    Raises:
        KeyError: If an explicitly requested game is unknown.
        ValueError: If ``game_limit`` is not positive.
    """
    if requested_games:
        unknown = sorted(set(requested_games) - videos_by_game.keys())
        if unknown:
            raise KeyError(f"Unknown BARD game folders: {unknown}")
        return list(dict.fromkeys(requested_games))
    games = sorted(videos_by_game)
    if all_games:
        return games
    if game_limit <= 0:
        raise ValueError("--game-limit must be positive")
    return games[:game_limit]


def download_selected_videos(
    selected_games: list[str],
    videos_by_game: dict[str, list[str]],
    source_dir: Path,
    workspace_root: Path,
    max_workers: int,
    download_retries: int,
    retry_delay: int,
) -> None:
    """Download or resume the final MP4 files for selected games only.

    Args:
        selected_games: BARD game folders to download.
        videos_by_game: Repository-relative video paths grouped by game.
        source_dir: Local Hugging Face mirror destination.
        workspace_root: Staging root used to skip clips already prepared.
        max_workers: Maximum concurrent download workers.
        download_retries: Number of retries after the initial attempt.
        retry_delay: Base seconds used for capped linear backoff.

    Raises:
        RuntimeError: If ``huggingface_hub`` is unavailable.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before using --download.") from exc
    allow_patterns = [
        video_key
        for game in selected_games
        for video_key in videos_by_game[game]
        if locate_available_video(source_dir, workspace_root, video_key) is None
    ]
    if not allow_patterns:
        print("All selected videos already exist locally; skipping video download.")
        return
    print(
        f"Downloading/resuming {len(allow_patterns)} final videos "
        f"from {len(selected_games)} games..."
    )
    if download_retries < 0:
        raise ValueError("--download-retries cannot be negative")
    if retry_delay < 0:
        raise ValueError("--retry-delay cannot be negative")

    total_attempts = download_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                allow_patterns=allow_patterns,
                local_dir=source_dir,
                max_workers=max_workers,
            )
            return
        except Exception as exc:
            # KeyboardInterrupt and SystemExit inherit from BaseException and
            # remain immediately interruptible by the user.
            if attempt == total_attempts:
                raise
            delay = min(retry_delay * attempt, 60)
            print(
                f"Download attempt {attempt}/{total_attempts} failed with "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"Retrying the resumable snapshot in {delay} seconds...")
            time.sleep(delay)


def parse_nba_event_url(url: str) -> dict[str, str | None]:
    """Extract NBA event metadata from a BARD source URL.

    Args:
        url: NBA event URL stored by BARD.

    Returns:
        Game, event, season, and play-description fields when present.
    """
    query = parse_qs(urlparse(url).query)

    def query_value(name: str) -> str | None:
        """Return the first non-empty URL query value.

        Args:
            name: Query parameter name.

        Returns:
            The first stripped value, or ``None``.
        """
        values = query.get(name)
        return values[0].strip() if values and values[0].strip() else None

    return {
        "game_id": query_value("GameID"),
        "game_event_id": query_value("GameEventID"),
        "season": query_value("Season"),
        "description": query_value("title"),
    }


def game_team_codes(game: str) -> list[str]:
    """Convert a BARD game folder into its two roster team codes.

    Args:
        game: Folder such as ``bkn-vs-det-0022400861``.

    Returns:
        Two team abbreviations used by ``players.csv``.

    Raises:
        ValueError: If the folder does not contain ``-vs-``.
    """
    matchup = game.rsplit("-", maxsplit=1)[0]
    left, separator, right = matchup.partition("-vs-")
    if not separator:
        raise ValueError(f"Cannot parse teams from game folder: {game}")
    return [TEAM_ALIASES.get(code, code.upper()) for code in (left, right)]


def build_roster(game: str, players: list[dict[str, str]]) -> dict[str, Any]:
    """Build auditable season-roster metadata for one BARD game.

    Args:
        game: BARD game folder.
        players: Rows loaded from ``players.csv``.

    Returns:
        JSON-serializable roster metadata.
    """
    team_codes = game_team_codes(game)
    roster = []
    for player in players:
        if player.get("Team") not in team_codes:
            continue
        jersey_numbers = [
            value for value in (player.get("Number"), player.get("Number2")) if value
        ]
        roster.append(
            {
                "team": player.get("Team"),
                "name": player.get("Player Name"),
                "short_name": player.get("Player Name 2"),
                "jersey_numbers": jersey_numbers,
            }
        )
    return {
        "schema_version": "bard_game_roster.v1",
        "game": game,
        "teams": team_codes,
        "note": (
            "BARD players.csv is season-level roster metadata; it does not prove "
            "that every listed player appeared in this game."
        ),
        "players": roster,
    }


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, readable UTF-8 JSON.

    Args:
        path: Destination path.
        value: JSON-serializable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one UTF-8 JSON object per line.

    Args:
        path: Destination JSONL path.
        rows: JSON-serializable dictionaries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    """Materialize a file by hard link or copy without overwriting differences.

    Args:
        source: Existing source file.
        destination: Desired destination file.
        mode: Either ``hardlink`` or ``copy``.

    Returns:
        A short outcome string used in summaries.

    Raises:
        FileExistsError: If an existing destination has a different size.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Destination differs from source: {destination}")
        return "existing"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlinked"
        except OSError:
            # Different disks and some network filesystems do not support links.
            shutil.copy2(source, destination)
            return "copied_after_hardlink_failure"
    shutil.copy2(source, destination)
    return "copied"


def ensure_staging_layout(game_root: Path) -> dict[str, Path]:
    """Create the user's current human-readable per-game directory layout.

    Args:
        game_root: Staging directory for one BARD game.

    Returns:
        Named paths for source videos, descriptions, and roster metadata.
    """
    paths = {
        "video": game_root / "video",
        "pbp": game_root / "description" / "PBP",
        "action": game_root / "description" / "action",
        "caption": game_root / "description" / "GT caption",
        "players": game_root / "description" / "players",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def locate_available_video(
    source_dir: Path, workspace_root: Path, video_key: str
) -> Path | None:
    """Find a clip in the raw mirror or an already prepared game.

    Args:
        source_dir: Raw Hugging Face mirror root.
        workspace_root: BARD staging root.
        video_key: Repository-relative path such as ``game/2.mp4``.

    Returns:
        Existing clip path, or ``None`` when unavailable.
    """
    repository_path = Path(video_key)
    raw_video = source_dir / repository_path
    if raw_video.is_file():
        return raw_video
    staged_video = (
        workspace_root / repository_path.parent / "video" / repository_path.name
    )
    return staged_video if staged_video.is_file() else None


def existing_relative_path(game_root: Path, candidate: Path) -> str | None:
    """Return a POSIX-style relative path only when the file exists.

    Args:
        game_root: Per-game staging directory.
        candidate: Expected file inside ``game_root``.

    Returns:
        Relative path for an existing file, otherwise ``None``.
    """
    if not candidate.is_file():
        return None
    return candidate.relative_to(game_root).as_posix()


def build_manifest_row(game_root: Path, video: Path) -> dict[str, Any]:
    """Build one auditable BARD clip-index record.

    Args:
        game_root: Per-game staging directory.
        video: Existing staged MP4 file.

    Returns:
        Manifest row linking the clip to all known metadata and annotation files.
    """
    stem = video.stem
    return {
        "schema_version": "bard_clip_manifest.v2",
        "bard_game": game_root.name,
        "game_id": extract_game_id(game_root.name),
        "video_name": stem,
        "video": existing_relative_path(game_root, video),
        "pbp": existing_relative_path(
            game_root, game_root / "description" / "PBP" / f"{stem}.json"
        ),
        "action": existing_relative_path(
            game_root, game_root / "description" / "action" / f"{stem}.json"
        ),
        "gt_caption": existing_relative_path(
            game_root,
            game_root / "description" / "GT caption" / f"{stem}.txt",
        ),
        "roster": existing_relative_path(
            game_root, game_root / "description" / "players" / "roster.json"
        ),
        "basket_event_annotation": existing_relative_path(
            game_root, game_root / "label" / f"{stem}.json"
        ),
    }


def prepare_game(
    game: str,
    expected_keys: list[str],
    records: dict[str, dict[str, Any]],
    players: list[dict[str, str]],
    source_dir: Path,
    workspace_root: Path,
    materialize: str,
    allow_partial: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Prepare one BARD game without creating model-derived annotations.

    Args:
        game: BARD game folder.
        expected_keys: Repository-relative video paths for the game.
        records: Parsed BARD metadata indexed by video path.
        players: Season roster rows.
        source_dir: Raw Hugging Face mirror root.
        workspace_root: Staging root.
        materialize: File materialization mode.
        allow_partial: Whether missing clips are permitted.
        dry_run: Whether to report without writing.

    Returns:
        Per-game processing summary.

    Raises:
        FileNotFoundError: If clips are missing and partial data is disallowed.
        KeyError: If a clip has no BARD GT caption.
    """
    available = []
    missing = []
    for video_key in expected_keys:
        source_video = locate_available_video(source_dir, workspace_root, video_key)
        if source_video is None:
            missing.append(video_key)
        else:
            available.append((video_key, source_video))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"{game}: missing {len(missing)} of {len(expected_keys)} clips. "
            "Use --download or explicitly accept --allow-partial."
        )
    if dry_run:
        return {
            "game": game,
            "expected_videos": len(expected_keys),
            "available_videos": len(available),
            "missing_videos": len(missing),
            "status": "dry_run",
        }

    game_root = workspace_root / game
    paths = ensure_staging_layout(game_root)
    write_json(paths["players"] / "roster.json", build_roster(game, players))
    outcomes: dict[str, int] = defaultdict(int)
    manifest_rows = []
    for video_key, source_video in available:
        record = records[video_key]
        caption = record.get("gt_caption")
        if not caption:
            raise KeyError(f"Missing GT caption for {video_key}")
        video_name = Path(video_key).name
        stem = Path(video_name).stem
        destination_video = paths["video"] / video_name
        if source_video.resolve() == destination_video.resolve():
            outcomes["existing"] += 1
        else:
            outcomes[
                materialize_file(source_video, destination_video, materialize)
            ] += 1

        pbp = {
            "schema_version": "bard_pbp.v1",
            "video": video_name,
            "nba_event_url": record["nba_event_url"],
            **parse_nba_event_url(record["nba_event_url"]),
        }
        action = {
            "schema_version": "bard_actions.v1",
            "video": video_name,
            "numerosity": record["numerosity"],
            "actions": record["actions"],
            "note": "BARD structured GT; this is not yet a BasketEvent label.",
        }
        write_json(paths["pbp"] / f"{stem}.json", pbp)
        write_json(paths["action"] / f"{stem}.json", action)
        (paths["caption"] / f"{stem}.txt").write_text(
            str(caption).rstrip() + "\n", encoding="utf-8"
        )

        manifest_rows.append(build_manifest_row(game_root, destination_video))

    manifest_path = game_root / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    return {
        "game": game,
        "expected_videos": len(expected_keys),
        "processed_videos": len(available),
        "missing_videos": len(missing),
        "materialize": dict(outcomes),
        "annotation_status": "not_generated_by_this_script",
    }


def discover_staging_games(workspace_root: Path) -> list[str]:
    """Discover prepared BARD game folders without scanning video contents.

    Args:
        workspace_root: BARD staging root.

    Returns:
        Sorted folder names containing both ``video`` and ``description``.
    """
    if not workspace_root.is_dir():
        return []
    return sorted(
        path.name
        for path in workspace_root.iterdir()
        if path.is_dir()
        and (path / "video").is_dir()
        and (path / "description").is_dir()
    )


def discover_annotated_games(games: Sequence[str], annotations_root: Path) -> list[str]:
    """Keep only games with at least one accepted Scheme-A annotation.

    Args:
        games: Staged BARD game folders.
        annotations_root: Artifact root populated by
            ``build_bard_annotations.py labels``.

    Returns:
        Sorted game names with at least one ``annotations/*.json`` file.
    """
    return sorted(
        game
        for game in games
        if any((annotations_root / game / "annotations").glob("*.json"))
    )


def repair_game_manifest(game_root: Path, dry_run: bool) -> dict[str, Any]:
    """Rebuild one game's manifest without reading video contents.

    Args:
        game_root: Existing per-game staging directory.
        dry_run: Whether to report without replacing ``manifest.jsonl``.

    Returns:
        Counts of indexed clips and missing related files.

    Raises:
        FileNotFoundError: If the game has no staged MP4 files.
    """
    videos = sorted((game_root / "video").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No staged MP4 files found in: {game_root / 'video'}")
    rows = [build_manifest_row(game_root, video) for video in videos]
    related_fields = (
        "pbp",
        "action",
        "gt_caption",
        "roster",
        "basket_event_annotation",
    )
    missing = {
        field: sum(row[field] is None for row in rows) for field in related_fields
    }
    if not dry_run:
        write_jsonl(game_root / "manifest.jsonl", rows)
    return {
        "game": game_root.name,
        "clips": len(rows),
        "missing": missing,
        "status": "dry_run" if dry_run else "repaired",
    }


def calculate_split_counts(
    game_count: int, train_ratio: float, valid_ratio: float
) -> tuple[int, int, int]:
    """Calculate practical game counts for train, valid, and test.

    Args:
        game_count: Number of complete games.
        train_ratio: Desired training fraction.
        valid_ratio: Desired validation fraction.

    Returns:
        Counts in train, valid, test order.

    Raises:
        ValueError: If ratios are invalid or no games are available.
    """
    if game_count <= 0:
        raise ValueError("No staged games were found")
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, valid >= 0, train+valid < 1")
    if game_count == 1:
        return 1, 0, 0
    if game_count == 2:
        return 1, 0, 1

    valid_count = max(1, round(game_count * valid_ratio))
    test_count = max(1, round(game_count * (1 - train_ratio - valid_ratio)))
    train_count = game_count - valid_count - test_count
    if train_count < 1:
        train_count = 1
        if valid_count >= test_count and valid_count > 1:
            valid_count -= 1
        elif test_count > 1:
            test_count -= 1
    return train_count, valid_count, test_count


def build_split_config(
    games: list[str], train_ratio: float, valid_ratio: float, seed: int
) -> dict[str, Any]:
    """Build a deterministic game-level split configuration.

    Args:
        games: Prepared BARD game folder names.
        train_ratio: Desired training fraction.
        valid_ratio: Desired validation fraction.
        seed: Random seed used before slicing the game list.

    Returns:
        JSON-serializable split configuration.
    """
    shuffled = sorted(games)
    random.Random(seed).shuffle(shuffled)
    train_count, valid_count, _ = calculate_split_counts(
        len(shuffled), train_ratio, valid_ratio
    )
    train_end = train_count
    valid_end = train_end + valid_count
    return {
        "schema_version": "bard_game_split.v1",
        "split_unit": "game",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": round(1 - train_ratio - valid_ratio, 10),
        },
        "splits": {
            "train": shuffled[:train_end],
            "valid": shuffled[train_end:valid_end],
            "test": shuffled[valid_end:],
        },
        "warning": (
            "A two-game pilot produces train=1, valid=0, test=1 and is suitable "
            "only for pipeline smoke testing, not a meaningful experiment."
            if len(shuffled) == 2
            else None
        ),
    }


def load_split_config(path: Path) -> dict[str, list[str]]:
    """Load and validate a game-level split configuration.

    Args:
        path: JSON file created by ``make-split`` or an equivalent manual file.

    Returns:
        Mapping from split name to game folder names.

    Raises:
        ValueError: If splits are absent, malformed, or overlap.
    """
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    splits = document.get("splits", document)
    if not all(isinstance(splits.get(name), list) for name in VALID_SPLITS):
        raise ValueError("Split file must contain train, valid, and test lists")
    normalized = {name: list(splits[name]) for name in VALID_SPLITS}
    flattened = [game for name in VALID_SPLITS for game in normalized[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A game appears in more than one split")
    return normalized


def extract_game_id(game: str) -> str:
    """Extract the ten-digit NBA game ID from a BARD game folder.

    Args:
        game: Folder such as ``bkn-vs-det-0022400861``.

    Returns:
        Numeric NBA game ID.

    Raises:
        ValueError: If the folder does not end in a ten-digit ID.
    """
    match = re.search(r"(\d{10})$", game)
    if match is None:
        raise ValueError(f"Cannot extract a ten-digit game ID from: {game}")
    return match.group(1)


def export_runtime(
    workspace_root: Path,
    annotations_root: Path,
    runtime_root: Path,
    splits: dict[str, list[str]],
    materialize: str,
    allow_missing_annotations: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Export staged games into the exact layout consumed by BasketEvent.

    Args:
        workspace_root: Human-readable BARD staging root.
        annotations_root: Artifact root containing accepted annotations.
        runtime_root: Destination containing videos/train/valid/test.
        splits: Game-level split assignment.
        materialize: File materialization mode.
        allow_missing_annotations: Whether missing final JSON is permitted.
        dry_run: Whether to validate and report without writing.

    Returns:
        Export summary with clip and missing-annotation counts.

    Raises:
        FileNotFoundError: If a staged game, video directory, or required
            annotation is missing.
    """
    summary: dict[str, Any] = {
        "schema_version": "basketevent_runtime_export.v1",
        "workspace_root": str(workspace_root.resolve()),
        "annotations_root": str(annotations_root.resolve()),
        "runtime_root": str(runtime_root.resolve()),
        "dry_run": dry_run,
        "splits": {},
    }
    manifest_rows = []
    seen_game_ids: dict[str, str] = {}
    for split in VALID_SPLITS:
        split_clips = 0
        split_missing = 0
        for game in splits[split]:
            game_root = workspace_root / game
            video_dir = game_root / "video"
            if not video_dir.is_dir():
                raise FileNotFoundError(f"Missing staged video directory: {video_dir}")
            game_id = extract_game_id(game)
            previous = seen_game_ids.setdefault(game_id, game)
            if previous != game:
                raise ValueError(
                    f"Game ID {game_id} is shared by {previous!r} and {game!r}"
                )
            videos = sorted(video_dir.glob("*.mp4"))
            if not videos:
                raise FileNotFoundError(f"No MP4 files found in: {video_dir}")
            for video in videos:
                annotation = (
                    annotations_root / game / "annotations" / f"{video.stem}.json"
                )
                destination_video = runtime_root / "videos" / game_id / video.name
                destination_annotation = (
                    runtime_root / split / game_id / f"{video.stem}.json"
                )
                if not annotation.is_file():
                    split_missing += 1
                    if not allow_missing_annotations:
                        raise FileNotFoundError(
                            f"Missing final BasketEvent annotation: {annotation}"
                        )
                    manifest_rows.append(
                        {
                            "status": "skipped_missing_annotation",
                            "split": split,
                            "bard_game": game,
                            "game_id": game_id,
                            "video_name": video.stem,
                            "source_video": str(video),
                            "source_annotation": None,
                            "video": None,
                            "annotation": None,
                        }
                    )
                    split_clips += 1
                    continue
                if not dry_run:
                    materialize_file(video, destination_video, materialize)
                    materialize_file(annotation, destination_annotation, materialize)
                manifest_rows.append(
                    {
                        "status": "exported",
                        "split": split,
                        "bard_game": game,
                        "game_id": game_id,
                        "video_name": video.stem,
                        "source_video": str(video),
                        "source_annotation": str(annotation),
                        "video": str(destination_video),
                        "annotation": str(destination_annotation),
                    }
                )
                split_clips += 1
        summary["splits"][split] = {
            "games": len(splits[split]),
            "clips": split_clips,
            "missing_annotations": split_missing,
        }

    if not dry_run:
        for split in VALID_SPLITS:
            (runtime_root / split).mkdir(parents=True, exist_ok=True)
        (runtime_root / "videos").mkdir(parents=True, exist_ok=True)
        manifest_path = runtime_root / "runtime_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(runtime_root / "runtime_summary.json", summary)
    return summary


def run_prepare(args: argparse.Namespace) -> None:
    """Execute the ``prepare`` subcommand.

    Args:
        args: Parsed prepare arguments.
    """
    workspace_root = args.workspace_root.resolve()
    source_dir = (args.source_dir or workspace_root / "_bard_source").resolve()
    missing_metadata = [
        name for name in METADATA_FILES if not (source_dir / name).is_file()
    ]
    if missing_metadata:
        if not args.download:
            raise FileNotFoundError(
                f"Missing BARD metadata under {source_dir}: {missing_metadata}. "
                "Rerun with --download."
            )
        if args.dry_run:
            raise FileNotFoundError(
                "A prepare dry-run needs local metadata first; run --download once."
            )
        download_metadata(source_dir, args.max_workers)

    records, videos_by_game, players = load_metadata(source_dir)
    selected_games = select_games(
        videos_by_game, args.games, args.all_games, args.game_limit
    )
    selection_summary = [
        {"game": game, "video_count": len(videos_by_game[game])}
        for game in selected_games
    ]
    print(
        json.dumps({"selected_games": selection_summary}, ensure_ascii=False, indent=2)
    )
    if args.metadata_only:
        print(
            json.dumps(
                {
                    "status": "metadata_only",
                    "available_game_count": len(videos_by_game),
                    "selected_game_count": len(selected_games),
                    "source_dir": str(source_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.download and not args.dry_run:
        download_selected_videos(
            selected_games,
            videos_by_game,
            source_dir,
            workspace_root,
            args.max_workers,
            args.download_retries,
            args.retry_delay,
        )

    results = [
        prepare_game(
            game=game,
            expected_keys=videos_by_game[game],
            records=records,
            players=players,
            source_dir=source_dir,
            workspace_root=workspace_root,
            materialize=args.materialize,
            allow_partial=args.allow_partial,
            dry_run=args.dry_run,
        )
        for game in selected_games
    ]
    summary = {
        "schema_version": "bard_basketevent_staging.v2",
        "source_dir": str(source_dir),
        "workspace_root": str(workspace_root),
        "selected_game_count": len(selected_games),
        "games": results,
        "important": (
            "The BARD staging tree is source data. Generated trajectories, "
            "annotations, and reports belong under Settings.artifacts_root."
        ),
    }
    if not args.dry_run:
        write_json(workspace_root / "bard_staging_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_repair_manifest(args: argparse.Namespace) -> None:
    """Execute the ``repair-manifest`` subcommand.

    Args:
        args: Parsed manifest-repair arguments.

    Raises:
        FileNotFoundError: If an explicitly requested game is not staged.
    """
    workspace_root = args.workspace_root.resolve()
    discovered = discover_staging_games(workspace_root)
    games = list(dict.fromkeys(args.games)) if args.games else discovered
    unknown = sorted(set(games) - set(discovered))
    if unknown:
        raise FileNotFoundError(
            f"Games are not staged under {workspace_root}: {unknown}"
        )
    results = [
        repair_game_manifest(workspace_root / game, args.dry_run) for game in games
    ]
    print(
        json.dumps(
            {
                "schema_version": "bard_manifest_repair.v1",
                "workspace_root": str(workspace_root),
                "games": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_make_split(args: argparse.Namespace) -> None:
    """Execute the ``make-split`` subcommand.

    Args:
        args: Parsed split-generation arguments.
    """
    workspace_root = args.workspace_root.resolve()
    annotations_root = args.annotations_root.resolve()
    output = args.output.resolve()
    staged_games = discover_staging_games(workspace_root)
    games = discover_annotated_games(staged_games, annotations_root)
    if not games:
        raise FileNotFoundError(
            "No accepted annotations were found under "
            f"{annotations_root}/<game>/annotations. Run the fixed-rule label "
            "builder after SAM3/Qwen track preparation."
        )
    config = build_split_config(games, args.train_ratio, args.valid_ratio, args.seed)
    config["annotations_root"] = str(annotations_root)
    config["excluded_unannotated_games"] = sorted(set(staged_games) - set(games))
    if not args.dry_run:
        write_json(output, config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


def run_export(args: argparse.Namespace) -> None:
    """Execute the ``export`` subcommand.

    Args:
        args: Parsed export arguments.
    """
    workspace_root = args.workspace_root.resolve()
    split_file = args.split_file.resolve()
    splits = load_split_config(split_file)
    staged_games = set(discover_staging_games(workspace_root))
    configured_games = {game for split in VALID_SPLITS for game in splits[split]}
    unknown = sorted(configured_games - staged_games)
    if unknown:
        raise FileNotFoundError(f"Split file references unstaged games: {unknown}")
    summary = export_runtime(
        workspace_root=workspace_root,
        annotations_root=args.annotations_root.resolve(),
        runtime_root=args.runtime_root.resolve(),
        splits=splits,
        materialize=args.materialize,
        allow_missing_annotations=args.allow_missing_annotations,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the requested conversion stage.

    Args:
        argv: Optional argument sequence used by tests.
    """
    args = parse_args(argv)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "repair-manifest":
        run_repair_manifest(args)
    elif args.command == "make-split":
        run_make_split(args)
    elif args.command == "export":
        run_export(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
