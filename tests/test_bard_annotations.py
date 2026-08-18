"""Tests for deterministic BARD roster and label preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from local_script.build_bard_annotations import run_labels
from local_script.convert_bard_subset import export_runtime
from src.bard.labeling import (
    BardAnnotationBuilder,
    BardLabelMapper,
    normalize_jersey_number,
)
from src.bard.roster import BardRosterAdapter


def _track(number: str, color: str) -> dict[str, object]:
    """Return a minimal valid cleaned player track for a unit test."""
    return {
        "jersey_number": number,
        "jersey_color": color,
        "player_name": None,
        "trajectory": [[1, 2, 3, 4]],
    }


class BardLabelMapperTest(unittest.TestCase):
    """Verify the fixed BARD-to-BasketEvent event mapping."""

    def test_made_shot_and_assist_create_two_contributions(self) -> None:
        """A made assisted shot labels the scorer and assistant exactly once."""
        document = {
            "numerosity": 1,
            "actions": [
                {
                    "player": "13",
                    "action": "2PT Shot",
                    "result": True,
                    "assisted": True,
                    "other_player": "20",
                    "color": "white",
                }
            ],
        }
        result = BardLabelMapper().map_document(document)

        self.assertTrue(result.accepted)
        self.assertEqual(
            [(item.jersey_number, item.label) for item in result.contributions],
            [("13", "Made Shot"), ("20", "ast")],
        )

    def test_distinct_labels_for_one_actor_are_excluded(self) -> None:
        """Scheme A rejects a player with shot and rebound labels in one clip."""
        document = {
            "numerosity": 2,
            "actions": [
                {
                    "player": "2",
                    "action": "2PT Shot",
                    "result": False,
                    "color": "white",
                },
                {
                    "player": "2",
                    "action": "Rebound",
                    "result": None,
                    "color": "white",
                },
            ],
        }
        result = BardLabelMapper().map_document(document)

        self.assertFalse(result.accepted)
        self.assertIn("MULTI_LABEL_ACTOR", {item.code for item in result.anomalies})

    def test_unknown_action_is_not_blank(self) -> None:
        """Unsupported BARD actions are visible errors rather than false blanks."""
        document = {
            "numerosity": 1,
            "actions": [{"player": "3", "action": "Violation", "color": "black"}],
        }
        result = BardLabelMapper().map_document(document)

        self.assertFalse(result.accepted)
        self.assertEqual(result.contributions, ())
        self.assertIn("UNMAPPED_ACTION", {item.code for item in result.anomalies})

    def test_zero_and_double_zero_remain_distinct(self) -> None:
        """The identity normalizer must not merge NBA jerseys 0 and 00."""
        self.assertEqual(normalize_jersey_number("0"), "0")
        self.assertEqual(normalize_jersey_number("00"), "00")
        self.assertNotEqual(normalize_jersey_number("0"), normalize_jersey_number("00"))


class BardAnnotationBuilderTest(unittest.TestCase):
    """Verify strict identity association and final author-compatible JSON."""

    def test_builder_attaches_events_and_keeps_blank_player(self) -> None:
        """Matched actors receive events and an uninvolved resolved player is blank."""
        tracks = {
            "player_0": _track("13", "white"),
            "player_1": _track("20", "white"),
            "player_2": _track("28", "black"),
            "ball": {"trajectory": [[5, 6, 7, 8]]},
        }
        actions = {
            "numerosity": 1,
            "actions": [
                {
                    "player": "13",
                    "action": "2PT Shot",
                    "result": True,
                    "assisted": True,
                    "other_player": "20",
                    "color": "white",
                }
            ],
        }
        result = BardAnnotationBuilder().build(
            tracks,
            actions,
            bard_game="bkn-vs-det-0022400861",
            game_id="0022400861",
            video_name="100",
        )

        self.assertTrue(result.accepted)
        assert result.annotation is not None
        self.assertEqual(
            result.annotation["player_0"]["event"]["actionType"], "Made Shot"
        )
        self.assertEqual(result.annotation["player_1"]["event"]["actionType"], "ast")
        self.assertNotIn("event", result.annotation["player_2"])

    def test_builder_rejects_ambiguous_tracks(self) -> None:
        """Two tracks with one identity cannot receive a unique GT assignment."""
        tracks = {
            "player_0": _track("13", "white"),
            "player_1": _track("13", "white"),
            "ball": {"trajectory": [[5, 6, 7, 8]]},
        }
        actions = {
            "numerosity": 1,
            "actions": [
                {
                    "player": "13",
                    "action": "2PT Shot",
                    "result": True,
                    "color": "white",
                }
            ],
        }
        result = BardAnnotationBuilder().build(
            tracks,
            actions,
            bard_game="bkn-vs-det-0022400861",
            game_id="0022400861",
            video_name="100",
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            "AMBIGUOUS_TRACK_IDENTITY",
            {item["code"] for item in result.report["anomalies"]},
        )

    def test_builder_rejects_empty_ball_trajectory(self) -> None:
        """The training loader requires a non-empty cleaned ball trajectory."""
        tracks = {
            "player_0": _track("13", "white"),
            "ball": {"trajectory": []},
        }
        actions = {
            "numerosity": 1,
            "actions": [
                {
                    "player": "13",
                    "action": "2PT Shot",
                    "result": True,
                    "color": "white",
                }
            ],
        }
        result = BardAnnotationBuilder().build(
            tracks,
            actions,
            bard_game="bkn-vs-det-0022400861",
            game_id="0022400861",
            video_name="100",
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            "MISSING_BALL_TRACK",
            {item["code"] for item in result.report["anomalies"]},
        )


class BardRosterAdapterTest(unittest.TestCase):
    """Verify BARD roster compatibility without action-label leakage."""

    def test_adapter_preserves_double_zero_and_reports_ambiguity(self) -> None:
        """Season duplicates remain visible instead of being guessed away."""
        source = {
            "game": "bkn-vs-det-0022400861",
            "teams": ["BRK", "DET"],
            "players": [
                {"team": "DET", "name": "Player A", "jersey_numbers": ["00"]},
                {"team": "DET", "name": "Player B", "jersey_numbers": ["00"]},
                {"team": "BRK", "name": "Player C", "jersey_numbers": ["0"]},
            ],
        }
        result = BardRosterAdapter().convert(
            source,
            {"BRK": "white", "DET": "black"},
            "0022400861",
        )

        self.assertTrue(result.accepted)
        assert result.roster is not None
        self.assertIn("00", {item["jersey"] for item in result.roster["players"]})
        self.assertIn(
            "AMBIGUOUS_SEASON_ROSTER_IDENTITY",
            {item.code for item in result.anomalies},
        )

    def test_adapter_requires_independent_team_colors(self) -> None:
        """Missing color configuration is an error, not an inferred field."""
        source = {
            "game": "bkn-vs-det-0022400861",
            "teams": ["BRK", "DET"],
            "players": [],
        }
        result = BardRosterAdapter().convert(source, {"BRK": "white"}, "0022400861")

        self.assertFalse(result.accepted)
        self.assertIn("MISSING_TEAM_COLOR", {item.code for item in result.anomalies})


class BardAnnotationCliTest(unittest.TestCase):
    """Verify that batch orchestration writes the planned artifact layout."""

    def test_run_labels_writes_annotation_report_and_summary(self) -> None:
        """An accepted clip should create three auditable JSON outputs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "basket"
            artifacts_root = root / "basket_artifacts"
            game = "bkn-vs-det-0022400861"
            video_dir = data_root / game / "video"
            action_dir = data_root / game / "description" / "action"
            clean_dir = artifacts_root / game / "tracks" / "clean"
            video_dir.mkdir(parents=True)
            action_dir.mkdir(parents=True)
            clean_dir.mkdir(parents=True)
            (video_dir / "100.mp4").touch()
            (action_dir / "100.json").write_text(
                json.dumps(
                    {
                        "numerosity": 1,
                        "actions": [
                            {
                                "player": "13",
                                "action": "2PT Shot",
                                "result": True,
                                "assisted": False,
                                "color": "white",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (clean_dir / "100.json").write_text(
                json.dumps(
                    {
                        "player_0": _track("13", "white"),
                        "ball": {"trajectory": [[5, 6, 7, 8]]},
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                data_root=data_root,
                artifacts_root=artifacts_root,
                games=[game],
                clips=["100"],
                overwrite=False,
                fail_fast=True,
                dry_run=False,
            )

            summary = run_labels(args)

            self.assertEqual(summary["accepted"], 1)
            self.assertTrue(
                (artifacts_root / game / "annotations" / "100.json").is_file()
            )
            self.assertTrue((artifacts_root / game / "reports" / "100.json").is_file())
            self.assertTrue(
                (artifacts_root / "annotation_build_summary.json").is_file()
            )

    def test_export_reads_annotations_from_artifacts(self) -> None:
        """Runtime export must not depend on labels inside BARD source data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "basket"
            artifacts_root = root / "basket_artifacts"
            runtime_root = root / "basket_runtime"
            game = "bkn-vs-det-0022400861"
            video = data_root / game / "video" / "100.mp4"
            annotation = artifacts_root / game / "annotations" / "100.json"
            video.parent.mkdir(parents=True)
            annotation.parent.mkdir(parents=True)
            video.write_bytes(b"test-video")
            annotation.write_text(
                json.dumps(
                    {
                        "player_0": {
                            **_track("13", "white"),
                            "event": {"actionType": "Made Shot"},
                        },
                        "ball": {"trajectory": [[5, 6, 7, 8]]},
                    }
                ),
                encoding="utf-8",
            )

            summary = export_runtime(
                workspace_root=data_root,
                annotations_root=artifacts_root,
                runtime_root=runtime_root,
                splits={"train": [game], "valid": [], "test": []},
                materialize="hardlink",
                allow_missing_annotations=False,
                dry_run=False,
            )

            self.assertEqual(summary["splits"]["train"]["clips"], 1)
            self.assertTrue(
                (runtime_root / "videos" / "0022400861" / "100.mp4").is_file()
            )
            self.assertTrue(
                (runtime_root / "train" / "0022400861" / "100.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
