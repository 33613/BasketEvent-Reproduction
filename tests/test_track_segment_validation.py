"""Unit tests for controlled PlayNet validation on one track segment."""

import argparse
import unittest
from pathlib import Path

from tests.run_track_segment_validation import (
    SegmentSpec,
    build_inference_command,
    build_segment_document,
    evaluate_prediction_report,
    normalize_event_name,
)


class TrackSegmentValidationTest(unittest.TestCase):
    """Verify trajectory masking and event evaluation without loading models."""

    def setUp(self) -> None:
        """Create a five-frame mixed-track fixture and a selected ball."""
        self.raw_tracks = {
            "player_8": {
                "trajectory": [
                    [0, 0, 10, 20],
                    [1, 0, 10, 20],
                    None,
                    [3, 0, 10, 20],
                    [4, 0, 10, 20],
                ]
            }
        }
        self.clean_tracks = {
            "player_0": {"trajectory": [[9, 9, 9, 9]] * 5},
            "ball": {
                "source_track_id": "ball_2",
                "trajectory": [[5, 5, 3, 3]] * 5,
            },
        }
        self.spec = SegmentSpec(
            raw_track_id="player_8",
            output_track_id="player_20_segment",
            start_frame=1,
            end_frame=3,
            jersey_color="white",
            jersey_number="20",
            player_name="Day'Ron Sharpe",
            evidence_note="fixture",
        )

    def test_segment_masks_boxes_outside_inclusive_range(self) -> None:
        """Source frame indices must remain aligned with the complete video."""
        document, statistics = build_segment_document(
            self.raw_tracks, self.clean_tracks, self.spec
        )

        trajectory = document["player_20_segment"]["trajectory"]
        self.assertEqual(
            trajectory,
            [None, [1, 0, 10, 20], None, [3, 0, 10, 20], None],
        )
        self.assertEqual(statistics["retained_valid_bbox_count"], 2)
        self.assertEqual(statistics["first_retained_frame"], 1)
        self.assertEqual(statistics["last_retained_frame"], 3)

    def test_segment_copies_only_selected_ball_and_one_player(self) -> None:
        """Unrelated Qwen players must not leak into the controlled input."""
        document, _ = build_segment_document(
            self.raw_tracks, self.clean_tracks, self.spec
        )

        self.assertEqual(set(document), {"player_20_segment", "ball"})
        self.assertEqual(document["ball"]["source_track_id"], "ball_2")
        self.assertEqual(document["player_20_segment"]["source_track_id"], "player_8")
        self.assertEqual(document["player_20_segment"]["jersey_number"], "20")

    def test_empty_selected_range_is_rejected(self) -> None:
        """An interval containing no valid box is not a usable experiment."""
        spec = SegmentSpec(
            **{
                **self.spec.__dict__,
                "start_frame": 2,
                "end_frame": 2,
            }
        )

        with self.assertRaisesRegex(ValueError, "contains no valid source boxes"):
            build_segment_document(self.raw_tracks, self.clean_tracks, spec)

    def test_assist_alias_matches_checkpoint_label(self) -> None:
        """Human-readable Assist should compare against the model label ast."""
        report = {
            "player_predictions": [
                {
                    "player_id": "player_20_segment",
                    "event": "ast",
                    "confidence": 0.71,
                }
            ]
        }

        evaluation = evaluate_prediction_report(report, "player_20_segment", "Assist")

        self.assertEqual(normalize_event_name("Assist"), "ast")
        self.assertTrue(evaluation["matched"])
        self.assertEqual(evaluation["actual_event"], "ast")

    def test_inference_command_targets_only_segment_player(self) -> None:
        """The generated command must isolate the experimental player key."""
        args = argparse.Namespace(
            video=Path("clip.mp4"),
            checkpoint=Path("playnet.pt"),
            timesformer_model=Path("timesformer"),
            gpu_id=0,
            output_track_id="player_20_segment",
            bag_clips=12,
            clip_len=8,
            fps_in=60,
            fps_out=4,
            img_size=224,
            topk=5,
            timeline_topk=2,
        )

        command = build_inference_command(
            args, Path("segment.json"), Path("prediction.json")
        )

        self.assertIn("--player_ids", command)
        self.assertEqual(
            command[command.index("--player_ids") + 1], "player_20_segment"
        )
        self.assertIn("prediction.json", command)


if __name__ == "__main__":
    unittest.main()
