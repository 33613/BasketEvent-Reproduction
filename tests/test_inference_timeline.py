"""Tests for converting PlayNet outputs into temporal evidence windows."""

import unittest

import torch

from src.modules.event_recognition.inference import build_temporal_prediction_report


class TemporalPredictionReportTest(unittest.TestCase):
    """Verify clip indices and MIL evidence map back to video time."""

    def test_non_background_prediction_exports_top_evidence_clip(self):
        """The strongest valid gated clip should become a timeline interval."""
        outputs = {
            "logits_person": torch.tensor([[0.0, 4.0, 1.0]]),
            "logits_clip": torch.tensor(
                [
                    [[0.0, 1.0, 0.0]],
                    [[0.0, 3.0, 0.0]],
                ]
            ),
            "person_valid": torch.tensor([True]),
            "person_clip_valid": torch.tensor([[True], [True]]),
            "gate_weights": torch.tensor([[[0.2]], [[0.8]]]),
        }
        data = {
            "idx": torch.tensor([[0, 10, 20], [30, 40, 50]]),
            "fps_in": 10.0,
            "total_frames": 100,
        }

        report = build_temporal_prediction_report(
            video_path="clip.mp4",
            trajectory_data={
                "player_0": {"jersey_number": "13", "jersey_color": "white"}
            },
            player_ids=["player_0"],
            data=data,
            outputs=outputs,
            timeline_topk=1,
        )

        self.assertEqual(
            report["schema_version"], "basketevent_temporal_predictions.v1"
        )
        self.assertTrue(report["localization_is_diagnostic"])
        self.assertEqual(len(report["temporal_events"]), 1)
        event = report["temporal_events"][0]
        self.assertEqual(event["jersey_number"], "13")
        self.assertEqual(event["clip_index"], 1)
        self.assertEqual(event["start_time"], 3.0)
        self.assertEqual(event["end_time"], 5.1)
        player = report["player_predictions"][0]
        self.assertEqual(player["paper_gate_segment"]["clip_index"], 1)
        self.assertAlmostEqual(sum(player["class_probabilities"].values()), 1, places=6)

    def test_paper_gate_and_product_score_can_select_different_clips(self):
        """论文最高 gate 和产品 gate×分类概率必须独立保存。"""
        report = build_temporal_prediction_report(
            video_path="clip.mp4", trajectory_data={}, player_ids=["player_0"],
            data={"idx": torch.tensor([[0, 1], [10, 11]]), "fps_in": 10, "total_frames": 20},
            outputs={"logits_person": torch.tensor([[0., 4.]]),
                     "logits_clip": torch.tensor([[[4., 0.]], [[0., 4.]]]),
                     "person_valid": torch.tensor([True]),
                     "person_clip_valid": torch.tensor([[True], [True]]),
                     "gate_weights": torch.tensor([[[.8]], [[.2]]])}, timeline_topk=1)
        self.assertEqual(report["player_predictions"][0]["paper_gate_segment"]["clip_index"], 0)
        self.assertEqual(report["temporal_events"][0]["clip_index"], 1)

    def test_blank_prediction_has_no_timeline_interval(self):
        """Background predictions should remain visible only in the audit list."""
        outputs = {
            "logits_person": torch.tensor([[5.0, 0.0]]),
            "logits_clip": torch.tensor([[[5.0, 0.0]]]),
            "person_valid": torch.tensor([True]),
            "person_clip_valid": torch.tensor([[True]]),
            "gate_weights": torch.tensor([[[1.0]]]),
        }
        data = {
            "idx": torch.tensor([[0, 1]]),
            "fps_in": 25.0,
            "total_frames": 20,
        }

        report = build_temporal_prediction_report(
            video_path="blank.mp4",
            trajectory_data={"player_0": {}},
            player_ids=["player_0"],
            data=data,
            outputs=outputs,
            timeline_topk=1,
        )

        self.assertEqual(report["temporal_events"], [])
        self.assertEqual(report["player_predictions"][0]["label_id"], 0)


if __name__ == "__main__":
    unittest.main()
