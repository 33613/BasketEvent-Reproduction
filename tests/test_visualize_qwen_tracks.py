"""Tests for Qwen-to-SAM3 trajectory matching used by visualization."""

import unittest

from local_script.visualize_qwen_tracks import (
    active_temporal_events,
    build_player_labels,
    event_display_label,
    match_clean_ball_to_raw,
    match_clean_tracks_to_raw,
    normalize_temporal_events,
)


class QwenTrackVisualizationTest(unittest.TestCase):
    """Verify clean trajectories remain traceable to raw SAM3 IDs."""

    def test_renumbered_clean_player_matches_by_trajectory(self):
        """An old clean ID should recover its copied raw trajectory ID."""
        raw_tracks = {
            "player_0": {"trajectory": [[0, 0, 10, 20], None]},
            "player_7": {"trajectory": [[30, 40, 12, 24], [31, 40, 12, 24]]},
        }
        clean_tracks = {
            "player_0": {
                "player_name": "Day'Ron Sharpe",
                "jersey_number": "20",
                "jersey_color": "white",
                "trajectory": [[30, 40, 12, 24], [31, 40, 12, 24]],
            }
        }

        matches, diagnostics = match_clean_tracks_to_raw(raw_tracks, clean_tracks)
        labels = build_player_labels(raw_tracks, clean_tracks, matches)

        self.assertEqual(matches, {"player_0": "player_7"})
        self.assertEqual(diagnostics[0]["method"], "trajectory_signature")
        self.assertTrue(labels["player_7"]["accepted"])
        self.assertIn("#20", labels["player_7"]["text"])
        self.assertNotIn("Day'Ron Sharpe", labels["player_7"]["text"])
        self.assertFalse(labels["player_0"]["accepted"])

    def test_source_track_id_takes_precedence(self):
        """New clean files should use their explicit raw-track reference."""
        raw_tracks = {
            "player_3": {"trajectory": [[1, 2, 3, 4]]},
        }
        clean_tracks = {
            "player_0": {
                "source_track_id": "player_3",
                "trajectory": [[99, 99, 1, 1]],
            }
        }

        matches, diagnostics = match_clean_tracks_to_raw(raw_tracks, clean_tracks)

        self.assertEqual(matches, {"player_0": "player_3"})
        self.assertEqual(diagnostics[0]["method"], "source_track_id")

    def test_duplicate_trajectory_is_reported_as_ambiguous(self):
        """Two identical raw trajectories must never be guessed apart."""
        trajectory = [[10, 20, 30, 40]]
        raw_tracks = {
            "player_1": {"trajectory": trajectory},
            "player_2": {"trajectory": trajectory},
        }
        clean_tracks = {"player_0": {"trajectory": trajectory}}

        matches, diagnostics = match_clean_tracks_to_raw(raw_tracks, clean_tracks)

        self.assertEqual(matches, {})
        self.assertEqual(diagnostics[0]["status"], "ambiguous")
        self.assertEqual(
            diagnostics[0]["candidate_raw_track_ids"],
            ["player_1", "player_2"],
        )

    def test_selected_ball_matches_by_trajectory(self):
        """A selected clean basketball should recover its raw candidate ID."""
        raw_tracks = {
            "ball_1": {"trajectory": [[1, 1, 4, 4]]},
            "ball_2": {"trajectory": [[8, 8, 5, 5]]},
        }
        clean_tracks = {"ball": {"trajectory": [[8, 8, 5, 5]]}}

        raw_id, diagnostic = match_clean_ball_to_raw(raw_tracks, clean_tracks)

        self.assertEqual(raw_id, "ball_2")
        self.assertEqual(diagnostic["status"], "matched")

    def test_temporal_events_are_validated_sorted_and_activated(self):
        """Timeline helpers should ignore malformed events and find active ones."""
        report = {
            "duration_seconds": 10.0,
            "temporal_events": [
                {
                    "player_id": "player_1",
                    "jersey_number": "20",
                    "event": "Assist",
                    "confidence": 0.8,
                    "start_time": 5.0,
                    "end_time": 6.5,
                },
                {
                    "player_id": "player_0",
                    "jersey_number": "13",
                    "event": "Made Shot",
                    "confidence": 0.9,
                    "start_time": 2.0,
                    "end_time": 4.0,
                },
                {"event": "invalid", "start_time": 4.0, "end_time": 3.0},
            ],
        }

        events, duration = normalize_temporal_events(report)

        self.assertEqual(duration, 10.0)
        self.assertEqual([event["event"] for event in events], ["Made Shot", "Assist"])
        active = active_temporal_events(events, current_time=3.0)
        self.assertEqual(len(active), 1)
        self.assertEqual(event_display_label(active[0]), "#13 Made Shot 0.90")


if __name__ == "__main__":
    unittest.main()
