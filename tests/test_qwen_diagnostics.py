"""Unit tests for Qwen runtime-diagnostic parsing and roster lookup."""

import unittest

from tests.run_qwen_diagnostics import (
    RosterIndex,
    _uniform_indices,
    aggregate_temporal_observations,
    classify_failure_stage,
    find_decomposed_output_inconsistencies,
    normalize_jersey_number,
    parse_json_response,
    validate_temporal_observation,
)


class QwenDiagnosticTest(unittest.TestCase):
    """Verify diagnostic stages remain deterministic without loading Qwen."""

    def setUp(self):
        """Create a two-team roster with distinct color-number identities."""
        self.roster = RosterIndex.from_document(
            {
                "jersey_color": {"Nets": "white", "Pistons": "black"},
                "players": [
                    {"team_name": "Nets", "jersey": "20", "name": "Player A"},
                    {"team_name": "Pistons", "jersey": "0", "name": "Player B"},
                ],
            }
        )

    def test_markdown_json_is_parsed(self):
        """A fenced JSON response should remain auditable and parseable."""
        parsed, error = parse_json_response(
            '```json\n{"is_on_court_player": true, "jersey_number": "20"}\n```'
        )

        self.assertIsNone(error)
        self.assertEqual(parsed["jersey_number"], "20")

    def test_lookup_uses_exact_color_and_number(self):
        """Roster lookup should resolve identity without a model-generated name."""
        result = self.roster.lookup(" WHITE ", "20")

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["player_name"], "Player A")

    def test_jersey_zero_and_double_zero_remain_distinct(self):
        """Normalization must not silently merge the NBA numbers 0 and 00."""
        self.assertEqual(normalize_jersey_number("0"), "0")
        self.assertEqual(normalize_jersey_number("00"), "00")
        self.assertNotEqual(
            self.roster.lookup("black", "0")["status"],
            self.roster.lookup("black", "00")["status"],
        )

    def test_failure_stage_separates_validity_from_number_ocr(self):
        """A valid player with no number should fail at OCR, not validation."""
        parsed = {
            "is_on_court_player": True,
            "jersey_color": "white",
            "jersey_number": None,
        }

        stage = classify_failure_stage(parsed, None, self.roster.lookup("white", None))

        self.assertEqual(stage, "jersey_number")

    def test_complete_identity_reaches_roster_match(self):
        """A visible valid identity should pass all four diagnostic stages."""
        parsed = {
            "is_on_court_player": True,
            "jersey_color": "white",
            "jersey_number": "20",
        }
        lookup = self.roster.lookup("white", "20")

        self.assertEqual(classify_failure_stage(parsed, None, lookup), "complete")

    def test_uniform_indices_never_repeat_an_observation(self):
        """A short trajectory must not be padded with duplicated frames."""
        indices = _uniform_indices(length=3, count=10)

        self.assertEqual(indices, [0, 1, 2])

    def test_free_text_number_conflict_is_reported(self):
        """A number mentioned in prose cannot coexist with a null field."""
        parsed = {
            "is_on_court_player": True,
            "jersey_color": "white",
            "jersey_number": None,
            "crop_observations": [
                {
                    "image_index": 1,
                    "number_readable": False,
                    "jersey_number": None,
                    "note": "Player wearing white jersey with number 13.",
                }
            ],
        }

        issues = find_decomposed_output_inconsistencies(parsed)

        self.assertEqual(len(issues), 1)
        self.assertEqual(
            classify_failure_stage(parsed, None, self.roster.lookup("white", None)),
            "model_output_inconsistency",
        )

    def test_temporal_schema_contradiction_is_rejected(self):
        """Unreadable and non-null number fields are contradictory."""
        validation = validate_temporal_observation(
            {
                "is_on_court_player": True,
                "jersey_color_candidate": "white",
                "number_readable": False,
                "jersey_number_candidate": "20",
                "number_confidence": 0.9,
            },
            None,
            self.roster.allowed_colors,
        )

        self.assertEqual(validation["status"], "model_output_inconsistency")
        self.assertEqual(validation["jersey_number"], "20")

    def test_one_temporal_identity_is_stable(self):
        """Repeated consistent timestamp evidence should remain one identity."""
        observations = [
            {
                "frame_index": frame,
                "validation": {
                    "status": "valid",
                    "jersey_color": "white",
                    "jersey_number": "20",
                    "number_confidence": confidence,
                },
            }
            for frame, confidence in ((10, 0.9), (20, 0.8))
        ]

        aggregate = aggregate_temporal_observations(observations)

        self.assertEqual(aggregate["status"], "stable_identity")
        self.assertEqual(
            aggregate["selected_identity"],
            {"jersey_color": "white", "jersey_number": "20"},
        )

    def test_two_temporal_numbers_preserve_identity_switch(self):
        """Conflicting chronological numbers must not be majority-voted away."""
        observations = [
            {
                "frame_index": 100,
                "validation": {
                    "status": "valid",
                    "jersey_color": "white",
                    "jersey_number": "13",
                    "number_confidence": 0.95,
                },
            },
            {
                "frame_index": 400,
                "validation": {
                    "status": "valid",
                    "jersey_color": "white",
                    "jersey_number": "20",
                    "number_confidence": 0.90,
                },
            },
        ]

        aggregate = aggregate_temporal_observations(observations)

        self.assertEqual(aggregate["status"], "mixed_identity")
        self.assertIsNone(aggregate["selected_identity"])
        self.assertEqual(
            [item["jersey_number"] for item in aggregate["identity_candidates"]],
            ["13", "20"],
        )
        self.assertEqual(len(aggregate["proposed_observation_segments"]), 2)


if __name__ == "__main__":
    unittest.main()
