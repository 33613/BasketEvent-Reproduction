"""Unit tests for Qwen runtime-diagnostic parsing and roster lookup."""

import unittest

from tests.run_qwen_diagnostics import (
    RosterIndex,
    classify_failure_stage,
    normalize_jersey_number,
    parse_json_response,
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


if __name__ == "__main__":
    unittest.main()
