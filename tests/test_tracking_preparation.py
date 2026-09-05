"""测试不依赖身份识别的轨迹结构准备。"""

import json
import tempfile
import unittest
from pathlib import Path

from src.modules.tracking.preparation import (
    prepare_model_tracks,
    prepare_model_tracks_file,
)


class TrackPreparationTest(unittest.TestCase):
    """验证 Qwen 失败不会删除有效人物轨迹。"""

    def test_all_structurally_valid_players_are_retained(self) -> None:
        """只按边界框有效性保留人物，不要求号码或姓名。"""
        prepared = prepare_model_tracks(
            {
                "player_0": {"trajectory": [[1, 2, 10, 20], None]},
                "player_1": {"trajectory": [None, [2, 3, -1, 20]]},
                "player_2": {"trajectory": [None, [5, 6, 7, 8]]},
                "ball_1": {"trajectory": [[1, 1, 2, 2], None]},
                "ball_2": {"trajectory": [[1, 1, 2, 2], [2, 2, 2, 2]]},
            }
        )

        self.assertEqual(
            [key for key in prepared if key.startswith("player_")],
            ["player_0", "player_2"],
        )
        self.assertEqual(prepared["player_0"]["identity_status"], "unresolved")
        self.assertEqual(prepared["ball"]["source_track_id"], "ball_2")

    def test_file_report_declares_no_identity_filtering(self) -> None:
        """审计报告应明确本步骤未使用身份信息过滤。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.json"
            output = root / "model.json"
            report = root / "report.json"
            source.write_text(
                json.dumps({"player_7": {"trajectory": [[0, 0, 10, 10]]}}),
                encoding="utf-8",
            )

            value = prepare_model_tracks_file(source, output, report)

            self.assertEqual(value["player_count"], 1)
            self.assertFalse(value["identity_was_used_for_filtering"])
            self.assertTrue(output.is_file())
            self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
