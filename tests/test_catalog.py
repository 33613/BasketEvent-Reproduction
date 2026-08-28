"""验证模型输出可以整理为统一素材对象。"""

from __future__ import annotations

import unittest

from src.modules.catalog import CatalogService


class CatalogServiceTest(unittest.TestCase):
    """验证处理结果可以被规范化。"""

    def setUp(self) -> None:
        """为每个测试创建无状态素材服务。"""
        self.service = CatalogService()

    def test_build_processed_clip(self) -> None:
        """人物级事件应整理为事件和球衣身份。"""
        item = self.service.build_material(
            source_video_id="game-001",
            segment_id="clip-100",
            video_path="clips/100.mp4",
            start_seconds=10.0,
            end_seconds=22.0,
            prediction_report={
                "player_predictions": [
                    {
                        "player_id": "player_8",
                        "jersey_color": "white",
                        "jersey_number": "20",
                        "event": "ast",
                        "confidence": 0.91,
                    }
                ],
                "temporal_events": [
                    {
                        "player_id": "player_8",
                        "event": "ast",
                        "confidence": 0.91,
                        "start_time": 12.0,
                        "end_time": 14.0,
                    }
                ],
            },
        )

        self.assertEqual(item.material_id, "game-001:clip-100")
        self.assertEqual(item.duration_seconds, 12.0)
        self.assertEqual(item.participants[0].participant_id, "white#20")
        self.assertEqual(item.events[0].event, "ast")

    def test_invalid_time_range_is_rejected(self) -> None:
        """素材结束时间不能早于开始时间。"""
        with self.assertRaises(ValueError):
            self.service.build_material(
                source_video_id="game-001",
                segment_id="clip-100",
                video_path="clips/100.mp4",
                start_seconds=12.0,
                end_seconds=0.0,
                prediction_report={"player_predictions": [], "temporal_events": []},
            )


if __name__ == "__main__":
    unittest.main()
