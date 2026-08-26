"""验证素材目录的登记和查询。"""

from __future__ import annotations

import unittest

from src.modules.catalog import InMemoryMaterialCatalog, MaterialCatalogService


class MaterialCatalogServiceTest(unittest.TestCase):
    """验证处理结果可以被规范化并检索。"""

    def setUp(self) -> None:
        """为每个测试创建独立的内存目录。"""
        self.catalog = InMemoryMaterialCatalog()
        self.service = MaterialCatalogService(self.catalog)

    def test_register_and_query_processed_clip(self) -> None:
        """人物级事件应同时支持按事件和球衣身份查询。"""
        item = self.service.register_processed_clip(
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
        self.assertEqual(self.catalog.query(event="ast"), [item])
        self.assertEqual(self.catalog.query(participant_id="white#20"), [item])

    def test_duplicate_material_is_rejected(self) -> None:
        """同一来源和片段不能在未声明覆盖时重复登记。"""
        arguments = {
            "source_video_id": "game-001",
            "segment_id": "clip-100",
            "video_path": "clips/100.mp4",
            "start_seconds": 0.0,
            "end_seconds": 12.0,
            "prediction_report": {"player_predictions": [], "temporal_events": []},
        }
        self.service.register_processed_clip(**arguments)
        with self.assertRaises(ValueError):
            self.service.register_processed_clip(**arguments)


if __name__ == "__main__":
    unittest.main()
