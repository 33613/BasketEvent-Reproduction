"""测试最终素材复核可视化中的时间与身份连接逻辑。"""

import unittest

from src.application.visualize_final_materials import (
    TrackReference,
    _event_identity_label,
    _material_index,
    bbox_for_global_time,
)


class FinalMaterialVisualizationTest(unittest.TestCase):
    """验证不依赖OpenCV的核心数据映射。"""

    def test_material_is_joined_with_global_event(self) -> None:
        """导出素材应通过event_id找到事件名称和时间。"""
        timeline = {
            "events": [{"event_id": "event-1", "event": "Made Shot"}],
            "material_drafts": [
                {
                    "material_id": "material-1",
                    "start_seconds": 3.0,
                    "end_seconds": 9.0,
                }
            ],
        }
        finalization = {
            "exported_materials": [
                {
                    "material_id": "material-1",
                    "video_path": "final_materials/00000.mp4",
                    "start_seconds": 3.0,
                    "end_seconds": 9.0,
                    "event_ids": ["event-1"],
                }
            ]
        }

        rows = _material_index(timeline, finalization)

        self.assertEqual(rows[0]["events"][0]["event"], "Made Shot")
        self.assertEqual(rows[0]["start_seconds"], 3.0)

    def test_bbox_uses_window_global_time(self) -> None:
        """素材全局时间应映射到窗口轨迹的对应帧。"""
        reference = TrackReference(
            reference="window-1/player_3",
            window_id="window-1",
            track_id="player_3",
            window_start=10.0,
            window_end=12.0,
            payload={"trajectory": [[0, 0, 10, 10], [20, 20, 10, 10]]},
        )

        bbox, source = bbox_for_global_time([reference], 11.1)

        self.assertEqual(bbox, [20.0, 20.0, 10.0, 10.0])
        self.assertEqual(source, "window-1/player_3")

    def test_identity_label_preserves_conflicting_candidates(self) -> None:
        """冲突身份不能被错误压缩成一个号码。"""
        label = _event_identity_label(
            {
                "status": "conflicting",
                "candidates": [
                    {"jersey_color": "white", "jersey_number": "13"},
                    {"jersey_color": "white", "jersey_number": "20"},
                ],
            }
        )

        self.assertIn("white #13", label)
        self.assertIn("white #20", label)


if __name__ == "__main__":
    unittest.main()
