"""验证Qwen实际输出变体不会导致整批身份轨迹失败。"""

from __future__ import annotations

import unittest

from src.modules.identity.models import TrackCrop
from src.modules.identity.qwen_observer import QwenTrackObserver


def _crops(count: int = 2) -> list[TrackCrop]:
    """创建不依赖OpenCV的最小截图元数据。"""
    return [
        TrackCrop("player_1", index, index * 10, image=object())
        for index in range(1, count + 1)
    ]


class QwenObserverParserTest(unittest.TestCase):
    """覆盖服务器上出现的连续JSON和顶层数组格式。"""

    def test_concatenated_objects_are_merged_as_observations(self) -> None:
        """连续对象不能再触发Extra data。"""
        output = """```json
{"image_index": 1, "is_on_court_player": true,
 "jersey_color": "white", "jersey_number": "13", "confidence": 0.9}
```
```json
{"image_index": 2, "is_on_court_player": true,
 "jersey_color": "white", "jersey_number": "13", "confidence": 0.8}
```"""

        observations = QwenTrackObserver.parse_observations(_crops(), output)

        self.assertEqual(len(observations), 2)
        self.assertEqual([value.jersey_number for value in observations], ["13", "13"])

    def test_top_level_array_is_accepted(self) -> None:
        """顶层数组应被视为逐帧观察列表。"""
        output = """[
          {"image_index": 1, "is_on_court_player": true,
           "jersey_color": "black", "jersey_number": "17", "confidence": 0.9},
          {"image_index": 2, "is_on_court_player": true,
           "jersey_color": "black", "jersey_number": null, "confidence": 0.2}
        ]"""

        observations = QwenTrackObserver.parse_observations(_crops(), output)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].jersey_number, "17")
        self.assertIsNone(observations[1].jersey_number)

    def test_concatenated_wrappers_are_merged(self) -> None:
        """模型逐段重复observations包装时应合并有效帧。"""
        output = """{"observations": [
          {"image_index": 1, "is_on_court_player": true,
           "jersey_color": "white", "jersey_number": "13", "confidence": 0.9}
        ]}
        {"observations": [
          {"image_index": 2, "is_on_court_player": true,
           "jersey_color": "white", "jersey_number": "13", "confidence": 0.8}
        ]}"""

        observations = QwenTrackObserver.parse_observations(_crops(), output)

        self.assertEqual(len(observations), 2)
        self.assertEqual([value.image_index for value in observations], [1, 2])

    def test_standard_wrapper_wins_over_repeated_suffix(self) -> None:
        """标准结果后出现重复对象时，应使用完整observations数组。"""
        output = """{
          "observations": [
            {"image_index": 1, "is_on_court_player": true,
             "jersey_color": "white", "jersey_number": "20", "confidence": 0.9}
          ]
        }
        {"image_index": 2, "jersey_number": "99"}"""

        observations = QwenTrackObserver.parse_observations(_crops(), output)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].jersey_number, "20")


if __name__ == "__main__":
    unittest.main()
