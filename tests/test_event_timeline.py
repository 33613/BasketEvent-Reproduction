"""测试固定窗口事件到源视频时间线的整理过程。"""

import unittest

from src.modules.event_recognition.timeline import EventTimelineService
from src.modules.segmentation import VideoSegment


def make_segment(index: int, start: float, end: float) -> VideoSegment:
    """创建具有确定全局时间的测试窗口。"""
    return VideoSegment(
        segment_id=f"video-test_{index:05d}",
        source_video_id="video-test",
        index=index,
        start_seconds=start,
        end_seconds=end,
        source_start_frame=int(start * 30),
        source_end_frame=int(end * 30),
        duration_seconds=end - start,
        output_filename=f"window-{index}.mp4",
    )


class EventTimelineServiceTest(unittest.TestCase):
    """验证全局映射、窗口消重和素材边界。"""

    def test_local_time_is_mapped_and_overlap_is_deduplicated(self) -> None:
        """相邻重叠窗口中的同一进球应成为一条全局事件。"""
        segments = [make_segment(0, 0, 12), make_segment(1, 10, 22)]
        reports = {
            segments[0].segment_id: {
                "temporal_events": [
                    {
                        "player_id": "player_3",
                        "event": "Made Shot",
                        "confidence": 0.80,
                        "temporal_score": 0.60,
                        "start_time": 10.2,
                        "end_time": 11.8,
                    }
                ]
            },
            segments[1].segment_id: {
                "temporal_events": [
                    {
                        "player_id": "player_7",
                        "event": "Made Shot",
                        "confidence": 0.90,
                        "temporal_score": 0.70,
                        "start_time": 0.1,
                        "end_time": 1.9,
                    }
                ]
            },
        }

        report = EventTimelineService().build_timeline(
            segments=segments,
            prediction_reports=reports,
            source_duration_seconds=30.0,
        )

        self.assertEqual(len(report["candidates"]), 2)
        self.assertAlmostEqual(report["candidates"][0]["global_start_seconds"], 10.1)
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(len(report["events"][0]["candidate_ids"]), 2)
        self.assertAlmostEqual(report["material_drafts"][0]["start_seconds"], 5.1)
        self.assertAlmostEqual(report["material_drafts"][0]["end_seconds"], 13.9)

    def test_different_tracks_in_one_window_remain_separate(self) -> None:
        """同一窗口中两个球员的相同事件不能被时间重叠误合并。"""
        segment = make_segment(0, 0, 12)
        report = EventTimelineService().build_timeline(
            segments=[segment],
            prediction_reports={
                segment.segment_id: {
                    "temporal_events": [
                        {
                            "player_id": "player_1",
                            "event": "Rebound",
                            "start_time": 3.0,
                            "end_time": 4.0,
                        },
                        {
                            "player_id": "player_2",
                            "event": "Rebound",
                            "start_time": 3.2,
                            "end_time": 4.1,
                        },
                    ]
                }
            },
            source_duration_seconds=12.0,
        )

        self.assertEqual(len(report["events"]), 2)

    def test_material_range_is_clamped_to_source_video(self) -> None:
        """事件上下文不能超出源视频开始或结束。"""
        segment = make_segment(0, 0, 6)
        report = EventTimelineService().build_timeline(
            segments=[segment],
            prediction_reports={
                segment.segment_id: {
                    "temporal_events": [
                        {
                            "player_id": "player_0",
                            "event": "ast",
                            "start_time": 0.2,
                            "end_time": 1.0,
                        }
                    ]
                }
            },
            source_duration_seconds=6.0,
        )

        material = report["material_drafts"][0]
        self.assertEqual(material["start_seconds"], 0.0)
        self.assertEqual(material["end_seconds"], 3.0)
        self.assertEqual(material["pre_roll_seconds"], 7.0)


if __name__ == "__main__":
    unittest.main()
