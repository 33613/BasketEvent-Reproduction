"""验证事件主体身份只复用被事件引用的窗口轨迹。"""

import tempfile
import unittest
from pathlib import Path

from src.modules.identity.event_actor import EventActorIdentityService
from src.modules.identity.models import IdentityObservation, TrackCrop
from src.modules.identity.resolver import IdentityResolver


class FakeSampler:
    """返回确定截图并记录实际请求的轨迹编号。"""

    sample_count = 10
    pad_ratio = 0.0

    def __init__(self) -> None:
        self.requests: list[tuple[str, ...] | None] = []

    def load_annotations(self, path):
        """返回两个可供测试引用的人物轨迹。"""
        return {
            "player_3": {"trajectory": [[1, 1, 10, 20]]},
            "player_8": {"trajectory": [[2, 2, 10, 20]]},
        }

    def sample(
        self,
        video_path,
        annotations,
        track_prefix="player",
        track_ids=None,
    ):
        """只为调用者指定的轨迹生成一张占位截图。"""
        requested = tuple(track_ids) if track_ids is not None else None
        self.requests.append(requested)
        selected = requested or tuple(annotations)
        return {
            track_id: [
                TrackCrop(
                    track_id=track_id,
                    image_index=1,
                    frame_index=10,
                    image=object(),
                    sharpness=100.0,
                )
            ]
            for track_id in selected
        }


class FakeObserver:
    """按轨迹编号返回确定的颜色号码，并统计Qwen调用次数。"""

    def __init__(self) -> None:
        self.call_count = 0

    def observe(self, crops):
        """player_3返回黑17，player_8返回白20。"""
        self.call_count += 1
        crop = crops[0]
        color, number = (
            ("black", "17") if crop.track_id == "player_3" else ("white", "20")
        )
        return [
            IdentityObservation(
                track_id=crop.track_id,
                image_index=crop.image_index,
                frame_index=crop.frame_index,
                is_on_court_player=True,
                jersey_color=color,
                jersey_number=number,
                confidence=0.9,
                evidence="测试证据",
            )
        ]


class EventActorIdentityServiceTest(unittest.TestCase):
    """验证事件、窗口轨迹、Qwen证据和稳定人物编号的绑定。"""

    def test_reuses_one_track_for_multiple_events_and_resumes_cache(self):
        """同一窗口轨迹被多个事件引用时只能调用一次观察器。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            tracks = root / "tracks"
            cache = root / "cache"
            videos.mkdir()
            tracks.mkdir()
            (videos / "window_00001.mp4").write_bytes(b"video")
            (tracks / "window_00001.json").write_text("{}", encoding="utf-8")

            timeline = {
                "events": [
                    {
                        "event_id": "event-1",
                        "event": "Made Shot",
                        "track_references": ["window_00001/player_3"],
                    },
                    {
                        "event_id": "event-2",
                        "event": "Rebound",
                        "track_references": ["window_00001/player_3"],
                    },
                ]
            }
            sampler = FakeSampler()
            observer = FakeObserver()
            service = EventActorIdentityService(
                sampler=sampler,
                observer=observer,
                resolver=IdentityResolver(),
            )

            first = service.process(
                source_video_id="video-test",
                timeline_report=timeline,
                window_video_directory=videos,
                raw_tracks_directory=tracks,
                cache_directory=cache,
            )
            second = service.process(
                source_video_id="video-test",
                timeline_report=timeline,
                window_video_directory=videos,
                raw_tracks_directory=tracks,
                cache_directory=cache,
            )

            self.assertEqual(first["event_count"], 2)
            self.assertEqual(first["unique_track_reference_count"], 1)
            self.assertEqual(observer.call_count, 1)
            self.assertEqual(sampler.requests, [("player_3",)])
            self.assertEqual(
                first["event_resolutions"][0]["participant_id"],
                "video-test:jersey:black:17",
            )
            self.assertEqual(
                second["event_resolutions"][1]["participant_id"],
                "video-test:jersey:black:17",
            )

    def test_conflicting_window_evidence_is_not_forced_to_one_identity(self):
        """重叠窗口支持不同号码时应保留冲突，而不是多数投票。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            tracks = root / "tracks"
            videos.mkdir()
            tracks.mkdir()
            for window_id in ("window_00001", "window_00002"):
                (videos / f"{window_id}.mp4").write_bytes(b"video")
                (tracks / f"{window_id}.json").write_text("{}", encoding="utf-8")

            report = EventActorIdentityService(
                sampler=FakeSampler(),
                observer=FakeObserver(),
                resolver=IdentityResolver(),
            ).process(
                source_video_id="video-test",
                timeline_report={
                    "events": [
                        {
                            "event_id": "event-1",
                            "event": "Made Shot",
                            "track_references": [
                                "window_00001/player_3",
                                "window_00002/player_8",
                            ],
                        }
                    ]
                },
                window_video_directory=videos,
                raw_tracks_directory=tracks,
                cache_directory=root / "cache",
            )

            resolution = report["event_resolutions"][0]
            self.assertEqual(resolution["status"], "conflicting")
            self.assertEqual(len(resolution["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
