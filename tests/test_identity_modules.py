"""验证最小身份流程的取样、观察、解析和编排规则。"""

from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from src.modules.identity.models import IdentityObservation, TrackCrop
from src.modules.identity.qwen_observer import QwenTrackObserver
from src.modules.identity.resolver import IdentityResolver, RosterLookup
from src.modules.identity.sampling import _uniform_positions
from src.modules.identity.service import IdentityService


def make_observation(
    track_id: str,
    frame_index: int,
    color: str | None,
    number: str | None,
    valid: bool = True,
) -> IdentityObservation:
    """创建用于规则测试的逐帧观察。"""
    return IdentityObservation(
        track_id=track_id,
        image_index=frame_index + 1,
        frame_index=frame_index,
        is_on_court_player=valid,
        jersey_color=color,
        jersey_number=number,
        confidence=0.9,
    )


class QwenTrackObserverTest(unittest.TestCase):
    """验证 Qwen 输出保留逐帧结果。"""

    def test_parser_preserves_frame_identity_switch(self) -> None:
        """同一轨迹中的 20 号与 13 号必须成为两个独立观察。"""
        crops = [
            TrackCrop("player_8", 1, 184, image=object()),
            TrackCrop("player_8", 2, 332, image=object()),
        ]
        output = """{
          "observations": [
            {"image_index": 1, "is_on_court_player": true,
             "jersey_color": "white", "jersey_number": "20", "confidence": 0.9},
            {"image_index": 2, "is_on_court_player": true,
             "jersey_color": "white", "jersey_number": "13", "confidence": 0.8}
          ]
        }"""

        observations = QwenTrackObserver.parse_observations(crops, output)

        self.assertEqual([item.frame_index for item in observations], [184, 332])
        self.assertEqual([item.jersey_number for item in observations], ["20", "13"])


class TrackSamplerTest(unittest.TestCase):
    """验证轨迹取样不会制造重复证据。"""

    def test_short_track_does_not_repeat_last_frame(self) -> None:
        """轨迹短于目标数量时只返回真实存在的不同位置。"""
        self.assertEqual(_uniform_positions(length=3, count=10), [0, 1, 2])


class IdentityResolverTest(unittest.TestCase):
    """验证三个明确、保守且可解释的身份解析分支。"""

    def test_one_candidate_creates_game_scoped_identity(self) -> None:
        """唯一颜色号码组合应生成比赛范围内稳定的人物编号。"""
        resolver = IdentityResolver(
            RosterLookup({("black", "17"): ("测试球员",)})
        )
        observations = [
            make_observation("player_3", 10, "black", "17"),
            make_observation("player_3", 20, "black", "17"),
        ]

        result = resolver.resolve(
            game_id="game-001",
            clip_id="101",
            track_id="player_3",
            observations=observations,
        )

        self.assertEqual(result.status, "identified")
        self.assertEqual(result.participant_id, "game-001:jersey:black:17")
        self.assertEqual(result.player_name, "测试球员")

    def test_conflict_keeps_track_with_anonymous_id(self) -> None:
        """身份切换不能被多数票掩盖，也不能导致轨迹被删除。"""
        result = IdentityResolver().resolve(
            game_id="game-001",
            clip_id="100",
            track_id="player_8",
            observations=[
                make_observation("player_8", 184, "white", "20"),
                make_observation("player_8", 332, "white", "13"),
            ],
        )

        self.assertEqual(result.status, "conflicting")
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.participant_id, "game-001:clip:100:track:player_8"
        )

    def test_missing_identity_keeps_track_as_anonymous(self) -> None:
        """Qwen 无法读号时仍应保留 SAM3 人物轨迹。"""
        result = IdentityResolver().resolve(
            game_id="game-001",
            clip_id="130",
            track_id="player_6",
            observations=[make_observation("player_6", 100, "white", None)],
        )

        self.assertEqual(result.status, "anonymous")
        self.assertTrue(result.accepted)


class FakeSampler:
    """避免读取真实视频的身份服务测试采样器。"""

    def sample(
        self,
        video_path: str,
        annotations: Mapping[str, Any],
    ) -> dict[str, list[TrackCrop]]:
        """返回一条固定人物轨迹。"""
        return {"player_0": [TrackCrop("player_0", 1, 5, image=object())]}

    def sample_ball_candidates(
        self,
        video_path: str,
        annotations: Mapping[str, Any],
    ) -> list[Any]:
        """本测试不提供篮球候选。"""
        return []


class FakeObserver:
    """为服务测试提供固定 Qwen 观察。"""

    def observe(self, crops: Sequence[TrackCrop]) -> list[IdentityObservation]:
        """返回一条号码可见的属性观察。"""
        return [make_observation(crops[0].track_id, 5, "black", "17")]


class FailingObserver:
    """模拟 Qwen 推理失败。"""

    def observe(self, crops: Sequence[TrackCrop]) -> list[IdentityObservation]:
        """抛出模型异常，验证服务的降级路径。"""
        raise RuntimeError("模型暂不可用")


class IdentityServiceTest(unittest.TestCase):
    """验证服务只编排最小流程并保留失败轨迹。"""

    def test_service_runs_sampling_observation_and_resolution(self) -> None:
        """正常观察应得到可进入事件模型的确定身份。"""
        service = IdentityService(
            sampler=FakeSampler(),
            observer=FakeObserver(),
            resolver=IdentityResolver(),
        )

        result = service.process(
            game_id="game-001",
            clip_id="101",
            video_path="unused.mp4",
            annotations={"player_0": {"trajectory": []}},
        )

        self.assertEqual(result.decisions[0].status, "identified")
        self.assertEqual(result.decisions[0].jersey_number, "17")

    def test_observer_failure_is_recorded_without_deleting_track(self) -> None:
        """Qwen 异常应形成匿名结果和错误记录，而不是终止片段。"""
        service = IdentityService(
            sampler=FakeSampler(),
            observer=FailingObserver(),
            resolver=IdentityResolver(),
        )

        result = service.process(
            game_id="game-001",
            clip_id="101",
            video_path="unused.mp4",
            annotations={"player_0": {"trajectory": []}},
        )

        self.assertEqual(result.decisions[0].status, "anonymous")
        self.assertIn("player_0", result.errors_by_track)


if __name__ == "__main__":
    unittest.main()
