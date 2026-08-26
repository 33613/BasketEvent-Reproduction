"""验证身份模块之间的稳定数据契约。"""

from __future__ import annotations

import unittest

from src.modules.identity.clustering import CrossClipIdentityClusterer
from src.modules.identity.models import IdentityObservation, TrackCrop
from src.modules.identity.qwen_observer import QwenTrackObserver
from src.modules.identity.resolver import IdentityResolver
from src.modules.identity.sampling import _uniform_positions


def make_observation(
    track_id: str,
    frame_index: int,
    color: str | None,
    number: str | None,
    valid: bool = True,
) -> IdentityObservation:
    """创建用于解析器测试的最小逐帧观察。"""
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
    """验证 Qwen 输出解析不会掩盖逐帧冲突。"""

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
    """验证轨迹取样不会伪造重复帧证据。"""

    def test_short_track_does_not_repeat_last_frame(self) -> None:
        """轨迹短于目标数量时只返回真实存在的不同位置。"""
        positions = _uniform_positions(length=3, count=10)

        self.assertEqual(positions, [0, 1, 2])
        self.assertEqual(len(positions), len(set(positions)))


class IdentityResolverTest(unittest.TestCase):
    """验证稳定、混合和未解析轨迹的保留策略。"""

    def test_stable_number_is_retained_without_roster(self) -> None:
        """产品模式没有名单时仍应保留颜色和号码。"""
        result = IdentityResolver().resolve(
            "player_6",
            [
                make_observation("player_6", 10, "white", "13"),
                make_observation("player_6", 20, "white", "13"),
            ],
        )

        self.assertEqual(result.status, "stable")
        self.assertTrue(result.accepted)
        self.assertEqual(result.jersey_number, "13")
        self.assertIsNone(result.player_name)

    def test_mixed_track_is_not_majority_voted(self) -> None:
        """轨迹中出现多个号码时必须等待拆分，不能保留整条轨迹。"""
        result = IdentityResolver().resolve(
            "player_8",
            [
                make_observation("player_8", 184, "white", "20"),
                make_observation("player_8", 332, "white", "13"),
                make_observation("player_8", 406, "white", "20"),
            ],
        )

        self.assertEqual(result.status, "mixed")
        self.assertFalse(result.accepted)
        self.assertEqual(
            {item.jersey_number for item in result.candidates}, {"13", "20"}
        )

    def test_unreadable_number_keeps_confirmed_player(self) -> None:
        """号码不可读不应再触发旧版整轨迹硬删除。"""
        result = IdentityResolver().resolve(
            "player_3",
            [make_observation("player_3", 100, "black", None)],
        )

        self.assertEqual(result.status, "unresolved")
        self.assertTrue(result.accepted)

    def test_roster_lookup_is_exact(self) -> None:
        """只有颜色和号码唯一匹配时才补充真实姓名。"""
        resolver = IdentityResolver({("white", "20"): ("Day'Ron Sharpe",)})
        result = resolver.resolve(
            "player_8", [make_observation("player_8", 184, "white", "20")]
        )

        self.assertEqual(result.player_name, "Day'Ron Sharpe")


class CrossClipIdentityClustererTest(unittest.TestCase):
    """验证不同片段中的同号同色轨迹会归入同一素材人物组。"""

    def test_exact_identity_is_reused_across_clips(self) -> None:
        """跨片段确定性基线应返回相同聚类编号。"""
        resolver = IdentityResolver()
        first = resolver.resolve(
            "player_0", [make_observation("player_0", 1, "black", "17")]
        )
        second = resolver.resolve(
            "player_3", [make_observation("player_3", 2, "black", "17")]
        )
        clusterer = CrossClipIdentityClusterer()

        first_assignment = clusterer.add_clip("clip_101", [first])
        second_assignment = clusterer.add_clip("clip_102", [second])

        self.assertEqual(first_assignment["player_0"], second_assignment["player_3"])
        self.assertEqual(len(clusterer.clusters()[0].members), 2)


if __name__ == "__main__":
    unittest.main()
