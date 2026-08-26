"""验证 Identity 各组件之间的数据契约和职责边界。"""

from __future__ import annotations

import math
import unittest
from typing import Any, Mapping, Sequence

from src.modules.identity.association import CrossClipIdentityAssociator
from src.modules.identity.evidence.qwen import QwenTrackObserver
from src.modules.identity.evidence.reid import ReIdTrackEvidenceProvider
from src.modules.identity.fusion import IdentityEvidenceFusion
from src.modules.identity.gallery import InMemoryIdentityGallery
from src.modules.identity.models import (
    IdentityEvidence,
    IdentityObservation,
    IdentityProfile,
    TrackCrop,
)
from src.modules.identity.sampling import _uniform_positions
from src.modules.identity.service import IdentityService


def make_observation(
    track_id: str,
    frame_index: int,
    color: str | None,
    number: str | None,
    valid: bool = True,
) -> IdentityObservation:
    """创建用于融合测试的最小逐帧观察。"""
    return IdentityObservation(
        track_id=track_id,
        image_index=frame_index + 1,
        frame_index=frame_index,
        is_on_court_player=valid,
        jersey_color=color,
        jersey_number=number,
        confidence=0.9,
    )


def make_qwen_evidence(
    track_id: str,
    observations: Sequence[IdentityObservation],
) -> IdentityEvidence:
    """把测试观察包装成 Qwen 证据。"""
    return IdentityEvidence(
        source="qwen",
        track_id=track_id,
        confidence=0.9,
        observations=tuple(observations),
        frame_indices=tuple(item.frame_index for item in observations),
    )


class QwenTrackObserverTest(unittest.TestCase):
    """验证 Qwen 只输出逐帧证据，不掩盖身份冲突。"""

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
    """验证单视频轨迹取样不会制造重复证据。"""

    def test_short_track_does_not_repeat_last_frame(self) -> None:
        """轨迹短于目标数量时只返回真实存在的不同位置。"""
        positions = _uniform_positions(length=3, count=10)

        self.assertEqual(positions, [0, 1, 2])
        self.assertEqual(len(positions), len(set(positions)))


class FakeEmbeddingExtractor:
    """返回固定向量的测试 ReID 实现。"""

    def extract(self, person_images: Sequence[Any]) -> Sequence[float]:
        """忽略图像内容并返回可验证的向量。"""
        return [3.0, 4.0]


class ReIdTrackEvidenceProviderTest(unittest.TestCase):
    """验证具体 ReID 模型可以通过接口接入证据阶段。"""

    def test_embedding_is_normalized_and_keeps_frame_indices(self) -> None:
        """ReID 输出应归一化，并保留用于提取特征的原始帧号。"""
        provider = ReIdTrackEvidenceProvider(FakeEmbeddingExtractor())
        crops = [
            TrackCrop("player_0", 1, 10, image=object()),
            TrackCrop("player_0", 2, 20, image=object()),
        ]

        evidence = provider.collect(crops)

        self.assertAlmostEqual(math.sqrt(sum(x * x for x in evidence.embedding)), 1.0)
        self.assertEqual(evidence.frame_indices, (10, 20))
        self.assertEqual(evidence.source, "reid")


class IdentityGalleryTest(unittest.TestCase):
    """验证内存人物库具有 PostgreSQL 实现将遵循的检索契约。"""

    def test_attribute_and_embedding_search_share_profile(self) -> None:
        """同一人物档案应同时支持属性检索和 ReID 检索。"""
        gallery = InMemoryIdentityGallery()
        profile = IdentityProfile(
            participant_id="person_13",
            display_name="测试球员",
            jersey_color="white",
            jersey_number="13",
        )
        gallery.save(profile)
        gallery.add_embedding(profile.participant_id, [1.0, 0.0])

        by_attributes = gallery.search_by_attributes("white", "13")
        by_embedding = gallery.search_by_embedding([0.99, 0.01])

        self.assertEqual(by_attributes[0].participant_id, "person_13")
        self.assertEqual(by_embedding[0].participant_id, "person_13")
        self.assertGreater(by_embedding[0].score, 0.9)


class IdentityEvidenceFusionTest(unittest.TestCase):
    """验证 Qwen 仅作为证据，不能单独删除人物轨迹。"""

    def test_stable_qwen_attribute_is_only_provisional(self) -> None:
        """没有人物库支持时，清晰号码也只是临时属性。"""
        evidence = make_qwen_evidence(
            "player_6",
            [
                make_observation("player_6", 10, "white", "13"),
                make_observation("player_6", 20, "white", "13"),
            ],
        )

        result = IdentityEvidenceFusion().resolve("player_6", [evidence])

        self.assertEqual(result.status, "provisional")
        self.assertTrue(result.accepted)
        self.assertEqual(result.jersey_number, "13")
        self.assertIsNone(result.participant_id)

    def test_conflicting_qwen_attributes_are_retained(self) -> None:
        """号码冲突要留下诊断状态，但不能由 Qwen 直接删除整条轨迹。"""
        evidence = make_qwen_evidence(
            "player_8",
            [
                make_observation("player_8", 184, "white", "20"),
                make_observation("player_8", 332, "white", "13"),
            ],
        )

        result = IdentityEvidenceFusion().resolve("player_8", [evidence])

        self.assertEqual(result.status, "conflicting")
        self.assertTrue(result.accepted)
        self.assertEqual(
            {item.jersey_number for item in result.candidates}, {"13", "20"}
        )

    def test_qwen_rejection_falls_back_to_anonymous(self) -> None:
        """Qwen 判断失败时，SAM3 人物候选仍以匿名身份进入事件推理。"""
        evidence = make_qwen_evidence(
            "player_3",
            [make_observation("player_3", 100, None, None, valid=False)],
        )

        result = IdentityEvidenceFusion().resolve("player_3", [evidence])

        self.assertEqual(result.status, "anonymous")
        self.assertTrue(result.accepted)


class CrossClipIdentityAssociatorTest(unittest.TestCase):
    """验证人物库和 ReID 可以在不同片段间复用稳定人物编号。"""

    def test_reid_reference_links_second_clip_to_first_clip(self) -> None:
        """首片段建立匿名档案后，第二片段应通过相似向量找回它。"""
        gallery = InMemoryIdentityGallery()
        fusion = IdentityEvidenceFusion(gallery_threshold=0.8)
        associator = CrossClipIdentityAssociator(gallery)
        first_evidence = IdentityEvidence(
            source="reid",
            track_id="player_0",
            confidence=1.0,
            embedding=(1.0, 0.0),
        )
        first = fusion.resolve("player_0", [first_evidence])
        first = associator.associate("clip_1", first, [first_evidence])

        second_evidence = IdentityEvidence(
            source="reid",
            track_id="player_3",
            confidence=1.0,
            embedding=(0.99, 0.01),
        )
        matches = gallery.search_by_embedding(second_evidence.embedding)
        second = fusion.resolve("player_3", [second_evidence], matches)
        second = associator.associate("clip_2", second, [second_evidence])

        self.assertEqual(first.participant_id, second.participant_id)
        self.assertEqual(second.status, "matched")
        self.assertEqual(len(associator.associations()), 2)


class FakeSampler:
    """避免读取真实视频的 IdentityService 测试采样器。"""

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


class FakeEvidenceProvider:
    """为 Service 测试提供固定 Qwen 证据。"""

    source = "fake"

    def collect(self, crops: Sequence[TrackCrop]) -> IdentityEvidence:
        """返回一条号码可见的属性证据。"""
        track_id = crops[0].track_id
        return make_qwen_evidence(
            track_id,
            [make_observation(track_id, crops[0].frame_index, "black", "17")],
        )


class IdentityServiceTest(unittest.TestCase):
    """验证 Service 只编排组件并返回统一结果。"""

    def test_service_wraps_sampling_evidence_fusion_and_association(self) -> None:
        """单视频流程应得到可进入事件模型的稳定匿名人物编号。"""
        gallery = InMemoryIdentityGallery()
        service = IdentityService(
            sampler=FakeSampler(),
            evidence_providers=[FakeEvidenceProvider()],
            gallery=gallery,
            fusion=IdentityEvidenceFusion(),
            associator=CrossClipIdentityAssociator(gallery),
        )

        result = service.process(
            clip_id="clip_101",
            video_path="unused.mp4",
            annotations={"player_0": {"trajectory": []}},
        )

        self.assertEqual(len(result.decisions), 1)
        self.assertTrue(result.decisions[0].accepted)
        self.assertTrue(result.decisions[0].participant_id.startswith("anonymous_"))
        self.assertEqual(result.decisions[0].jersey_number, "17")


if __name__ == "__main__":
    unittest.main()
