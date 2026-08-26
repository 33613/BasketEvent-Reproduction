"""综合 Qwen、ReID 和人物库候选，生成不阻塞下游的身份结论。"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from src.modules.identity.models import (
    GalleryMatch,
    IdentityCandidate,
    IdentityDecision,
    IdentityEvidence,
    IdentityObservation,
)


class IdentityEvidenceFusion:
    """把不同来源的证据融合为匹配、临时属性、冲突或匿名状态。"""

    def __init__(
        self,
        gallery_threshold: float = 0.75,
        minimum_score_gap: float = 0.05,
    ) -> None:
        """设置人物库接受阈值和第一、第二候选的最小差值。"""
        if not 0.0 <= gallery_threshold <= 1.0:
            raise ValueError("gallery_threshold 必须位于 [0, 1]")
        if minimum_score_gap < 0:
            raise ValueError("minimum_score_gap 不能为负数")
        self.gallery_threshold = gallery_threshold
        self.minimum_score_gap = minimum_score_gap

    @staticmethod
    def _attribute_candidates(
        observations: Sequence[IdentityObservation],
    ) -> tuple[IdentityCandidate, ...]:
        """按颜色和号码聚合逐帧观察，但不使用多数票消除冲突。"""
        grouped: dict[tuple[str, str], list[IdentityObservation]] = defaultdict(list)
        for item in observations:
            if (
                item.is_on_court_player is True
                and item.jersey_color is not None
                and item.jersey_number is not None
            ):
                grouped[(item.jersey_color, item.jersey_number)].append(item)
        candidates = [
            IdentityCandidate(
                jersey_color=color,
                jersey_number=number,
                support_count=len(items),
                confidence_sum=sum(item.confidence for item in items),
                frames=tuple(item.frame_index for item in items),
            )
            for (color, number), items in grouped.items()
        ]
        candidates.sort(
            key=lambda item: (item.support_count, item.confidence_sum), reverse=True
        )
        return tuple(candidates)

    def _accepted_gallery_match(
        self,
        matches: Sequence[GalleryMatch],
    ) -> GalleryMatch | None:
        """仅在候选足够可信且没有近似并列时接受人物库结果。"""
        if not matches:
            return None
        ordered = sorted(matches, key=lambda item: item.score, reverse=True)
        best = ordered[0]
        if best.score < self.gallery_threshold:
            return None
        if len(ordered) > 1 and best.score - ordered[1].score < self.minimum_score_gap:
            return None
        return best

    def resolve(
        self,
        track_id: str,
        evidence: Sequence[IdentityEvidence],
        gallery_matches: Sequence[GalleryMatch] = (),
    ) -> IdentityDecision:
        """融合一条轨迹的全部证据；Qwen 单独失败时仍保留匿名轨迹。"""
        observations = tuple(
            observation for item in evidence for observation in item.observations
        )
        candidates = self._attribute_candidates(observations)
        sources = tuple(dict.fromkeys(item.source for item in evidence))
        accepted_match = self._accepted_gallery_match(gallery_matches)
        if accepted_match is not None:
            profile = accepted_match.profile
            return IdentityDecision(
                track_id=track_id,
                status="matched",
                accepted=True,
                is_on_court_player=True,
                participant_id=profile.participant_id,
                jersey_color=profile.jersey_color,
                jersey_number=profile.jersey_number,
                player_name=profile.display_name,
                candidates=candidates,
                gallery_matches=tuple(gallery_matches),
                evidence_sources=sources,
                reason=f"人物库通过 {accepted_match.method} 返回唯一可信候选",
            )
        if len(candidates) == 1:
            candidate = candidates[0]
            return IdentityDecision(
                track_id=track_id,
                status="provisional",
                accepted=True,
                is_on_court_player=True,
                jersey_color=candidate.jersey_color,
                jersey_number=candidate.jersey_number,
                candidates=candidates,
                gallery_matches=tuple(gallery_matches),
                evidence_sources=sources,
                reason="只有属性证据，暂不把 Qwen 观察当作最终人物身份",
            )
        if len(candidates) > 1:
            return IdentityDecision(
                track_id=track_id,
                status="conflicting",
                accepted=True,
                is_on_court_player=True,
                candidates=candidates,
                gallery_matches=tuple(gallery_matches),
                evidence_sources=sources,
                reason="同一轨迹存在多个属性候选，保留轨迹并等待拆分或更多证据",
            )

        visible_colors = {
            item.jersey_color
            for item in observations
            if item.is_on_court_player is True and item.jersey_color is not None
        }
        return IdentityDecision(
            track_id=track_id,
            status="anonymous",
            accepted=True,
            is_on_court_player=True,
            jersey_color=(
                next(iter(visible_colors)) if len(visible_colors) == 1 else None
            ),
            gallery_matches=tuple(gallery_matches),
            evidence_sources=sources,
            reason="身份数据不足；Qwen 不能单独删除 SAM3 人物轨迹",
        )
