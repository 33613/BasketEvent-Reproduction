"""把单片段身份结论关联为跨片段稳定人物编号。"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from src.modules.identity.gallery import IdentityGallery
from src.modules.identity.models import (
    IdentityDecision,
    IdentityEvidence,
    TrackAssociation,
)


class CrossClipIdentityAssociator:
    """维护轨迹与人物档案的关系，并把 ReID 特征补充到人物库。"""

    def __init__(self, gallery: IdentityGallery) -> None:
        """注入人物身份库并创建空关联历史。"""
        self.gallery = gallery
        self._associations: list[TrackAssociation] = []

    def associate(
        self,
        clip_id: str,
        decision: IdentityDecision,
        evidence: Sequence[IdentityEvidence],
    ) -> IdentityDecision:
        """为一条已融合轨迹分配稳定人物编号，并登记外观向量。"""
        participant_id = decision.participant_id
        method = "gallery"
        confidence = 1.0
        if participant_id is None:
            profile = self.gallery.create_anonymous(
                jersey_color=decision.jersey_color,
                jersey_number=decision.jersey_number,
            )
            participant_id = profile.participant_id
            method = "anonymous"
            confidence = 0.0

        for item in evidence:
            if item.embedding is not None:
                self.gallery.add_embedding(participant_id, item.embedding)
                if method == "anonymous":
                    method = "anonymous_with_reid_reference"

        self._associations.append(
            TrackAssociation(
                clip_id=clip_id,
                track_id=decision.track_id,
                participant_id=participant_id,
                method=method,
                confidence=confidence,
            )
        )
        return replace(decision, participant_id=participant_id)

    def associations(self) -> tuple[TrackAssociation, ...]:
        """返回当前全部跨片段轨迹关联的不可变快照。"""
        return tuple(self._associations)

    def to_dict(self) -> dict[str, object]:
        """导出可供未来数据库批量写入的关联记录。"""
        return {
            "schema_version": "basketevent_cross_clip_identity.v2",
            "associations": [
                {
                    "clip_id": item.clip_id,
                    "track_id": item.track_id,
                    "participant_id": item.participant_id,
                    "method": item.method,
                    "confidence": item.confidence,
                }
                for item in self._associations
            ],
        }
