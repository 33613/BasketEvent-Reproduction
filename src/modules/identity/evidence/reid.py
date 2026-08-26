"""定义人物 ReID 特征提取接口和单轨迹证据适配器。"""

from __future__ import annotations

import math
from typing import Any, Protocol, Sequence

from src.modules.identity.models import IdentityEvidence, TrackCrop


class PersonEmbeddingExtractor(Protocol):
    """约束具体人物 ReID 模型必须提供的最小能力。"""

    def extract(self, person_images: Sequence[Any]) -> Sequence[float]:
        """把同一轨迹的多张人物截图编码成一个固定长度向量。"""
        ...


class ReIdTrackEvidenceProvider:
    """把任意符合接口的 ReID 模型包装成单视频证据提供器。"""

    source = "reid"

    def __init__(self, extractor: PersonEmbeddingExtractor) -> None:
        """注入具体特征提取器，避免业务层绑定某个 ReID 框架。"""
        self.extractor = extractor

    def collect(self, crops: Sequence[TrackCrop]) -> IdentityEvidence:
        """提取并归一化一条轨迹的外观特征。"""
        if not crops:
            raise ValueError("ReID 至少需要一张轨迹截图")
        raw = self.extractor.extract([crop.image for crop in crops])
        values = [float(value) for value in raw]
        if not values:
            raise ValueError("ReID 特征向量不能为空")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("ReID 特征向量不能是零向量")
        embedding = tuple(value / norm for value in values)
        return IdentityEvidence(
            source=self.source,
            track_id=crops[0].track_id,
            confidence=1.0,
            embedding=embedding,
            frame_indices=tuple(crop.frame_index for crop in crops),
            metadata={"crop_count": len(crops)},
        )
