"""定义单视频轨迹证据提供器的接口。"""

from __future__ import annotations

from typing import Protocol, Sequence

from src.modules.identity.models import IdentityEvidence, TrackCrop


class IdentityEvidenceProvider(Protocol):
    """约束 Qwen、ReID 等实现如何向 IdentityService 提供证据。"""

    source: str

    def collect(self, crops: Sequence[TrackCrop]) -> IdentityEvidence:
        """根据同一条轨迹的截图生成一组结构化证据。"""
        ...
