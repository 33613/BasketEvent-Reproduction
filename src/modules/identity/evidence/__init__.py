"""提供单视频人物证据的统一接口以及 Qwen、ReID 实现。"""

from src.modules.identity.evidence.base import IdentityEvidenceProvider
from src.modules.identity.evidence.qwen import QwenTrackObserver
from src.modules.identity.evidence.reid import (
    PersonEmbeddingExtractor,
    ReIdTrackEvidenceProvider,
)

__all__ = [
    "IdentityEvidenceProvider",
    "PersonEmbeddingExtractor",
    "QwenTrackObserver",
    "ReIdTrackEvidenceProvider",
]
