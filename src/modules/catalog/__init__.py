"""提供处理后视频素材的数据结构、整理和统计功能。"""

from src.modules.catalog.models import (
    CatalogItem,
    EventTag,
    MaterialStatistics,
    ParticipantReference,
)
from src.modules.catalog.service import CatalogService

__all__ = [
    "CatalogItem",
    "EventTag",
    "CatalogService",
    "MaterialStatistics",
    "ParticipantReference",
]
