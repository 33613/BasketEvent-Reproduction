"""提供处理后视频素材的登记、查询、统计和人物重识别接口。"""

from src.modules.catalog.models import CatalogItem, EventTag, ParticipantReference
from src.modules.catalog.reid import CosineReIdMatcher, PersonEmbeddingExtractor
from src.modules.catalog.service import InMemoryMaterialCatalog, MaterialCatalogService
from src.modules.catalog.statistics import MaterialStatistics, MaterialStatisticsService

__all__ = [
    "CatalogItem",
    "CosineReIdMatcher",
    "EventTag",
    "InMemoryMaterialCatalog",
    "MaterialCatalogService",
    "MaterialStatistics",
    "MaterialStatisticsService",
    "ParticipantReference",
    "PersonEmbeddingExtractor",
]
