"""提供处理后视频素材的登记、查询和统计接口。"""

from src.modules.catalog.models import CatalogItem, EventTag, ParticipantReference
from src.modules.catalog.repository import InMemoryMaterialCatalog, MaterialCatalog
from src.modules.catalog.service import MaterialCatalogService
from src.modules.catalog.statistics import MaterialStatistics, MaterialStatisticsService

__all__ = [
    "CatalogItem",
    "EventTag",
    "InMemoryMaterialCatalog",
    "MaterialCatalog",
    "MaterialCatalogService",
    "MaterialStatistics",
    "MaterialStatisticsService",
    "ParticipantReference",
]
