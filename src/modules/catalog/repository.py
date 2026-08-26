"""定义素材库抽象接口及用于测试的内存实现。"""

from __future__ import annotations

from typing import Protocol

from src.modules.catalog.models import CatalogItem


class MaterialCatalog(Protocol):
    """约束素材登记和检索所需的最小持久化能力。"""

    def add(self, item: CatalogItem, replace: bool = False) -> None:
        """登记素材。"""
        ...

    def get(self, material_id: str) -> CatalogItem | None:
        """按素材编号读取素材。"""
        ...

    def query(
        self,
        event: str | None = None,
        participant_id: str | None = None,
        minimum_confidence: float = 0.0,
    ) -> list[CatalogItem]:
        """按事件、人物和最低置信度检索素材。"""
        ...

    def all_items(self) -> tuple[CatalogItem, ...]:
        """返回全部素材。"""
        ...


class InMemoryMaterialCatalog:
    """提供不依赖数据库的内存素材库，用于单元测试。"""

    def __init__(self) -> None:
        """创建空素材索引。"""
        self._items: dict[str, CatalogItem] = {}

    def add(self, item: CatalogItem, replace: bool = False) -> None:
        """登记素材；默认拒绝覆盖同一素材编号。"""
        if item.material_id in self._items and not replace:
            raise ValueError(f"素材编号已存在：{item.material_id}")
        self._items[item.material_id] = item

    def get(self, material_id: str) -> CatalogItem | None:
        """按素材编号查询单个片段。"""
        return self._items.get(material_id)

    def query(
        self,
        event: str | None = None,
        participant_id: str | None = None,
        minimum_confidence: float = 0.0,
    ) -> list[CatalogItem]:
        """按事件、人物和最低置信度筛选素材。"""
        results: list[CatalogItem] = []
        for item in self._items.values():
            if participant_id is not None and all(
                participant.participant_id != participant_id
                for participant in item.participants
            ):
                continue
            if event is not None and all(
                tag.event != event or tag.confidence < minimum_confidence
                for tag in item.events
            ):
                continue
            results.append(item)
        return sorted(
            results, key=lambda item: (item.source_video_id, item.start_seconds)
        )

    def all_items(self) -> tuple[CatalogItem, ...]:
        """返回当前全部素材的不可变快照。"""
        return tuple(self._items.values())
