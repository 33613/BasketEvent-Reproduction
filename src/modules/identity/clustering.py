"""按球衣身份把不同视频片段中的球员轨迹归并为素材人物组。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from src.modules.identity.models import IdentityCluster, ResolvedIdentity


class CrossClipIdentityClusterer:
    """提供无需名单的跨片段基础聚类，并为以后接入 ReID 保留接口。"""

    def __init__(self) -> None:
        """创建空的稳定身份索引。"""
        self._clusters: OrderedDict[tuple[str, str], IdentityCluster] = OrderedDict()

    def add_clip(
        self, clip_id: str, identities: Iterable[ResolvedIdentity]
    ) -> dict[str, str]:
        """把稳定的颜色与号码身份加入聚类并返回轨迹到聚类的映射。"""
        assignments: dict[str, str] = {}
        for identity in identities:
            if (
                identity.status != "stable"
                or identity.jersey_color is None
                or identity.jersey_number is None
            ):
                continue
            key = (identity.jersey_color, identity.jersey_number)
            cluster = self._clusters.get(key)
            if cluster is None:
                cluster = IdentityCluster(
                    cluster_id=f"identity_{len(self._clusters):04d}",
                    jersey_color=identity.jersey_color,
                    jersey_number=identity.jersey_number,
                )
                self._clusters[key] = cluster
            cluster.members.append((clip_id, identity.track_id))
            assignments[identity.track_id] = cluster.cluster_id
        return assignments

    def clusters(self) -> tuple[IdentityCluster, ...]:
        """返回当前全部人物组的不可变快照。"""
        return tuple(self._clusters.values())

    def to_dict(self) -> dict[str, object]:
        """导出可用于数据库写入或素材统计的 JSON 字典。"""
        return {
            "schema_version": "basketevent_cross_clip_identity.v1",
            "clusters": [cluster.to_dict() for cluster in self.clusters()],
        }
