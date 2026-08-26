"""定义人物重识别特征接口和一个余弦相似度聚类基线。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class PersonEmbeddingExtractor(Protocol):
    """约束人物 ReID 模型适配器必须提供的特征提取接口。"""

    def extract(self, person_images: Sequence[Any]) -> Sequence[float]:
        """把同一人物的多张截图编码为一个固定长度向量。"""
        ...


@dataclass
class _ReIdCluster:
    """保存一个人物簇的归一化中心及累计样本数。"""

    cluster_id: str
    centroid: list[float]
    sample_count: int
    identity_hint: str | None


class CosineReIdMatcher:
    """使用余弦相似度把跨片段人物特征归并为稳定人物簇。"""

    def __init__(self, similarity_threshold: float = 0.75) -> None:
        """设置匹配阈值并创建空人物簇。"""
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold 必须位于 [-1, 1]")
        self.similarity_threshold = float(similarity_threshold)
        self._clusters: list[_ReIdCluster] = []

    @staticmethod
    def _normalize(embedding: Sequence[float]) -> list[float]:
        """校验向量并进行 L2 归一化。"""
        values = [float(value) for value in embedding]
        if not values:
            raise ValueError("ReID 特征向量不能为空")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("ReID 特征向量不能是零向量")
        return [value / norm for value in values]

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        """计算两个已归一化向量的余弦相似度。"""
        if len(left) != len(right):
            raise ValueError("ReID 特征维度不一致")
        return float(sum(a * b for a, b in zip(left, right)))

    def assign(
        self,
        embedding: Sequence[float],
        identity_hint: str | None = None,
    ) -> tuple[str, float]:
        """匹配已有簇或创建新簇，并返回簇编号与相似度。"""
        normalized = self._normalize(embedding)
        best_cluster: _ReIdCluster | None = None
        best_similarity = -1.0
        for cluster in self._clusters:
            if len(cluster.centroid) != len(normalized):
                continue
            if (
                identity_hint is not None
                and cluster.identity_hint is not None
                and identity_hint != cluster.identity_hint
            ):
                continue
            similarity = self._cosine(normalized, cluster.centroid)
            if similarity > best_similarity:
                best_cluster = cluster
                best_similarity = similarity

        if best_cluster is None or best_similarity < self.similarity_threshold:
            cluster = _ReIdCluster(
                cluster_id=f"reid_{len(self._clusters):05d}",
                centroid=list(normalized),
                sample_count=1,
                identity_hint=identity_hint,
            )
            self._clusters.append(cluster)
            return cluster.cluster_id, 1.0

        count = best_cluster.sample_count
        averaged = [
            (old * count + new) / (count + 1)
            for old, new in zip(best_cluster.centroid, normalized)
        ]
        best_cluster.centroid = self._normalize(averaged)
        best_cluster.sample_count += 1
        return best_cluster.cluster_id, best_similarity

    def clusters(self) -> tuple[dict[str, object], ...]:
        """返回不暴露内部可变对象的人物簇快照。"""
        return tuple(
            {
                "cluster_id": cluster.cluster_id,
                "sample_count": cluster.sample_count,
                "identity_hint": cluster.identity_hint,
            }
            for cluster in self._clusters
        )
