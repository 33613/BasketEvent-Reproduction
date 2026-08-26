"""定义人物身份库接口以及不依赖数据库的内存实现。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.modules.identity.models import GalleryMatch, IdentityProfile


class IdentityGallery(Protocol):
    """约束人物档案、属性检索和向量检索所需的最小数据库能力。"""

    def get(self, participant_id: str) -> IdentityProfile | None:
        """按稳定人物编号读取档案。"""
        ...

    def save(self, profile: IdentityProfile, replace: bool = False) -> None:
        """保存人物档案。"""
        ...

    def create_anonymous(
        self,
        jersey_color: str | None = None,
        jersey_number: str | None = None,
    ) -> IdentityProfile:
        """创建尚未命名的稳定人物档案。"""
        ...

    def search_by_attributes(
        self,
        jersey_color: str | None,
        jersey_number: str | None,
    ) -> tuple[GalleryMatch, ...]:
        """按球衣属性返回所有精确候选。"""
        ...

    def search_by_embedding(
        self,
        embedding: Sequence[float],
        limit: int = 5,
    ) -> tuple[GalleryMatch, ...]:
        """按人物 ReID 向量返回最相似候选。"""
        ...

    def add_embedding(
        self,
        participant_id: str,
        embedding: Sequence[float],
    ) -> None:
        """向人物档案增加一条参考向量。"""
        ...


class InMemoryIdentityGallery:
    """使用 Python 容器实现人物库，供测试和数据库接入前使用。"""

    def __init__(self) -> None:
        """创建空人物档案和特征索引。"""
        self._profiles: dict[str, IdentityProfile] = {}
        self._embeddings: dict[str, list[tuple[float, ...]]] = {}
        self._anonymous_counter = 0

    @classmethod
    def from_roster_file(cls, path: str | Path | None) -> "InMemoryIdentityGallery":
        """把可选比赛名单加载为已知人物档案，不把名单写死在融合器中。"""
        gallery = cls()
        if path is None:
            return gallery
        source = Path(path)
        with source.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, Mapping):
            raise ValueError(f"名单 JSON 顶层必须是对象：{source}")
        colors = {
            str(team): str(color).strip().lower()
            for team, color in document.get("jersey_color", {}).items()
        }
        for index, player in enumerate(document.get("players", [])):
            if not isinstance(player, Mapping):
                continue
            color = colors.get(str(player.get("team_name")))
            number = str(player.get("jersey", "")).strip() or None
            name = str(player.get("name", "")).strip() or None
            if color is None and number is None and name is None:
                continue
            gallery.save(
                IdentityProfile(
                    participant_id=f"roster_{index:04d}",
                    display_name=name,
                    jersey_color=color,
                    jersey_number=number,
                    metadata={"source": "roster"},
                )
            )
        return gallery

    @staticmethod
    def _normalize(embedding: Sequence[float]) -> tuple[float, ...]:
        """将向量转换为可计算余弦相似度的单位向量。"""
        values = tuple(float(value) for value in embedding)
        if not values:
            raise ValueError("人物特征向量不能为空")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("人物特征向量不能是零向量")
        return tuple(value / norm for value in values)

    def get(self, participant_id: str) -> IdentityProfile | None:
        """按稳定人物编号读取档案。"""
        return self._profiles.get(participant_id)

    def save(self, profile: IdentityProfile, replace: bool = False) -> None:
        """保存人物档案，并默认阻止无意覆盖。"""
        if profile.participant_id in self._profiles and not replace:
            raise ValueError(f"人物编号已经存在：{profile.participant_id}")
        self._profiles[profile.participant_id] = profile

    def create_anonymous(
        self,
        jersey_color: str | None = None,
        jersey_number: str | None = None,
    ) -> IdentityProfile:
        """创建匿名人物；身份信息以后可以在数据库中继续补全。"""
        while True:
            participant_id = f"anonymous_{self._anonymous_counter:06d}"
            self._anonymous_counter += 1
            if participant_id not in self._profiles:
                break
        profile = IdentityProfile(
            participant_id=participant_id,
            jersey_color=jersey_color,
            jersey_number=jersey_number,
            metadata={"source": "automatic"},
        )
        self.save(profile)
        return profile

    def search_by_attributes(
        self,
        jersey_color: str | None,
        jersey_number: str | None,
    ) -> tuple[GalleryMatch, ...]:
        """按球衣颜色与号码精确检索；缺少任一字段时不猜测。"""
        if jersey_color is None or jersey_number is None:
            return ()
        matches = [
            GalleryMatch(
                participant_id=profile.participant_id,
                score=1.0,
                method="attributes",
                profile=profile,
            )
            for profile in self._profiles.values()
            if profile.jersey_color == jersey_color
            and profile.jersey_number == jersey_number
        ]
        return tuple(matches)

    def search_by_embedding(
        self,
        embedding: Sequence[float],
        limit: int = 5,
    ) -> tuple[GalleryMatch, ...]:
        """以内存中的参考向量执行余弦相似度检索。"""
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        query = self._normalize(embedding)
        matches: list[GalleryMatch] = []
        for participant_id, references in self._embeddings.items():
            compatible = [item for item in references if len(item) == len(query)]
            if not compatible:
                continue
            score = max(sum(a * b for a, b in zip(query, item)) for item in compatible)
            profile = self._profiles[participant_id]
            matches.append(
                GalleryMatch(
                    participant_id=participant_id,
                    score=float(score),
                    method="reid",
                    profile=profile,
                )
            )
        matches.sort(key=lambda item: item.score, reverse=True)
        return tuple(matches[:limit])

    def add_embedding(
        self,
        participant_id: str,
        embedding: Sequence[float],
    ) -> None:
        """为已经存在的人物档案保存一条归一化参考向量。"""
        if participant_id not in self._profiles:
            raise KeyError(f"人物编号不存在：{participant_id}")
        self._embeddings.setdefault(participant_id, []).append(
            self._normalize(embedding)
        )

    def profiles(self) -> tuple[IdentityProfile, ...]:
        """返回当前全部人物档案的不可变快照。"""
        return tuple(self._profiles.values())
