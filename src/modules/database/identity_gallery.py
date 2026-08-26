"""使用 SQLite 持久化人物档案和 ReID 参考向量。"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any, Mapping, Sequence

from src.modules.database.connection import SQLiteDatabase
from src.modules.identity.models import GalleryMatch, IdentityProfile


def _json_text(value: Mapping[str, Any]) -> str:
    """把可审计元数据编码为稳定 JSON 文本。"""
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


class SQLiteIdentityGallery:
    """实现可跨进程复用的 SQLite 人物身份库。"""

    def __init__(self, database: SQLiteDatabase) -> None:
        """注入已初始化的数据库连接管理器。"""
        self.database = database

    @staticmethod
    def _normalize(embedding: Sequence[float]) -> tuple[float, ...]:
        """校验并归一化 ReID 向量。"""
        values = tuple(float(value) for value in embedding)
        if not values:
            raise ValueError("人物特征向量不能为空")
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0:
            raise ValueError("人物特征向量不能是零向量")
        return tuple(value / norm for value in values)

    @staticmethod
    def _profile(row: Mapping[str, Any]) -> IdentityProfile:
        """把 SQLite 行转换为领域人物档案。"""
        metadata = json.loads(str(row["metadata_json"]))
        return IdentityProfile(
            participant_id=str(row["participant_id"]),
            display_name=row["display_name"],
            jersey_color=row["jersey_color"],
            jersey_number=row["jersey_number"],
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def get(self, participant_id: str) -> IdentityProfile | None:
        """按稳定人物编号读取档案。"""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
        return self._profile(row) if row is not None else None

    def save(self, profile: IdentityProfile, replace: bool = False) -> None:
        """保存人物档案；默认拒绝覆盖已经存在的人物。"""
        with self.database.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id = ?",
                (profile.participant_id,),
            ).fetchone()
            if exists is not None and not replace:
                raise ValueError(f"人物编号已经存在：{profile.participant_id}")
            values = (
                profile.display_name,
                profile.jersey_color,
                profile.jersey_number,
                _json_text(profile.metadata),
                profile.participant_id,
            )
            if exists is not None:
                connection.execute(
                    """
                    UPDATE participants
                    SET display_name = ?, jersey_color = ?, jersey_number = ?,
                        metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE participant_id = ?
                    """,
                    values,
                )
            else:
                connection.execute(
                    """
                    INSERT INTO participants(
                        display_name, jersey_color, jersey_number,
                        metadata_json, participant_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    values,
                )

    def create_anonymous(
        self,
        jersey_color: str | None = None,
        jersey_number: str | None = None,
    ) -> IdentityProfile:
        """创建不会因程序重启而重复的匿名人物编号。"""
        profile = IdentityProfile(
            participant_id=f"anonymous_{uuid.uuid4().hex}",
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
        """按颜色和号码精确检索；字段不完整时不猜测。"""
        if jersey_color is None or jersey_number is None:
            return ()
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM participants
                WHERE jersey_color = ? AND jersey_number = ?
                ORDER BY participant_id
                """,
                (jersey_color, jersey_number),
            ).fetchall()
        return tuple(
            GalleryMatch(
                participant_id=str(row["participant_id"]),
                score=1.0,
                method="attributes",
                profile=self._profile(row),
            )
            for row in rows
        )

    def search_by_embedding(
        self,
        embedding: Sequence[float],
        limit: int = 5,
    ) -> tuple[GalleryMatch, ...]:
        """在 Python 中计算余弦相似度，作为 SQLite 阶段的可靠基线。"""
        if limit <= 0:
            raise ValueError("limit 必须为正整数")
        query = self._normalize(embedding)
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT e.participant_id, e.dimension, e.embedding_json,
                       p.display_name, p.jersey_color, p.jersey_number,
                       p.metadata_json
                FROM participant_embeddings AS e
                JOIN participants AS p
                  ON p.participant_id = e.participant_id
                WHERE e.dimension = ?
                """,
                (len(query),),
            ).fetchall()

        best: dict[str, tuple[float, IdentityProfile]] = {}
        for row in rows:
            reference = tuple(
                float(value) for value in json.loads(row["embedding_json"])
            )
            score = float(sum(left * right for left, right in zip(query, reference)))
            participant_id = str(row["participant_id"])
            previous = best.get(participant_id)
            if previous is None or score > previous[0]:
                best[participant_id] = (score, self._profile(row))
        ordered = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        return tuple(
            GalleryMatch(
                participant_id=participant_id,
                score=score,
                method="reid",
                profile=profile,
            )
            for participant_id, (score, profile) in ordered[:limit]
        )

    def add_embedding(
        self,
        participant_id: str,
        embedding: Sequence[float],
        *,
        model_name: str | None = None,
        source_track_id: str | None = None,
        quality_score: float | None = None,
    ) -> None:
        """向人物档案追加一条带来源信息的归一化 ReID 向量。"""
        normalized = self._normalize(embedding)
        with self.database.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"人物编号不存在：{participant_id}")
            connection.execute(
                """
                INSERT INTO participant_embeddings(
                    participant_id, dimension, embedding_json, model_name,
                    source_track_id, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    participant_id,
                    len(normalized),
                    json.dumps(normalized, allow_nan=False),
                    model_name,
                    source_track_id,
                    quality_score,
                ),
            )
