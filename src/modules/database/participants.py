"""使用 SQLite 保存产品中的最小人物档案。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.modules.database.connection import SQLiteDatabase


@dataclass(frozen=True)
class ParticipantRecord:
    """表示产品数据库中的一个人物档案。"""

    participant_id: str
    display_name: str | None = None
    jersey_color: str | None = None
    jersey_number: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SQLiteParticipantRepository:
    """提供人物档案保存和球衣属性精确查询。"""

    def __init__(self, database: SQLiteDatabase) -> None:
        """注入数据库连接管理器。"""
        self.database = database

    @staticmethod
    def _record(row: Mapping[str, Any]) -> ParticipantRecord:
        """把 SQLite 行转换为人物档案。"""
        metadata = json.loads(str(row["metadata_json"]))
        return ParticipantRecord(
            participant_id=str(row["participant_id"]),
            display_name=row["display_name"],
            jersey_color=row["jersey_color"],
            jersey_number=row["jersey_number"],
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def get(self, participant_id: str) -> ParticipantRecord | None:
        """按稳定人物编号读取档案。"""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def save(self, record: ParticipantRecord, replace: bool = False) -> None:
        """保存人物档案；默认拒绝覆盖已经存在的记录。"""
        metadata_json = json.dumps(
            dict(record.metadata), ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        with self.database.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM participants WHERE participant_id = ?",
                (record.participant_id,),
            ).fetchone()
            if exists is not None and not replace:
                raise ValueError(f"人物编号已经存在：{record.participant_id}")
            values = (
                record.display_name,
                record.jersey_color,
                record.jersey_number,
                metadata_json,
                record.participant_id,
            )
            if exists is None:
                connection.execute(
                    """
                    INSERT INTO participants(
                        display_name, jersey_color, jersey_number,
                        metadata_json, participant_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    values,
                )
            else:
                connection.execute(
                    """
                    UPDATE participants
                    SET display_name = ?, jersey_color = ?, jersey_number = ?,
                        metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE participant_id = ?
                    """,
                    values,
                )

    def find_by_jersey(
        self, jersey_color: str, jersey_number: str
    ) -> tuple[ParticipantRecord, ...]:
        """按球衣颜色和号码精确查询人物，不执行相似度猜测。"""
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM participants
                WHERE jersey_color = ? AND jersey_number = ?
                ORDER BY participant_id
                """,
                (jersey_color.strip().lower(), jersey_number.strip()),
            ).fetchall()
        return tuple(self._record(row) for row in rows)
