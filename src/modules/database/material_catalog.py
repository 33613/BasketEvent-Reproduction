"""使用 SQLite 持久化可检索的视频素材。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from src.modules.catalog.models import CatalogItem, EventTag, ParticipantReference
from src.modules.database.connection import SQLiteDatabase


class SQLiteMaterialCatalog:
    """实现跨进程持久化的素材登记和组合条件检索。"""

    def __init__(self, database: SQLiteDatabase) -> None:
        """注入已初始化的产品数据库。"""
        self.database = database

    @staticmethod
    def _metadata_text(metadata: Mapping[str, Any]) -> str:
        """把素材元数据编码为稳定 JSON 文本。"""
        return json.dumps(
            dict(metadata), ensure_ascii=False, sort_keys=True, allow_nan=False
        )

    @staticmethod
    def _load_item(connection: sqlite3.Connection, row: sqlite3.Row) -> CatalogItem:
        """从素材主表和两张关系表恢复完整领域对象。"""
        material_id = str(row["material_id"])
        event_rows = connection.execute(
            """
            SELECT * FROM material_events
            WHERE material_id = ? ORDER BY event_id
            """,
            (material_id,),
        ).fetchall()
        participant_rows = connection.execute(
            """
            SELECT * FROM material_participants
            WHERE material_id = ? ORDER BY participant_id, track_id
            """,
            (material_id,),
        ).fetchall()
        metadata = json.loads(str(row["metadata_json"]))
        return CatalogItem(
            material_id=material_id,
            source_video_id=str(row["source_video_id"]),
            segment_id=str(row["segment_id"]),
            video_path=Path(str(row["video_path"])),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
            processing_status=str(row["processing_status"]),
            events=tuple(
                EventTag(
                    event=str(value["event_name"]),
                    confidence=float(value["confidence"]),
                    player_id=value["player_id"],
                    start_seconds=value["start_seconds"],
                    end_seconds=value["end_seconds"],
                )
                for value in event_rows
            ),
            participants=tuple(
                ParticipantReference(
                    participant_id=str(value["participant_id"]),
                    track_id=str(value["track_id"]),
                    jersey_color=value["jersey_color"],
                    jersey_number=value["jersey_number"],
                    player_name=value["player_name"],
                    identity_status=value["identity_status"],
                    reid_cluster_id=value["reid_cluster_id"],
                )
                for value in participant_rows
            ),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def add(self, item: CatalogItem, replace: bool = False) -> None:
        """在一个事务中写入素材、事件和人物关系。"""
        if item.end_seconds < item.start_seconds:
            raise ValueError("素材结束时间不能小于开始时间")
        try:
            with self.database.transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM materials WHERE material_id = ?",
                    (item.material_id,),
                ).fetchone()
                if exists is not None and not replace:
                    raise ValueError(f"素材编号已存在：{item.material_id}")
                if exists is not None:
                    connection.execute(
                        "DELETE FROM materials WHERE material_id = ?",
                        (item.material_id,),
                    )
                connection.execute(
                    """
                    INSERT INTO materials(
                        material_id, source_video_id, segment_id, video_path,
                        start_seconds, end_seconds, processing_status,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.material_id,
                        item.source_video_id,
                        item.segment_id,
                        str(item.video_path),
                        item.start_seconds,
                        item.end_seconds,
                        item.processing_status,
                        self._metadata_text(item.metadata),
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO material_events(
                        material_id, event_name, confidence, player_id,
                        start_seconds, end_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.material_id,
                            event.event,
                            event.confidence,
                            event.player_id,
                            event.start_seconds,
                            event.end_seconds,
                        )
                        for event in item.events
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO material_participants(
                        material_id, participant_id, track_id, jersey_color,
                        jersey_number, player_name, identity_status,
                        reid_cluster_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.material_id,
                            participant.participant_id,
                            participant.track_id,
                            participant.jersey_color,
                            participant.jersey_number,
                            participant.player_name,
                            participant.identity_status,
                            participant.reid_cluster_id,
                        )
                        for participant in item.participants
                    ],
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"素材记录违反数据库约束：{error}") from error

    def get(self, material_id: str) -> CatalogItem | None:
        """按素材编号读取完整素材。"""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM materials WHERE material_id = ?",
                (material_id,),
            ).fetchone()
            return self._load_item(connection, row) if row is not None else None

    def query(
        self,
        event: str | None = None,
        participant_id: str | None = None,
        minimum_confidence: float = 0.0,
    ) -> list[CatalogItem]:
        """使用 SQL 按事件、人物和置信度组合检索素材。"""
        clauses: list[str] = []
        parameters: list[Any] = []
        if event is not None:
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM material_events AS e
                    WHERE e.material_id = m.material_id
                      AND e.event_name = ? AND e.confidence >= ?
                )
                """
            )
            parameters.extend((event, float(minimum_confidence)))
        if participant_id is not None:
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM material_participants AS p
                    WHERE p.material_id = m.material_id
                      AND p.participant_id = ?
                )
                """
            )
            parameters.append(participant_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        statement = f"""
            SELECT m.* FROM materials AS m
            {where}
            ORDER BY m.source_video_id, m.start_seconds
        """
        with self.database.transaction() as connection:
            rows = connection.execute(statement, parameters).fetchall()
            return [self._load_item(connection, row) for row in rows]

    def all_items(self) -> tuple[CatalogItem, ...]:
        """按来源视频和时间顺序返回全部素材。"""
        return tuple(self.query())
