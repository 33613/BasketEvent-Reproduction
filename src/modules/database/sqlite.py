"""用一个 SQLite 文件保存产品人物和视频素材。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from src.core.config import SETTINGS
from src.modules.catalog import CatalogItem, EventTag, ParticipantReference
from src.modules.database.schema import SCHEMA_SQL, SCHEMA_VERSION


@dataclass(frozen=True)
class ParticipantRecord:
    """表示人物表中的一条最小档案。"""

    participant_id: str
    display_name: str | None = None
    jersey_color: str | None = None
    jersey_number: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductStorageLayout:
    """集中给出数据库和媒体文件的产品目录。"""

    root: Path

    @property
    def database_path(self) -> Path:
        """返回 SQLite 文件路径。"""
        return self.root / "database" / "basketevent.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        """返回用户上传视频目录。"""
        return self.root / "media" / "uploads"

    @property
    def segments_dir(self) -> Path:
        """返回切分后视频目录。"""
        return self.root / "media" / "segments"

    @property
    def visualizations_dir(self) -> Path:
        """返回可视化结果目录。"""
        return self.root / "media" / "visualizations"

    def initialize(self) -> None:
        """创建产品运行需要的目录。"""
        for path in (
            self.database_path.parent,
            self.uploads_dir,
            self.segments_dir,
            self.visualizations_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class ProductDatabase:
    """直接提供当前产品需要的 SQLite 保存和查询操作。"""

    def __init__(self, storage: ProductStorageLayout) -> None:
        """保存目录配置；请使用 ``open`` 创建已初始化对象。"""
        self.storage = storage
        self.path = storage.database_path

    @classmethod
    def open(cls, root: str | Path | None = None) -> "ProductDatabase":
        """创建产品目录、初始化表结构并返回数据库对象。"""
        storage = ProductStorageLayout(
            Path(root).expanduser() if root is not None else SETTINGS.product_data_root
        )
        storage.initialize()
        database = cls(storage)
        database._initialize_schema()
        return database

    def _connect(self) -> sqlite3.Connection:
        """创建启用外键和行字段访问的短连接。"""
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """在成功时提交事务，在异常时回滚。"""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        """创建表结构，并拒绝代码无法读取的更高版本数据库。"""
        with self._transaction() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA_SQL)
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_versions"
            ).fetchone()
            current = int(row["version"] or 0)
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"数据库版本 {current} 高于代码支持的版本 {SCHEMA_VERSION}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO schema_versions(version) VALUES (?)",
                (SCHEMA_VERSION,),
            )

    @staticmethod
    def _json_text(value: Mapping[str, Any]) -> str:
        """把元数据编码为稳定的 JSON 文本。"""
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False
        )

    @staticmethod
    def _participant_from_row(row: Mapping[str, Any]) -> ParticipantRecord:
        """把数据库行转换为人物档案。"""
        metadata = json.loads(str(row["metadata_json"]))
        return ParticipantRecord(
            participant_id=str(row["participant_id"]),
            display_name=row["display_name"],
            jersey_color=row["jersey_color"],
            jersey_number=row["jersey_number"],
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    @staticmethod
    def _material_from_row(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> CatalogItem:
        """读取素材主记录及其事件、人物关系。"""
        material_id = str(row["material_id"])
        event_rows = connection.execute(
            "SELECT * FROM material_events WHERE material_id = ? ORDER BY event_id",
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
                )
                for value in participant_rows
            ),
            metadata=metadata if isinstance(metadata, Mapping) else {},
        )

    def save_participant(
        self, record: ParticipantRecord, replace: bool = False
    ) -> None:
        """保存人物档案；默认拒绝覆盖同一人物编号。"""
        with self._transaction() as connection:
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
                self._json_text(record.metadata),
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

    def get_participant(self, participant_id: str) -> ParticipantRecord | None:
        """按人物编号读取档案。"""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM participants WHERE participant_id = ?",
                (participant_id,),
            ).fetchone()
        return self._participant_from_row(row) if row is not None else None

    def find_participants_by_jersey(
        self, jersey_color: str, jersey_number: str
    ) -> tuple[ParticipantRecord, ...]:
        """按球衣颜色和号码精确查询人物。"""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM participants
                WHERE jersey_color = ? AND jersey_number = ?
                ORDER BY participant_id
                """,
                (jersey_color.strip().lower(), jersey_number.strip()),
            ).fetchall()
        return tuple(self._participant_from_row(row) for row in rows)

    def save_material(self, item: CatalogItem, replace: bool = False) -> None:
        """在一个事务中保存素材及其事件、人物关系。"""
        if item.end_seconds < item.start_seconds:
            raise ValueError("素材结束时间不能小于开始时间")
        try:
            with self._transaction() as connection:
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
                        self._json_text(item.metadata),
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
                        jersey_number, player_name, identity_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
                        )
                        for participant in item.participants
                    ],
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"素材记录违反数据库约束：{error}") from error

    def get_material(self, material_id: str) -> CatalogItem | None:
        """按素材编号读取完整素材。"""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM materials WHERE material_id = ?",
                (material_id,),
            ).fetchone()
            return self._material_from_row(connection, row) if row is not None else None

    def find_materials(
        self,
        event: str | None = None,
        participant_id: str | None = None,
        minimum_confidence: float = 0.0,
    ) -> list[CatalogItem]:
        """按事件、人物和最低置信度组合查询素材。"""
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
        with self._transaction() as connection:
            rows = connection.execute(statement, parameters).fetchall()
            return [self._material_from_row(connection, row) for row in rows]

    def list_materials(self) -> tuple[CatalogItem, ...]:
        """按来源视频和时间顺序返回全部素材。"""
        return tuple(self.find_materials())

    def schema_version(self) -> int:
        """返回当前数据库结构版本。"""
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_versions"
            ).fetchone()
        return int(row["version"] or 0)

    def status(self) -> dict[str, object]:
        """返回数据库路径和各表记录数量。"""
        with self._transaction() as connection:
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in (
                    "participants",
                    "materials",
                    "material_events",
                    "material_participants",
                )
            }
        return {
            "schema_version": self.schema_version(),
            "database_path": str(self.path),
            "storage_root": str(self.storage.root),
            "counts": counts,
        }
