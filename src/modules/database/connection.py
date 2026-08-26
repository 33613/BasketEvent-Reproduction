"""封装 SQLite 连接、事务和数据库初始化。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.modules.database.schema import SCHEMA_SQL, SCHEMA_VERSION


class SQLiteDatabase:
    """管理单个产品 SQLite 文件，不向业务层暴露连接细节。"""

    def __init__(self, path: str | Path) -> None:
        """保存数据库路径；文件和父目录由 ``initialize`` 创建。"""
        self.path = Path(path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        """创建启用外键、WAL 和行对象的数据库连接。"""
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        """创建目录和当前版本表结构，并拒绝未知的新版本数据库。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
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
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """提供自动提交或回滚的短事务连接。"""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def schema_version(self) -> int:
        """返回当前数据库结构版本。"""
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT MAX(version) AS version FROM schema_versions"
            ).fetchone()
        return int(row["version"] or 0)
