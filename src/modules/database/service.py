"""提供产品存储目录和数据库仓库的统一构造入口。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.core.config import SETTINGS
from src.modules.database.connection import SQLiteDatabase
from src.modules.database.identity_gallery import SQLiteIdentityGallery
from src.modules.database.material_catalog import SQLiteMaterialCatalog


@dataclass(frozen=True)
class ProductStorageLayout:
    """集中描述产品运行数据目录，避免业务代码拼接路径。"""

    root: Path

    @property
    def database_dir(self) -> Path:
        """返回数据库文件目录。"""
        return self.root / "database"

    @property
    def database_path(self) -> Path:
        """返回默认 SQLite 文件路径。"""
        return self.database_dir / "basketevent.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        """返回用户上传原始视频目录。"""
        return self.root / "media" / "uploads"

    @property
    def segments_dir(self) -> Path:
        """返回模型处理用短片目录。"""
        return self.root / "media" / "segments"

    @property
    def visualizations_dir(self) -> Path:
        """返回可视化和产品导出目录。"""
        return self.root / "media" / "visualizations"

    def initialize(self) -> None:
        """创建产品所需的全部本地目录。"""
        for path in (
            self.database_dir,
            self.uploads_dir,
            self.segments_dir,
            self.visualizations_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class ProductDatabase:
    """统一暴露人物库和素材库，不让应用层依赖 SQLite 细节。"""

    def __init__(self, storage: ProductStorageLayout) -> None:
        """根据产品目录构造数据库及两个持久化仓库。"""
        self.storage = storage
        self.database = SQLiteDatabase(storage.database_path)
        self.identity_gallery = SQLiteIdentityGallery(self.database)
        self.material_catalog = SQLiteMaterialCatalog(self.database)

    @classmethod
    def open(cls, root: str | Path | None = None) -> "ProductDatabase":
        """初始化并打开指定产品目录，默认读取集中配置。"""
        storage = ProductStorageLayout(
            Path(root).expanduser() if root is not None else SETTINGS.product_data_root
        )
        storage.initialize()
        service = cls(storage)
        service.database.initialize()
        return service

    def status(self) -> dict[str, object]:
        """返回不包含个人敏感内容的数据库数量摘要。"""
        with self.database.transaction() as connection:
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in (
                    "participants",
                    "participant_embeddings",
                    "materials",
                    "material_events",
                    "material_participants",
                )
            }
        return {
            "schema_version": self.database.schema_version(),
            "database_path": str(self.storage.database_path),
            "storage_root": str(self.storage.root),
            "counts": counts,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析产品数据库初始化和状态命令。"""
    parser = argparse.ArgumentParser(description="初始化 BasketEvent 产品数据库。")
    parser.add_argument(
        "command", choices=("init", "status"), nargs="?", default="status"
    )
    parser.add_argument(
        "--root",
        default=str(SETTINGS.product_data_root),
        help="产品数据根目录",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """初始化产品数据库并输出当前路径和记录数量。"""
    args = _parse_args(argv)
    product_database = ProductDatabase.open(args.root)
    result = product_database.status()
    result["command"] = args.command
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
