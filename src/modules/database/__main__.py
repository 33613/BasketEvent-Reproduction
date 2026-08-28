"""允许使用 ``python -m src.modules.database`` 初始化或查看数据库。"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from src.core.config import SETTINGS
from src.modules.database.sqlite import ProductDatabase


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析数据库命令。"""
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
    """打开数据库并输出当前路径和记录数量。"""
    args = _parse_args(argv)
    database = ProductDatabase.open(args.root)
    result = database.status()
    result["command"] = args.command
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
