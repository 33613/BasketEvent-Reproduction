"""按事件或人物查询已经登记的产品素材。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.modules.database import ProductDatabase


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析素材检索命令行参数。"""
    parser = argparse.ArgumentParser(description="查询 BasketEvent 产品素材库。")
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--event", default=None, help="事件名称，如 Made Shot")
    parser.add_argument("--participant-id", default=None, help="稳定人物编号")
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行组合查询并输出 JSON。"""
    args = parse_args(argv)
    database = ProductDatabase.open(args.database_root)
    materials = database.find_materials(
        event=args.event,
        participant_id=args.participant_id,
        minimum_confidence=args.minimum_confidence,
    )
    print(
        json.dumps(
            {
                "count": len(materials),
                "materials": [item.to_dict() for item in materials],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
