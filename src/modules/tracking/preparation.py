"""把 SAM3 原始轨迹整理为 PlayNet 可直接读取的结构。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _normalize_bbox(value: Any) -> list[float] | None:
    """验证一个 ``[x, y, width, height]`` 边界框。"""
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    x, y, width, height = (float(item) for item in value)
    if width <= 0 or height <= 0:
        return None
    return [x, y, width, height]


def _normalize_trajectory(value: Any) -> tuple[list[list[float] | None], int]:
    """规范化一条轨迹，并返回有效边界框数量。"""
    if not isinstance(value, list):
        return [], 0
    trajectory = [_normalize_bbox(item) for item in value]
    return trajectory, sum(item is not None for item in trajectory)


def prepare_model_tracks(raw_document: Mapping[str, Any]) -> dict[str, Any]:
    """保留所有有效人物轨迹，并选择覆盖帧最多的篮球轨迹。

    该步骤只处理数据结构，不识别球衣、号码或真实身份。Qwen 失败不会
    导致人物轨迹从 PlayNet 输入中消失。
    """
    players: dict[str, Any] = {}
    ball_candidates: list[tuple[int, str, list[list[float] | None]]] = []
    maximum_length = 0

    for track_id, raw_value in raw_document.items():
        if not isinstance(raw_value, Mapping):
            continue
        trajectory, valid_count = _normalize_trajectory(raw_value.get("trajectory"))
        maximum_length = max(maximum_length, len(trajectory))
        if str(track_id).startswith("player_") and valid_count > 0:
            players[str(track_id)] = {
                "trajectory": trajectory,
                "source_track_id": str(track_id),
                "identity_status": "unresolved",
                "valid_bbox_count": valid_count,
            }
        elif str(track_id) == "ball" or str(track_id).startswith("ball_"):
            ball_candidates.append((valid_count, str(track_id), trajectory))

    result = dict(sorted(players.items()))
    if ball_candidates:
        # 优先选择有效帧最多的候选；数量相同时使用稳定的轨迹编号排序。
        _, source_track_id, trajectory = max(
            ball_candidates,
            key=lambda item: (item[0], item[1]),
        )
        result["ball"] = {
            "trajectory": trajectory,
            "source_track_id": source_track_id,
        }
    else:
        result["ball"] = {
            "trajectory": [None] * maximum_length,
            "source_track_id": None,
        }
    return result


def prepare_model_tracks_file(
    raw_json_path: str | Path,
    output_json_path: str | Path,
    report_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """读取 SAM3 JSON，写出 PlayNet 输入及可选审计报告。"""
    source = Path(raw_json_path).expanduser()
    destination = Path(output_json_path).expanduser()
    with source.open("r", encoding="utf-8") as file:
        raw_document = json.load(file)
    if not isinstance(raw_document, Mapping):
        raise ValueError("SAM3 轨迹 JSON 根节点必须是对象")

    prepared = prepare_model_tracks(raw_document)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    player_ids = [key for key in prepared if key.startswith("player_")]
    report = {
        "schema_version": "basketevent_track_preparation.v1",
        "source": str(source),
        "output": str(destination),
        "player_count": len(player_ids),
        "player_ids": player_ids,
        "selected_ball_track_id": prepared["ball"].get("source_track_id"),
        "identity_was_used_for_filtering": False,
    }
    if report_json_path is not None:
        report_path = Path(report_json_path).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析轨迹结构准备命令。"""
    parser = argparse.ArgumentParser(description="准备不依赖身份识别的 PlayNet 轨迹。")
    parser.add_argument("--raw-json", required=True, help="SAM3 原始轨迹 JSON")
    parser.add_argument("--output-json", required=True, help="PlayNet 输入轨迹 JSON")
    parser.add_argument("--report-json", default="", help="可选审计报告 JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """执行轨迹结构准备并打印审计信息。"""
    args = _parse_args(argv)
    report = prepare_model_tracks_file(
        raw_json_path=args.raw_json,
        output_json_path=args.output_json,
        report_json_path=args.report_json or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
