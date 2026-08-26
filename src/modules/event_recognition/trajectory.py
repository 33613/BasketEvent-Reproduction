"""提供事件训练与推理共用的轨迹缩放函数。"""

from __future__ import annotations

from typing import Any

import torch


def load_bbox_from_json_resized_onepid(
    bbox_info: dict[str, Any],
    person_id: str,
    kept_indices: list[int],
    scale_x: float,
    scale_y: float,
    to_xyxy: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取单条人物轨迹，并把边界框缩放到模型输入分辨率。

    Args:
        bbox_info: 已加载的轨迹 JSON 根对象。
        person_id: 需要读取的人物轨迹编号。
        kept_indices: 模型实际采样的原视频帧号。
        scale_x: 水平方向缩放比例。
        scale_y: 垂直方向缩放比例。
        to_xyxy: 是否把 ``[x, y, w, h]`` 转成 ``[x1, y1, x2, y2]``。
        dtype: 输出张量的数据类型。

    Returns:
        缩放后的 ``(T, 4)`` 边界框和 ``(T,)`` 有效性掩码。
    """
    pid_key = str(person_id)
    if pid_key not in bbox_info:
        frame_count = len(kept_indices)
        return (
            torch.zeros((frame_count, 4), dtype=dtype),
            torch.zeros((frame_count,), dtype=dtype),
        )

    trajectory = bbox_info[pid_key].get("trajectory", [])
    boxes = torch.zeros((len(kept_indices), 4), dtype=dtype)
    valid = torch.zeros((len(kept_indices),), dtype=dtype)
    for output_index, frame_index in enumerate(kept_indices):
        if not trajectory:
            continue
        safe_index = min(max(int(frame_index), 0), len(trajectory) - 1)
        box = trajectory[safe_index]
        if not isinstance(box, list) or len(box) != 4:
            continue
        x, y, width, height = map(float, box)
        x *= scale_x
        width *= scale_x
        y *= scale_y
        height *= scale_y
        values = [x, y, x + width, y + height] if to_xyxy else [x, y, width, height]
        boxes[output_index] = torch.tensor(values, dtype=dtype)
        valid[output_index] = 1.0
    return boxes, valid


def load_ball_from_json_resized(
    bbox_info: dict[str, Any],
    kept_indices: list[int],
    scale_x: float,
    scale_y: float,
    to_xyxy: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取篮球轨迹，并把边界框缩放到模型输入分辨率。"""
    return load_bbox_from_json_resized_onepid(
        bbox_info=bbox_info,
        person_id="ball",
        kept_indices=kept_indices,
        scale_x=scale_x,
        scale_y=scale_y,
        to_xyxy=to_xyxy,
        dtype=dtype,
    )
