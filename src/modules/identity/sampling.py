"""从 SAM3 轨迹中抽取供视觉模型判断的截图证据。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.modules.identity.models import BallCandidate, TrackCrop


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """校验并规范化 ``[x, y, w, h]`` 边界框。"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    x, y, width, height = (float(item) for item in value)
    if width <= 1 or height <= 1:
        return None
    return x, y, width, height


def _uniform_positions(length: int, count: int) -> list[int]:
    """生成不重复的均匀位置，短轨迹不会复制最后一帧。"""
    if length <= 0 or count <= 0:
        return []
    if length <= count:
        return list(range(length))
    return sorted(set(np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()))


def _crop_player(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float,
) -> np.ndarray | None:
    """按边界框裁剪球员，并限制裁剪区域不越过画面。"""
    height, width = frame.shape[:2]
    x, y, box_width, box_height = bbox
    pad_width = box_width * pad_ratio
    pad_height = box_height * pad_ratio
    x1 = max(0, int(round(x - pad_width)))
    y1 = max(0, int(round(y - pad_height)))
    x2 = min(width, int(round(x + box_width + pad_width)))
    y2 = min(height, int(round(y + box_height + pad_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def _sharpness(image: np.ndarray) -> float:
    """使用拉普拉斯方差记录截图清晰度，供后续策略比较。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class TrackSampler:
    """按每条轨迹自身的有效帧均匀抽样，不制造重复证据。"""

    def __init__(self, sample_count: int = 10, pad_ratio: float = 0.0) -> None:
        """保存取样数量和球员边界框扩张比例。"""
        if sample_count <= 0:
            raise ValueError("sample_count 必须为正整数")
        if pad_ratio < 0:
            raise ValueError("pad_ratio 不能为负数")
        self.sample_count = sample_count
        self.pad_ratio = pad_ratio

    @staticmethod
    def load_annotations(path: str | Path) -> dict[str, Any]:
        """读取并校验 SAM3 轨迹 JSON 的顶层结构。"""
        source = Path(path)
        with source.open("r", encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"轨迹 JSON 顶层必须是对象：{source}")
        return value

    def sample(
        self,
        video_path: str | Path,
        annotations: Mapping[str, Any],
        track_prefix: str = "player",
        track_ids: Sequence[str] | None = None,
    ) -> dict[str, list[TrackCrop]]:
        """为指定类型的轨迹提取按时间排序的截图。

        ``track_ids`` 为空时处理所有匹配前缀的轨迹；传入编号时只读取这些
        轨迹。事件身份阶段使用后者，避免把同一窗口里的无关球员送入Qwen。
        """
        selected_track_ids = (
            {str(track_id) for track_id in track_ids} if track_ids is not None else None
        )
        selected_frames: dict[
            str, list[tuple[int, tuple[float, float, float, float]]]
        ] = {}
        for raw_track_id, payload in annotations.items():
            track_id = str(raw_track_id)
            if not track_id.startswith(track_prefix) or not isinstance(
                payload, Mapping
            ):
                continue
            if selected_track_ids is not None and track_id not in selected_track_ids:
                continue
            trajectory = payload.get("trajectory")
            if not isinstance(trajectory, list):
                continue
            valid = [
                (frame_index, bbox)
                for frame_index, value in enumerate(trajectory)
                if (bbox := _normalize_bbox(value)) is not None
            ]
            positions = _uniform_positions(len(valid), self.sample_count)
            selected_frames[track_id] = [valid[position] for position in positions]

        required_frames = sorted(
            {
                frame_index
                for items in selected_frames.values()
                for frame_index, _ in items
            }
        )
        frames = self._read_frames(video_path, required_frames)
        result: dict[str, list[TrackCrop]] = {}
        for track_id, items in selected_frames.items():
            crops: list[TrackCrop] = []
            for image_index, (frame_index, bbox) in enumerate(items, start=1):
                frame = frames.get(frame_index)
                if frame is None:
                    continue
                crop = _crop_player(frame, bbox, self.pad_ratio)
                if crop is None or crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(
                    TrackCrop(
                        track_id=track_id,
                        image_index=image_index,
                        frame_index=frame_index,
                        image=Image.fromarray(rgb).convert("RGB"),
                        sharpness=_sharpness(crop),
                    )
                )
            if crops:
                result[track_id] = crops
        return result

    @staticmethod
    def _read_frames(
        video_path: str | Path, frame_indices: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """按需读取视频帧，并在多条轨迹之间复用解码结果。"""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{video_path}")
        frames: dict[int, np.ndarray] = {}
        try:
            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames[int(frame_index)] = frame
        finally:
            capture.release()
        return frames

    def sample_ball_candidates(
        self,
        video_path: str | Path,
        annotations: Mapping[str, Any],
        maximum_candidates: int = 8,
    ) -> list[BallCandidate]:
        """抽取篮球候选，并按几何与运动启发式分数排序。"""
        rows: list[tuple[str, list[tuple[int, tuple[float, float, float, float]]]]] = []
        for raw_track_id, payload in annotations.items():
            track_id = str(raw_track_id)
            if not track_id.startswith("ball") or not isinstance(payload, Mapping):
                continue
            trajectory = payload.get("trajectory")
            if not isinstance(trajectory, list):
                continue
            valid = [
                (frame_index, bbox)
                for frame_index, value in enumerate(trajectory)
                if (bbox := _normalize_bbox(value)) is not None
            ]
            if valid:
                rows.append((track_id, valid))

        required_frames = sorted(
            {
                items[position][0]
                for _, items in rows
                for position in _uniform_positions(
                    len(items), min(8, self.sample_count)
                )
            }
        )
        frames = self._read_frames(video_path, required_frames)
        candidates: list[BallCandidate] = []
        for track_id, items in rows:
            statistics = self._ball_statistics(track_id, items)
            crops: list[TrackCrop] = []
            positions = _uniform_positions(len(items), min(8, self.sample_count))
            for image_index, position in enumerate(positions, start=1):
                frame_index, bbox = items[position]
                frame = frames.get(frame_index)
                if frame is None:
                    continue
                crop = self._crop_ball_context(frame, bbox)
                if crop is None:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(
                    TrackCrop(
                        track_id=track_id,
                        image_index=image_index,
                        frame_index=frame_index,
                        image=Image.fromarray(rgb).convert("RGB"),
                        sharpness=_sharpness(crop),
                    )
                )
            candidates.append(
                BallCandidate(track_id, tuple(crops), statistics=statistics)
            )
        candidates.sort(
            key=lambda item: float(item.statistics["heuristic_score"]), reverse=True
        )
        return candidates[:maximum_candidates]

    @staticmethod
    def _ball_statistics(
        track_id: str,
        items: Sequence[tuple[int, tuple[float, float, float, float]]],
    ) -> dict[str, float | int | str]:
        """计算篮球候选的尺寸、连续性和运动范围。"""
        frames = [frame for frame, _ in items]
        boxes = np.asarray([bbox for _, bbox in items], dtype=np.float32)
        widths, heights = boxes[:, 2], boxes[:, 3]
        center_x = boxes[:, 0] + widths / 2
        center_y = boxes[:, 1] + heights / 2
        speeds = []
        for index in range(1, len(items)):
            gap = frames[index] - frames[index - 1]
            if gap > 0:
                speeds.append(
                    math.hypot(
                        float(center_x[index] - center_x[index - 1]),
                        float(center_y[index] - center_y[index - 1]),
                    )
                    / gap
                )
        move_range = math.hypot(
            float(np.max(center_x) - np.min(center_x)),
            float(np.max(center_y) - np.min(center_y)),
        )
        mean_width = float(np.mean(widths))
        mean_height = float(np.mean(heights))
        mean_area = float(np.mean(widths * heights))
        aspect = float(np.mean(widths / np.maximum(heights, 1e-6)))
        size_score = 1.0
        if mean_width < 6 or mean_height < 6:
            size_score *= 0.2
        if mean_width > 50 or mean_height > 50:
            size_score *= 0.15
        if mean_area > 2500:
            size_score *= 0.05
        aspect_score = min(1.0, max(0.0, 1.4 - abs(aspect - 1.0)))
        motion_score = min(1.0, move_range / 500.0)
        speed_score = min(1.0, (float(np.median(speeds)) if speeds else 0.0) / 10.0)
        continuity_score = min(1.0, len(items) / 80.0)
        score = (
            0.30 * size_score
            + 0.20 * aspect_score
            + 0.20 * motion_score
            + 0.20 * speed_score
            + 0.10 * continuity_score
        )
        return {
            "id": track_id,
            "n_frames": len(items),
            "mean_width": mean_width,
            "mean_height": mean_height,
            "mean_area": mean_area,
            "move_range": move_range,
            "median_speed": float(np.median(speeds)) if speeds else 0.0,
            "heuristic_score": float(score),
        }

    @staticmethod
    def _crop_ball_context(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        pad_ratio: float = 3.5,
    ) -> np.ndarray | None:
        """裁剪篮球周围上下文，并用红框标出候选位置。"""
        height, width = frame.shape[:2]
        x, y, box_width, box_height = bbox
        center_x = x + box_width / 2
        center_y = y + box_height / 2
        side = max(max(box_width, box_height) * pad_ratio, 112)
        x1 = max(0, int(round(center_x - side / 2)))
        y1 = max(0, int(round(center_y - side / 2)))
        x2 = min(width, int(round(center_x + side / 2)))
        y2 = min(height, int(round(center_y + side / 2)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(
            crop,
            (int(round(x - x1)), int(round(y - y1))),
            (int(round(x + box_width - x1)), int(round(y + box_height - y1))),
            (0, 0, 255),
            2,
        )
        return crop


def build_ball_contact_sheet(
    candidates: Sequence[BallCandidate], cell_size: int = 180
) -> Image.Image:
    """把多个篮球候选排成一张供 Qwen 比较的联系表。"""
    if not candidates:
        raise ValueError("没有可用于生成联系表的篮球候选")
    columns = max((len(candidate.crops) for candidate in candidates), default=1)
    row_height = cell_size + 30
    sheet = Image.new(
        "RGB", (columns * cell_size, len(candidates) * row_height), "white"
    )
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for row_index, candidate in enumerate(candidates):
        for column_index, crop in enumerate(candidate.crops):
            image = crop.image.resize((cell_size, cell_size), Image.Resampling.BICUBIC)
            x = column_index * cell_size
            y = row_index * row_height
            sheet.paste(image, (x, y))
        draw.text(
            (4, row_index * row_height + cell_size + 7),
            f"candidate={candidate.track_id}",
            fill="black",
            font=font,
        )
    return sheet
