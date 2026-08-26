"""把事件预测汇总为可检索的视频素材统计。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MaterialStatistics:
    """保存一个或多个视频片段的聚合统计。"""

    clip_count: int
    player_prediction_count: int
    non_background_prediction_count: int
    temporal_event_count: int
    mean_confidence: float | None
    event_counts: dict[str, int]
    participant_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """返回可以直接写入 JSON 的统计字典。"""
        return asdict(self)


class MaterialStatisticsService:
    """根据 PlayNet JSON 报告生成确定性的素材统计。"""

    @staticmethod
    def _participant_label(prediction: Mapping[str, Any]) -> str:
        """在不依赖球员名单的情况下生成人物显示标签。"""
        color = str(prediction.get("jersey_color") or "unknown").lower()
        number = prediction.get("jersey_number")
        if number is not None and str(number).strip():
            return f"{color} #{str(number).strip()}"
        return str(prediction.get("player_id") or f"{color} unknown")

    def summarize(
        self,
        reports: Sequence[Mapping[str, Any]],
    ) -> MaterialStatistics:
        """汇总已经加载到内存的事件预测报告。

        Args:
            reports: 事件识别模块输出的时序预测文档。

        Returns:
            可直接用于素材目录页面的聚合统计。
        """
        event_counts: Counter[str] = Counter()
        participant_counts: Counter[str] = Counter()
        confidences: list[float] = []
        prediction_count = 0
        non_background_count = 0
        temporal_event_count = 0

        for report in reports:
            predictions = report.get("player_predictions", [])
            if not isinstance(predictions, list):
                raise ValueError("player_predictions 必须是列表")
            temporal_events = report.get("temporal_events", [])
            if not isinstance(temporal_events, list):
                raise ValueError("temporal_events 必须是列表")
            temporal_event_count += len(temporal_events)
            for prediction in predictions:
                if not isinstance(prediction, Mapping):
                    raise ValueError("每条球员预测必须是对象")
                prediction_count += 1
                event = str(prediction.get("event") or "unknown")
                event_counts[event] += 1
                participant_counts[self._participant_label(prediction)] += 1
                if event != "blank":
                    non_background_count += 1
                confidence = prediction.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidences.append(float(confidence))

        mean_confidence = sum(confidences) / len(confidences) if confidences else None
        return MaterialStatistics(
            clip_count=len(reports),
            player_prediction_count=prediction_count,
            non_background_prediction_count=non_background_count,
            temporal_event_count=temporal_event_count,
            mean_confidence=mean_confidence,
            event_counts=dict(sorted(event_counts.items())),
            participant_counts=dict(sorted(participant_counts.items())),
        )

    def summarize_files(
        self,
        report_paths: Sequence[str | Path],
    ) -> MaterialStatistics:
        """读取多个预测 JSON 文件并汇总。"""
        reports: list[Mapping[str, Any]] = []
        for path_value in report_paths:
            path = Path(path_value).expanduser()
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
            if not isinstance(value, Mapping):
                raise ValueError(f"预测报告根节点必须是对象：{path}")
            reports.append(value)
        return self.summarize(reports)
