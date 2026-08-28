"""使用可解释的固定规则解析单条轨迹的球衣身份。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.modules.identity.models import (
    IdentityCandidate,
    IdentityDecision,
    IdentityObservation,
)


def _identifier_part(value: str) -> str:
    """把业务字段转换为可安全写入人物编号的短文本。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


class RosterLookup:
    """按球队球衣颜色和号码查询可选名单中的球员姓名。"""

    def __init__(self, names: Mapping[tuple[str, str], Sequence[str]] | None = None):
        """保存标准化后的 ``(颜色, 号码)`` 到姓名列表映射。"""
        self._names = {
            (color.strip().lower(), str(number).strip()): tuple(values)
            for (color, number), values in (names or {}).items()
        }

    @classmethod
    def from_file(cls, path: str | Path | None) -> "RosterLookup":
        """读取作者示例格式的名单；没有名单时返回空查询器。"""
        if path is None:
            return cls()
        source = Path(path)
        with source.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, Mapping):
            raise ValueError(f"名单 JSON 顶层必须是对象：{source}")

        colors = {
            str(team): str(color).strip().lower()
            for team, color in document.get("jersey_color", {}).items()
        }
        names: dict[tuple[str, str], list[str]] = defaultdict(list)
        for player in document.get("players", []):
            if not isinstance(player, Mapping):
                continue
            color = colors.get(str(player.get("team_name")))
            number = str(player.get("jersey", "")).strip()
            name = str(player.get("name", "")).strip()
            if color and number and name:
                names[(color, number)].append(name)
        return cls(names)

    def unique_name(self, color: str, number: str) -> str | None:
        """仅当颜色和号码恰好对应一个姓名时返回该姓名。"""
        values = self._names.get((color.strip().lower(), number.strip()), ())
        return values[0] if len(values) == 1 else None


class IdentityResolver:
    """将逐帧观察转换为确定、冲突或匿名三种结果。

    规则故意保持保守：同一轨迹只出现一个完整的颜色号码组合时才能确定
    身份；出现多个组合时标记冲突；没有完整组合时标记匿名。三种情况都
    保留原始 SAM3 轨迹，避免 Qwen 成为事件推理前的硬过滤器。
    """

    def __init__(self, roster: RosterLookup | None = None) -> None:
        """注入可选名单查询；名单只补充姓名，不参与轨迹保留决策。"""
        self.roster = roster or RosterLookup()

    @staticmethod
    def _candidates(
        observations: Sequence[IdentityObservation],
    ) -> tuple[IdentityCandidate, ...]:
        """按颜色和号码聚合支持帧，不用多数票掩盖身份切换。"""
        grouped: dict[tuple[str, str], list[IdentityObservation]] = defaultdict(list)
        for item in observations:
            if (
                item.is_on_court_player is True
                and item.jersey_color is not None
                and item.jersey_number is not None
            ):
                grouped[(item.jersey_color, item.jersey_number)].append(item)
        candidates = [
            IdentityCandidate(
                jersey_color=color,
                jersey_number=number,
                support_count=len(items),
                confidence_sum=sum(item.confidence for item in items),
                frames=tuple(item.frame_index for item in items),
            )
            for (color, number), items in grouped.items()
        ]
        candidates.sort(
            key=lambda item: (item.support_count, item.confidence_sum), reverse=True
        )
        return tuple(candidates)

    @staticmethod
    def _anonymous_id(game_id: str, clip_id: str, track_id: str) -> str:
        """生成同一次输入重复运行时保持一致的片段内匿名编号。"""
        return ":".join(
            (
                _identifier_part(game_id),
                "clip",
                _identifier_part(clip_id),
                "track",
                _identifier_part(track_id),
            )
        )

    def resolve(
        self,
        *,
        game_id: str,
        clip_id: str,
        track_id: str,
        observations: Sequence[IdentityObservation],
    ) -> IdentityDecision:
        """根据固定规则解析一条轨迹，并返回稳定人物编号。"""
        candidates = self._candidates(observations)
        if len(candidates) == 1:
            candidate = candidates[0]
            participant_id = ":".join(
                (
                    _identifier_part(game_id),
                    "jersey",
                    _identifier_part(candidate.jersey_color),
                    _identifier_part(candidate.jersey_number),
                )
            )
            return IdentityDecision(
                track_id=track_id,
                status="identified",
                accepted=True,
                participant_id=participant_id,
                jersey_color=candidate.jersey_color,
                jersey_number=candidate.jersey_number,
                player_name=self.roster.unique_name(
                    candidate.jersey_color, candidate.jersey_number
                ),
                candidates=candidates,
                reason="所有完整观察只支持一个球衣颜色和号码组合",
            )

        visible_colors = {
            item.jersey_color
            for item in observations
            if item.is_on_court_player is True and item.jersey_color is not None
        }
        status = "conflicting" if len(candidates) > 1 else "anonymous"
        reason = (
            "同一 SAM3 轨迹出现多个球衣身份，需后续人工检查或轨迹拆分"
            if status == "conflicting"
            else "没有得到完整且一致的球衣颜色和号码"
        )
        return IdentityDecision(
            track_id=track_id,
            status=status,
            accepted=True,
            participant_id=self._anonymous_id(game_id, clip_id, track_id),
            jersey_color=(
                next(iter(visible_colors)) if len(visible_colors) == 1 else None
            ),
            candidates=candidates,
            reason=reason,
        )
