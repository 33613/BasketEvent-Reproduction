"""把多个事件素材中的确定身份归并为源视频范围内的人物。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _identifier(value: str) -> str:
    """生成稳定、可用于数据库主键的短文本。"""
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


class CrossMaterialIdentityAssociator:
    """用保守规则归并素材人物，不执行外观相似度猜测。

    只有 ``identified`` 且同时具有球衣颜色和号码的结果才跨素材合并。
    ``anonymous`` 与 ``conflicting`` 始终保留为素材内人物，避免把不同球员
    因为模型不确定而错误合并。
    """

    def associate(
        self,
        *,
        source_video_id: str,
        identity_reports: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """整理身份报告并返回人物档案及每段素材的人物引用。"""
        profiles: dict[str, dict[str, Any]] = {}
        material_participants: dict[str, list[dict[str, Any]]] = {}

        for material_id, report in sorted(identity_reports.items()):
            raw_resolutions = report.get("resolutions", report.get("decisions", []))
            if not isinstance(raw_resolutions, Sequence) or isinstance(
                raw_resolutions, (str, bytes)
            ):
                raise ValueError(f"{material_id} 的身份结论必须是列表")
            references: list[dict[str, Any]] = []
            for value in raw_resolutions:
                if not isinstance(value, Mapping):
                    continue
                track_id = str(value.get("track_id") or "unknown")
                status = str(value.get("status") or "anonymous")
                color = (
                    str(value["jersey_color"]).strip().lower()
                    if value.get("jersey_color") is not None
                    else None
                )
                number = (
                    str(value["jersey_number"]).strip()
                    if value.get("jersey_number") is not None
                    else None
                )
                if status == "identified" and color and number:
                    participant_id = str(value.get("participant_id") or "").strip()
                    if not participant_id:
                        participant_id = ":".join(
                            (
                                _identifier(source_video_id),
                                "jersey",
                                _identifier(color),
                                _identifier(number),
                            )
                        )
                    association_method = "jersey_color_and_number"
                else:
                    participant_id = ":".join(
                        (
                            _identifier(source_video_id),
                            "material",
                            _identifier(material_id),
                            "track",
                            _identifier(track_id),
                        )
                    )
                    association_method = "material_scoped_anonymous"

                reference = {
                    "participant_id": participant_id,
                    "track_id": track_id,
                    "jersey_color": color,
                    "jersey_number": number,
                    "player_name": value.get("player_name"),
                    "identity_status": status,
                    "association_method": association_method,
                }
                references.append(reference)
                profile = profiles.setdefault(
                    participant_id,
                    {
                        "participant_id": participant_id,
                        "display_name": value.get("player_name"),
                        "jersey_color": color,
                        "jersey_number": number,
                        "appearance_count": 0,
                        "material_ids": [],
                    },
                )
                profile["appearance_count"] += 1
                if material_id not in profile["material_ids"]:
                    profile["material_ids"].append(material_id)
            material_participants[material_id] = references

        return {
            "schema_version": "basketevent_identity_association.v1",
            "source_video_id": source_video_id,
            "association_rule": (
                "仅合并具有唯一球衣颜色和号码的 identified 结果；"
                "匿名和冲突结果保持素材隔离"
            ),
            "participants": [profiles[key] for key in sorted(profiles)],
            "material_participants": material_participants,
        }
