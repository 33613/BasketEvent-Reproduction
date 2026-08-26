"""聚合 Qwen 逐帧观察、检索可选名单并生成干净轨迹。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.config import SETTINGS
from src.modules.identity.models import (
    IdentityCandidate,
    IdentityObservation,
    ResolvedIdentity,
)
from src.modules.identity.qwen_observer import QwenTrackObserver
from src.modules.identity.sampling import TrackSampler


class IdentityResolver:
    """把一条轨迹的多帧观察解析为稳定、混合或未解析身份。"""

    def __init__(self, roster: Mapping[tuple[str, str], tuple[str, ...]] | None = None):
        """保存可选的 ``(颜色, 号码) -> 姓名`` 精确检索表。"""
        self.roster = dict(roster or {})

    @classmethod
    def from_roster_file(cls, path: str | Path | None) -> "IdentityResolver":
        """从比赛名单构建解析器；不提供名单时仍可按颜色和号码工作。"""
        if path is None:
            return cls()
        source = Path(path)
        with source.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, Mapping):
            raise ValueError(f"名单 JSON 顶层必须是对象：{source}")
        team_colors = {
            str(team): str(color).strip().lower()
            for team, color in document.get("jersey_color", {}).items()
        }
        names: dict[tuple[str, str], list[str]] = defaultdict(list)
        for player in document.get("players", []):
            if not isinstance(player, Mapping):
                continue
            color = team_colors.get(str(player.get("team_name")))
            number = str(player.get("jersey", "")).strip()
            name = str(player.get("name", "")).strip()
            if color and number and name:
                names[(color, number)].append(name)
        return cls({key: tuple(value) for key, value in names.items()})

    def resolve(
        self, track_id: str, observations: Sequence[IdentityObservation]
    ) -> ResolvedIdentity:
        """按时序证据判断轨迹状态，不把相互冲突的号码强行多数投票。"""
        on_court = [item for item in observations if item.is_on_court_player is True]
        rejected = [item for item in observations if item.is_on_court_player is False]
        if not on_court:
            reason = "没有帧被确认成场上球员"
            if not observations:
                reason = "Qwen 没有返回可用观察"
            elif rejected:
                reason = "所有明确判断都不是场上球员"
            return ResolvedIdentity(
                track_id=track_id,
                status="invalid",
                accepted=False,
                is_on_court_player=False,
                reason=reason,
            )

        grouped: dict[tuple[str, str], list[IdentityObservation]] = defaultdict(list)
        for item in on_court:
            if item.jersey_color is not None and item.jersey_number is not None:
                grouped[(item.jersey_color, item.jersey_number)].append(item)
        candidates = tuple(
            sorted(
                (
                    IdentityCandidate(
                        jersey_color=color,
                        jersey_number=number,
                        support_count=len(items),
                        confidence_sum=sum(item.confidence for item in items),
                        frames=tuple(item.frame_index for item in items),
                    )
                    for (color, number), items in grouped.items()
                ),
                key=lambda item: (item.support_count, item.confidence_sum),
                reverse=True,
            )
        )
        if not candidates:
            visible_colors = {
                item.jersey_color for item in on_court if item.jersey_color is not None
            }
            return ResolvedIdentity(
                track_id=track_id,
                status="unresolved",
                accepted=True,
                is_on_court_player=True,
                jersey_color=(
                    next(iter(visible_colors)) if len(visible_colors) == 1 else None
                ),
                reason="确认是场上球员，但球衣号码证据不足",
            )
        if len(candidates) > 1:
            return ResolvedIdentity(
                track_id=track_id,
                status="mixed",
                accepted=False,
                is_on_court_player=True,
                candidates=candidates,
                reason="同一 SAM3 轨迹包含多个球衣身份，需要先按时间拆分",
            )

        candidate = candidates[0]
        roster_names = self.roster.get(
            (candidate.jersey_color, candidate.jersey_number), ()
        )
        player_name = roster_names[0] if len(roster_names) == 1 else None
        return ResolvedIdentity(
            track_id=track_id,
            status="stable",
            accepted=True,
            is_on_court_player=True,
            jersey_color=candidate.jersey_color,
            jersey_number=candidate.jersey_number,
            player_name=player_name,
            candidates=candidates,
            reason=None if self.roster else "未提供名单，仅保留球衣颜色与号码",
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析身份识别命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对 SAM3 轨迹取样，调用 Qwen 观察并解析球员身份。"
    )
    parser.add_argument("--video_path", required=True, help="输入视频路径")
    parser.add_argument("--bbox_json_path", required=True, help="SAM3 原始轨迹 JSON")
    parser.add_argument("--json_save_path", required=True, help="干净轨迹输出 JSON")
    parser.add_argument(
        "--roster_json",
        default=None,
        help="可选比赛名单；缺省时按球衣颜色和号码标识球员",
    )
    parser.add_argument(
        "--gpus_to_use", default=SETTINGS.gpu_ids, help="Qwen 使用的 GPU 编号"
    )
    parser.add_argument(
        "--qwen_model", default=str(SETTINGS.qwen_model), help="本地 Qwen 模型目录"
    )
    parser.add_argument("--num_crops", type=int, default=10, help="每条轨迹抽取帧数")
    parser.add_argument(
        "--pad_ratio", type=float, default=0.0, help="球员边界框扩张比例"
    )
    return parser.parse_args(argv)


def _write_outputs(
    save_path: Path,
    annotations: Mapping[str, Any],
    resolutions: Sequence[ResolvedIdentity],
    observations: Mapping[str, Sequence[IdentityObservation]],
    selected_ball_id: str | None,
    ball_review: Mapping[str, Any],
) -> None:
    """分别写入下游轨迹和完整身份诊断报告。"""
    clean: dict[str, Any] = {}
    accepted_index = 0
    for identity in resolutions:
        if not identity.accepted:
            continue
        payload = annotations.get(identity.track_id, {})
        clean[f"player_{accepted_index}"] = {
            "source_track_id": identity.track_id,
            "identity_status": identity.status,
            "jersey_number": identity.jersey_number,
            "jersey_color": identity.jersey_color,
            "player_name": identity.player_name,
            "trajectory": (
                payload.get("trajectory") if isinstance(payload, Mapping) else None
            ),
        }
        accepted_index += 1
    if selected_ball_id is not None and selected_ball_id in annotations:
        payload = annotations[selected_ball_id]
        clean["ball"] = {
            "source_track_id": selected_ball_id,
            "trajectory": (
                payload.get("trajectory") if isinstance(payload, Mapping) else None
            ),
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(clean, file, ensure_ascii=False, indent=2)
    report_path = save_path.with_name(f"{save_path.stem}_identity.json")
    report = {
        "schema_version": "basketevent_identity_resolution.v2",
        "resolutions": [identity.to_dict() for identity in resolutions],
        "observations": {
            track_id: [
                {
                    "image_index": item.image_index,
                    "frame_index": item.frame_index,
                    "is_on_court_player": item.is_on_court_player,
                    "jersey_color": item.jersey_color,
                    "jersey_number": item.jersey_number,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                }
                for item in items
            ]
            for track_id, items in observations.items()
        },
        "ball_review": dict(ball_review),
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    """依次执行取样、Qwen 观察、身份解析和篮球候选选择。"""
    args = _parse_args(argv)
    video_path = SETTINGS.require_file(args.video_path, "输入视频")
    raw_path = SETTINGS.require_file(args.bbox_json_path, "SAM3 原始轨迹 JSON")
    model_path = SETTINGS.require_directory(args.qwen_model, "Qwen 模型目录")
    roster_path = (
        SETTINGS.require_file(args.roster_json, "比赛名单")
        if args.roster_json
        else None
    )

    import torch

    device = (
        f"cuda:{str(args.gpus_to_use).split(',')[0]}"
        if torch.cuda.is_available()
        else "cpu"
    )
    sampler = TrackSampler(sample_count=args.num_crops, pad_ratio=args.pad_ratio)
    annotations = sampler.load_annotations(raw_path)
    crops_by_track = sampler.sample(video_path, annotations)
    observer = QwenTrackObserver.from_pretrained(
        str(model_path), device=device, local_files_only=SETTINGS.hf_local_files_only
    )
    resolver = IdentityResolver.from_roster_file(roster_path)

    observations: dict[str, list[IdentityObservation]] = {}
    resolutions: list[ResolvedIdentity] = []
    for track_id, crops in crops_by_track.items():
        print(f"正在观察轨迹 {track_id}")
        track_observations = observer.observe(crops)
        observations[track_id] = track_observations
        resolutions.append(resolver.resolve(track_id, track_observations))

    ball_candidates = sampler.sample_ball_candidates(video_path, annotations)
    ball_review = observer.select_ball(ball_candidates)
    _write_outputs(
        Path(args.json_save_path),
        annotations,
        resolutions,
        observations,
        ball_review.get("selected_ball_id"),
        ball_review,
    )


if __name__ == "__main__":
    main()
