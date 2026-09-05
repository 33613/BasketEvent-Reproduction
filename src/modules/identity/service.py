"""编排轨迹取样、Qwen 观察和固定规则身份解析。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.core.config import SETTINGS
from src.modules.identity.models import (
    BallCandidate,
    IdentityObservation,
    IdentityProcessingResult,
    TrackCrop,
)
from src.modules.identity.qwen_observer import QwenTrackObserver
from src.modules.identity.resolver import IdentityResolver, RosterLookup
from src.modules.identity.sampling import TrackSampler


class TrackObserver(Protocol):
    """约束把一组轨迹截图转换为逐帧观察的最小接口。"""

    def observe(self, crops: Sequence[TrackCrop]) -> list[IdentityObservation]:
        """返回每张截图的球衣属性观察。"""
        ...


class BallSelector(Protocol):
    """约束篮球候选选择器的最小接口。"""

    def select_ball(self, candidates: Sequence[BallCandidate]) -> dict[str, Any]:
        """从候选轨迹中选择真实比赛用球。"""
        ...


class IdentityService:
    """提供最小身份处理流程的稳定对外接口。

    Service 的执行顺序直接对应产品工作流：取样、观察、规则解析。模型失败
    只会写入诊断信息，不会删除 SAM3 人物轨迹。
    """

    def __init__(
        self,
        *,
        sampler: TrackSampler,
        observer: TrackObserver,
        resolver: IdentityResolver,
        ball_selector: BallSelector | None = None,
    ) -> None:
        """注入取样器、观察器和规则解析器。"""
        self.sampler = sampler
        self.observer = observer
        self.resolver = resolver
        self.ball_selector = ball_selector

    def process(
        self,
        *,
        game_id: str,
        clip_id: str,
        video_path: str | Path,
        annotations: Mapping[str, Any],
    ) -> IdentityProcessingResult:
        """处理一个视频片段，并返回人物结论和篮球选择结果。"""
        crops_by_track = self.sampler.sample(video_path, annotations)
        observations_by_track: dict[str, tuple[IdentityObservation, ...]] = {}
        errors_by_track: dict[str, str] = {}
        decisions = []
        for track_id, crops in crops_by_track.items():
            print(f"正在观察轨迹 {track_id}")
            try:
                observations = tuple(self.observer.observe(crops))
            except Exception as error:  # 第三方模型异常不能中断后续事件识别。
                observations = ()
                errors_by_track[track_id] = str(error)
            observations_by_track[track_id] = observations
            decisions.append(
                self.resolver.resolve(
                    game_id=game_id,
                    clip_id=clip_id,
                    track_id=track_id,
                    observations=observations,
                )
            )

        ball_candidates = self.sampler.sample_ball_candidates(video_path, annotations)
        if self.ball_selector is None:
            ball_review: dict[str, Any] = {
                "selected_ball_id": (
                    ball_candidates[0].track_id if len(ball_candidates) == 1 else None
                ),
                "reason": "没有配置篮球候选选择器",
            }
        else:
            ball_review = self.ball_selector.select_ball(ball_candidates)
        return IdentityProcessingResult(
            game_id=game_id,
            clip_id=clip_id,
            decisions=tuple(decisions),
            observations_by_track=observations_by_track,
            errors_by_track=errors_by_track,
            selected_ball_id=ball_review.get("selected_ball_id"),
            ball_review=ball_review,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析身份服务命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对 SAM3 轨迹取样，用 Qwen 观察并按固定规则生成身份。"
    )
    parser.add_argument("--video_path", required=True, help="输入视频路径")
    parser.add_argument("--bbox_json_path", required=True, help="SAM3 原始轨迹 JSON")
    parser.add_argument("--json_save_path", required=True, help="下游轨迹输出 JSON")
    parser.add_argument("--game_id", default=None, help="比赛编号；用于生成稳定人物编号")
    parser.add_argument("--roster_json", default=None, help="可选比赛名单，仅补充姓名")
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
    result: IdentityProcessingResult,
) -> None:
    """写入 PlayNet 轨迹和包含逐帧观察的身份审计报告。"""
    clean: dict[str, Any] = {}
    for index, decision in enumerate(result.decisions):
        payload = annotations.get(decision.track_id, {})
        clean[f"player_{index}"] = {
            "source_track_id": decision.track_id,
            "participant_id": decision.participant_id,
            "identity_status": decision.status,
            "jersey_number": decision.jersey_number,
            "jersey_color": decision.jersey_color,
            "player_name": decision.player_name,
            "trajectory": (
                payload.get("trajectory") if isinstance(payload, Mapping) else None
            ),
        }
    if result.selected_ball_id is not None and result.selected_ball_id in annotations:
        payload = annotations[result.selected_ball_id]
        clean["ball"] = {
            "source_track_id": result.selected_ball_id,
            "trajectory": (
                payload.get("trajectory") if isinstance(payload, Mapping) else None
            ),
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(clean, file, ensure_ascii=False, indent=2)
    report_path = save_path.with_name(f"{save_path.stem}_identity.json")
    report = result.to_dict()
    report["schema_version"] = "basketevent_identity_resolution.v4"
    report["resolutions"] = [item.to_dict() for item in result.decisions]
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    """构造最小身份流程并处理一个视频片段。"""
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
    qwen = QwenTrackObserver.from_pretrained(
        str(model_path), device=device, local_files_only=SETTINGS.hf_local_files_only
    )
    service = IdentityService(
        sampler=sampler,
        observer=qwen,
        resolver=IdentityResolver(RosterLookup.from_file(roster_path)),
        ball_selector=qwen,
    )
    game_id = args.game_id or video_path.parent.parent.name or "unknown_game"
    result = service.process(
        game_id=game_id,
        clip_id=video_path.stem,
        video_path=video_path,
        annotations=annotations,
    )
    _write_outputs(Path(args.json_save_path), annotations, result)


if __name__ == "__main__":
    main()
