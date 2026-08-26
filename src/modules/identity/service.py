"""编排单视频取样、证据生成、人物检索、融合和跨片段关联。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.core.config import SETTINGS
from src.modules.identity.association import CrossClipIdentityAssociator
from src.modules.identity.evidence import IdentityEvidenceProvider, QwenTrackObserver
from src.modules.identity.fusion import IdentityEvidenceFusion
from src.modules.identity.gallery import IdentityGallery, InMemoryIdentityGallery
from src.modules.identity.models import (
    BallCandidate,
    GalleryMatch,
    IdentityDecision,
    IdentityEvidence,
    IdentityObservation,
    IdentityProcessingResult,
    TrackCrop,
)
from src.modules.identity.sampling import TrackSampler


class BallSelector(Protocol):
    """约束篮球候选选择器的最小接口。"""

    def select_ball(self, candidates: Sequence[BallCandidate]) -> dict[str, Any]:
        """从候选轨迹中选择真实比赛用球。"""
        ...


class IdentityService:
    """提供单视频 Identity 流程的稳定对外接口。"""

    def __init__(
        self,
        *,
        sampler: TrackSampler,
        evidence_providers: Sequence[IdentityEvidenceProvider],
        gallery: IdentityGallery,
        fusion: IdentityEvidenceFusion,
        associator: CrossClipIdentityAssociator,
        ball_selector: BallSelector | None = None,
    ) -> None:
        """注入所有可替换组件，Service 本身不依赖具体模型或数据库。"""
        self.sampler = sampler
        self.evidence_providers = tuple(evidence_providers)
        self.gallery = gallery
        self.fusion = fusion
        self.associator = associator
        self.ball_selector = ball_selector

    def _collect_evidence(
        self,
        track_id: str,
        crops: Sequence[TrackCrop],
    ) -> tuple[IdentityEvidence, ...]:
        """调用所有证据提供器；单个实现失败时保留可审计错误而不中断流程。"""
        evidence: list[IdentityEvidence] = []
        for provider in self.evidence_providers:
            try:
                item = provider.collect(crops)
            except Exception as error:  # 模型实现可能抛出第三方库异常。
                item = IdentityEvidence(
                    source=provider.source,
                    track_id=track_id,
                    confidence=0.0,
                    metadata={"error": str(error)},
                )
            if item.track_id != track_id:
                raise ValueError(
                    f"证据轨迹编号不一致：期望 {track_id}，实际 {item.track_id}"
                )
            evidence.append(item)
        return tuple(evidence)

    def _gallery_matches(
        self,
        evidence: Sequence[IdentityEvidence],
    ) -> tuple[GalleryMatch, ...]:
        """分别使用 ReID 向量和 Qwen 属性查询人物库，并按人物去重。"""
        candidates: list[GalleryMatch] = []
        attribute_keys: set[tuple[str, str]] = set()
        for item in evidence:
            if item.embedding is not None:
                candidates.extend(self.gallery.search_by_embedding(item.embedding))
            for observation in item.observations:
                if (
                    observation.is_on_court_player is True
                    and observation.jersey_color is not None
                    and observation.jersey_number is not None
                ):
                    attribute_keys.add(
                        (observation.jersey_color, observation.jersey_number)
                    )
        for color, number in attribute_keys:
            candidates.extend(self.gallery.search_by_attributes(color, number))

        best_by_person: dict[str, GalleryMatch] = {}
        for match in candidates:
            previous = best_by_person.get(match.participant_id)
            if previous is None or match.score > previous.score:
                best_by_person[match.participant_id] = match
        return tuple(
            sorted(best_by_person.values(), key=lambda item: item.score, reverse=True)
        )

    def process(
        self,
        *,
        clip_id: str,
        video_path: str | Path,
        annotations: Mapping[str, Any],
    ) -> IdentityProcessingResult:
        """处理一个视频片段，并返回人物结论、证据和篮球选择结果。"""
        crops_by_track = self.sampler.sample(video_path, annotations)
        evidence_by_track: dict[str, tuple[IdentityEvidence, ...]] = {}
        decisions: list[IdentityDecision] = []
        for track_id, crops in crops_by_track.items():
            print(f"正在生成轨迹证据 {track_id}")
            evidence = self._collect_evidence(track_id, crops)
            matches = self._gallery_matches(evidence)
            decision = self.fusion.resolve(track_id, evidence, matches)
            decision = self.associator.associate(clip_id, decision, evidence)
            evidence_by_track[track_id] = evidence
            decisions.append(decision)

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
            clip_id=clip_id,
            decisions=tuple(decisions),
            evidence_by_track=evidence_by_track,
            selected_ball_id=ball_review.get("selected_ball_id"),
            ball_review=ball_review,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析 Identity 服务命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对 SAM3 轨迹取样，生成身份证据并关联跨片段人物编号。"
    )
    parser.add_argument("--video_path", required=True, help="输入视频路径")
    parser.add_argument("--bbox_json_path", required=True, help="SAM3 原始轨迹 JSON")
    parser.add_argument("--json_save_path", required=True, help="下游轨迹输出 JSON")
    parser.add_argument("--roster_json", default=None, help="可选比赛人物资料")
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
    """写入 PlayNet 轨迹和包含全部证据的 Identity 审计报告。"""
    clean: dict[str, Any] = {}
    for index, decision in enumerate(
        item for item in result.decisions if item.accepted
    ):
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
    report["schema_version"] = "basketevent_identity_resolution.v3"
    # 保留旧字段名称，便于 Catalog 和已有诊断工具逐步迁移。
    report["resolutions"] = [item.to_dict() for item in result.decisions]
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    """构造当前 Qwen 证据实现并运行单视频 Identity 服务。"""
    args = _parse_args(argv)
    video_path = SETTINGS.require_file(args.video_path, "输入视频")
    raw_path = SETTINGS.require_file(args.bbox_json_path, "SAM3 原始轨迹 JSON")
    model_path = SETTINGS.require_directory(args.qwen_model, "Qwen 模型目录")
    roster_path = (
        SETTINGS.require_file(args.roster_json, "比赛人物资料")
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
    gallery = InMemoryIdentityGallery.from_roster_file(roster_path)
    service = IdentityService(
        sampler=sampler,
        evidence_providers=[qwen],
        gallery=gallery,
        fusion=IdentityEvidenceFusion(),
        associator=CrossClipIdentityAssociator(gallery),
        ball_selector=qwen,
    )
    result = service.process(
        clip_id=video_path.stem,
        video_path=video_path,
        annotations=annotations,
    )
    _write_outputs(Path(args.json_save_path), annotations, result)


if __name__ == "__main__":
    main()
