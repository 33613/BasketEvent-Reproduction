"""BARD 独立测试集的构建、链路运行和离线评分入口。

构建不依赖 SAM3/Qwen 成功与否；运行不读取 GT；评分不修改预测。
人工标注保存在 annotations.json，原始 BARD 描述只作为标注提示。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from src.modules.evaluation.metrics import (
    EVENTS, event_name, player_pair_metrics, sequence_metrics, temporal_hits,
)
from src.modules.ingestion.bard.annotations_cli import discover_games
from src.modules.ingestion.bard.labeling import BardLabelMapper

PROJECT = Path(__file__).resolve().parents[2]
KNOWN_DEVELOPMENT_GAME = "bkn-vs-det-0022400861"


def read_json(path):
    """兼容 Windows 编辑器写入的 UTF-8 BOM。"""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    """原子写入运行报告，避免中断留下不完整 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    """流式计算内容摘要，校验传输且固定测试输入。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root, relative):
    """清单路径只能指向指定数据根目录内，禁止路径穿越。"""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"路径越出数据根目录：{relative}")
    return path


def draft_events(document):
    """逐动作映射，不套用训练用 Scheme A 的多标签排除/跨动作去重。

    BARD 的动作列表及隐含助攻不保证时间顺序；这里的结果必须人工核验，
    跳球还需要补第二位参与人，不能将该草稿直接称为序列 GT。
    """
    events, anomalies = [], []
    for index, action in enumerate(document.get("actions", [])):
        result = BardLabelMapper().map_document({"actions": [action]})
        anomalies.extend(item.to_dict() for item in result.anomalies)
        for item in result.contributions:
            events.append({"event": item.label, "actor": f"{item.jersey_color}#{item.jersey_number}",
                           "start_seconds": None, "end_seconds": None,
                           "source_action_index": index})
    return events, anomalies


def build_bundle(args):
    """先按比赛留出，再按事件抽样；只扫描原视频和原始结构化描述。"""
    root, output = args.data_root.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("输出目录已存在；请换新名称，避免覆盖人工标注或冻结清单。")
    rng = random.Random(args.seed)
    all_clips = getattr(args, "all_clips", False)
    games = discover_games(root, args.games)
    if all_clips and not args.games:
        raise ValueError("--all-clips 必须显式给出 --games，避免意外打包全部 BARD。")
    if not args.games:
        games = [g for g in games if g != KNOWN_DEVELOPMENT_GAME]
    if not games:
        raise ValueError("没有独立候选比赛；已排除反复调试过的 bkn-vs-det 比赛。")
    rng.shuffle(games)
    games = sorted(games if args.games else games[:args.game_count])
    candidates, source_errors = [], []
    for game in games:
        for video in sorted((root / game / "video").glob("*.mp4")):
            action_path = root / game / "description" / "action" / (video.stem + ".json")
            try:
                source = read_json(action_path)
                events, anomalies = draft_events(source)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                source_errors.append({"game": game, "clip": video.stem, "error": str(error)})
                if not all_clips:
                    continue
                source = {"video": video.name, "actions": [], "source_error": str(error)}
                events = []
                anomalies = [{"code": "INVALID_OR_MISSING_ACTION_SOURCE", "severity": "error",
                              "message": str(error), "context": {"game": game, "clip": video.stem}}]
            candidates.append({"game": game, "clip": video.stem, "source": source,
                               "video": video, "events": events, "anomalies": anomalies})
    if all_clips:
        # 整场模式保留无可映射动作的片段，不能只留下“有答案”的容易样本。
        selected = sorted(candidates, key=lambda item: (item["game"], item["clip"]))
    else:
        # 少样本类别优先；一个片段可以同时支持进球与助攻，不复制视频充数。
        rng.shuffle(candidates)
        available = Counter(event["event"] for c in candidates for event in c["events"])
        selected, selected_keys = [], set()
        for label in sorted(EVENTS, key=lambda name: (available[name], name)):
            supporting = sum(any(e["event"] == label for e in c["events"]) for c in selected)
            for item in candidates:
                if supporting >= args.per_class:
                    break
                key = (item["game"], item["clip"])
                if key not in selected_keys and any(e["event"] == label for e in item["events"]):
                    selected.append(item)
                    selected_keys.add(key)
                    supporting += 1
    if not selected:
        raise ValueError("候选池没有可映射事件；检查数据路径和 BARD 动作格式。")
    items, annotations = [], []
    for item in selected:
        sample_id = f"{item['game']}__{item['clip']}"
        video_relative = f"videos/{sample_id}.mp4"
        source_relative = f"sources/{sample_id}.json"
        write_json(output / source_relative, item["source"])
        if args.copy_videos:
            (output / "videos").mkdir(exist_ok=True)
            shutil.copy2(item["video"], output / video_relative)
        items.append({"sample_id": sample_id, "game": item["game"], "clip": item["clip"],
                      "video": video_relative, "source_video": str(item["video"]),
                      "source_action": source_relative, "sha256": sha256(item["video"]),
                      "size_bytes": item["video"].stat().st_size,
                      "strata": sorted({e["event"] for e in item["events"]})})
        annotations.append({"sample_id": sample_id, "reviewed": False, "reviewer": "",
                            "events": item["events"], "mapping_warnings": item["anomalies"],
                            "review_note": "核验全部可见事件、颜色号码、先后顺序；时间可暂缺。"})
    coverage = {name: sum(name in item["strata"] for item in items) for name in EVENTS}
    manifest = {"schema_version": "basketevent_bard_evaluation.v1", "seed": args.seed,
                "purpose": ("complete_games_product_evaluation" if all_clips
                            else "independent_product_pilot_not_official_benchmark"),
                "author_checkpoint_training_overlap": "unknown",
                "known_development_games_included": [
                    game for game in games if game == KNOWN_DEVELOPMENT_GAME
                ],
                "excluded_development_games": ([] if KNOWN_DEVELOPMENT_GAME in games
                                               else [KNOWN_DEVELOPMENT_GAME]),
                "games": games,
                "per_class_requested": None if all_clips else args.per_class,
                "class_clip_coverage": coverage,
                "sampling": ("all_clips_from_explicit_games" if all_clips
                             else "game_subset_then_event_stratified_not_natural_frequency"),
                "unmapped_candidate_count": sum(not c["events"] for c in candidates),
                "source_errors": source_errors, "items": items,
                "total_bytes": sum(item["size_bytes"] for item in items)}
    write_json(output / "manifest.json", manifest)
    write_json(output / "annotations.json", {"identity_key": "color#number_within_game",
                                             "samples": annotations})
    return {"bundle": str(output), "sample_count": len(items), "games": games,
            "total_mib": round(manifest["total_bytes"] / 1024**2, 1),
            "class_clip_coverage": coverage, "copied_videos": args.copy_videos}


def materialize_bundle(bundle, method="copy"):
    """按冻结清单复制或硬链接媒体；不重新抽样、不覆盖不同内容。"""
    if method not in {"copy", "hardlink"}:
        raise ValueError(f"未知素材落盘方式：{method}")
    manifest = read_json(bundle / "manifest.json")
    for item in manifest["items"]:
        target = inside(bundle, item["video"])
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise ValueError(f"已有视频摘要不一致：{target}")
            continue
        source = Path(item["source_video"])
        if sha256(source) != item["sha256"]:
            raise ValueError(f"源视频发生变化：{source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if method == "hardlink":
            try:
                os.link(source, target)
            except OSError as error:
                raise OSError(
                    f"无法创建硬链接（源和目标可能不在同一文件系统）：{source} -> {target}"
                ) from error
        else:
            shutil.copy2(source, target)
    return {"copied_linked_or_verified": len(manifest["items"]), "method": method}


def verify_bundle(bundle):
    """服务器收包后校验全部媒体，缺失或摘要不同直接失败。"""
    manifest = read_json(bundle / "manifest.json")
    for item in manifest["items"]:
        if sha256(inside(bundle, item["video"])) != item["sha256"]:
            raise ValueError(f"视频摘要不一致：{item['sample_id']}")
    return {"verified_videos": len(manifest["items"])}


def run_bundle(args):
    """顺序运行现有产品链路；只传视频，不向模型泄漏描述、事件或号码 GT。"""
    import cv2

    bundle, run_root = args.bundle.resolve(), args.run_root.resolve()
    fallback_gpu = args.gpu or "0"
    sam3_gpus = args.sam3_gpus or fallback_gpu
    playnet_gpu = args.playnet_gpu if args.playnet_gpu is not None else int(fallback_gpu)
    identity_gpus = args.identity_gpus or fallback_gpu
    manifest = read_json(bundle / "manifest.json")
    manifest_hash = sha256(bundle / "manifest.json")
    run_root.mkdir(parents=True, exist_ok=True)
    from src.core.config import SETTINGS
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                            text=True, capture_output=True).stdout.strip()
    config = {"manifest_sha256": manifest_hash, "mode": args.pipeline_mode,
              "identity_num_crops": 10, "fps_policy": "probe_each_video_integer_cfr",
              "with_identity": True, "window_seconds": 12 if args.pipeline_mode == "product" else 3600,
              "commit": commit, "checkpoint_sha256": sha256(SETTINGS.event_checkpoint),
              "sam3_checkpoint": str(SETTINGS.sam3_checkpoint),
              "timesformer_model": str(SETTINGS.timesformer_model),
              "qwen_model": str(SETTINGS.qwen_model)}
    config_path = run_root / "evaluation_config.json"
    if config_path.exists() and read_json(config_path)["configuration"] != config:
        raise ValueError("本实验目录的清单或协议已改变；比较新配置请使用新的 run-root。")
    if not config_path.exists():
        write_json(config_path, {"configuration": config, "commit": commit,
                                "python": sys.version, "checkpoint": str(SETTINGS.event_checkpoint),
                                "checkpoint_sha256": sha256(SETTINGS.event_checkpoint)})
    report_path = run_root / "run_report.json"
    report = read_json(report_path) if report_path.exists() else {"results": {}}
    report["manifest_sha256"] = manifest_hash
    items = manifest["items"][:args.limit] if args.limit else manifest["items"]
    for index, item in enumerate(items, 1):
        sample_id = item["sample_id"]
        video = inside(bundle, item["video"])
        if sha256(video) != item["sha256"]:
            raise ValueError(f"测试视频摘要不一致：{sample_id}")
        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not math.isfinite(fps) or fps <= 0 or abs(fps - round(fps)) > 0.01:
            raise ValueError(f"当前推理接口要求整数帧率；先显式准备 CFR 副本：{video}, fps={fps}")
        sample_root = inside(run_root, sample_id)
        sample_root.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-m", "src.application.process_long_video", str(video),
                   "--runtime-root", str(sample_root), "--ffmpeg-binary", args.ffmpeg_binary,
                   "--window-seconds", str(config["window_seconds"]),
                   "--overlap-seconds", "2" if args.pipeline_mode == "product" else "0",
                   "--sam3-gpus", sam3_gpus, "--playnet-gpu", str(playnet_gpu),
                   "--fps-in", str(round(fps)), "--with-identity",
                   "--identity-gpus", identity_gpus,
                   "--identity-num-crops", "10"]
        print(f"[{index}/{len(items)}] {sample_id} fps={fps}", flush=True)
        row = {"status": "running", "command": command, "fps": fps}
        report["results"][sample_id] = row
        write_json(report_path, report)
        with (sample_root / "pipeline.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        states = list(sample_root.glob("*/job_state.json"))
        row.update(return_code=completed.returncode, status="failed", job_root=None)
        if len(states) == 1:
            state = read_json(states[0])
            row.update(job_root=str(states[0].parent), status=state.get("status", "unknown"))
        elif states:
            row["error"] = "存在多个任务状态；不能确定本次结果，拒绝任选一个。"
        if completed.returncode != 0:
            row["status"] = "failed"
        write_json(report_path, report)
    return report


def load_product_predictions(job_root):
    """按时间线事件而非 MP4 文件计数，避免一事件多素材重复得分。"""
    timeline = read_json(job_root / "event_timeline.json")
    identity_path = job_root / "event_identity.json"
    identities = read_json(identity_path).get("event_resolutions", []) if identity_path.exists() else []
    by_id = {row["event_id"]: row for row in identities}
    predictions = []
    for row in timeline["events"]:
        label = event_name(row["event"])
        if label == "blank":
            continue
        identity = by_id.get(row["event_id"], {})
        color, number = identity.get("jersey_color"), identity.get("jersey_number")
        actor = (f"{str(color).strip().lower()}#{number}"
                 if identity.get("status") == "identified" and color and number is not None else None)
        start, end = float(row["evidence_start_seconds"]), float(row["evidence_end_seconds"])
        if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
            raise ValueError("时间线包含无效事件区间")
        predictions.append({"event": label, "actor": actor, "event_id": row["event_id"],
                            "interval": [start, end], "confidence": row["confidence"],
                            "identity_status": identity.get("status", "missing")})
    # 时间是模型输出；绝不依照 GT 顺序重排。相同中点用 event_id 固定次序。
    return sorted(predictions, key=lambda row: (sum(row["interval"]) / 2, row["event_id"]))


def validate_annotations(manifest, annotations, allow_draft=False):
    """正式评分要求每个冻结样本都人工核验，不能只挑识别成功的样本。"""
    rows = annotations["samples"]
    by_id = {row["sample_id"]: row for row in rows}
    expected = {row["sample_id"] for row in manifest["items"]}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise ValueError("人工标注的样本集合与冻结清单不一致，或有重复 ID。")
    for row in rows:
        if not allow_draft and (row.get("reviewed") is not True or not row.get("reviewer", "").strip()):
            raise ValueError(f"尚未人工核验：{row['sample_id']}；草稿只能用 --allow-draft 演练。")
        for event in row["events"]:
            event["event"] = event_name(event["event"])
            if event["event"] == "blank" or not isinstance(event.get("actor"), str) or not event["actor"]:
                raise ValueError("GT events 只记录真实非背景事件，且必须明确参与人。")
    return by_id


def score_bundle(args):
    """全清单评分：未运行/失败样本算空预测，身份失败事件仍计入预测分母。"""
    manifest = read_json(args.bundle / "manifest.json")
    annotations_path = args.annotations or args.bundle / "annotations.json"
    silver = getattr(args, "accept_bard_silver", False)
    if silver and args.allow_draft:
        raise ValueError("--accept-bard-silver 与 --allow-draft 不能同时使用。")
    truth = validate_annotations(
        manifest, read_json(annotations_path), args.allow_draft or silver
    )
    report_path = args.run_root / "run_report.json"
    report = read_json(report_path) if report_path.exists() else {"results": {}}
    if report.get("manifest_sha256") not in (None, sha256(args.bundle / "manifest.json")):
        raise ValueError("运行清单和评分清单不一致。")
    samples, failures, statuses, identity_counts = [], [], Counter(), Counter()
    samples_by_game = {game: [] for game in manifest.get("games", [])}
    for item in manifest["items"]:
        key = item["sample_id"]
        row = report["results"].get(key, {})
        statuses[row.get("status", "not_run")] += 1
        predictions = []
        try:
            if not row.get("job_root") or row.get("status") not in ("completed", "completed_with_warnings"):
                raise ValueError("任务未完成，按空预测计入评估")
            predictions = load_product_predictions(Path(row["job_root"]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append({"sample_id": key, "error": str(error)})
        identity_counts.update(p["identity_status"] for p in predictions)
        normalized = {"sample_id": key, "references": truth[key]["events"],
                      "predictions": predictions}
        samples.append(normalized)
        samples_by_game.setdefault(item.get("game", "unknown"), []).append(normalized)
    per_game = {
        game: {"sample_count": len(rows),
               "q8_adapted": None if silver else sequence_metrics(rows),
               "unordered_player_event_sets": player_pair_metrics(rows)}
        for game, rows in samples_by_game.items()
    }
    result = {"schema_version": "basketevent_bard_scores.v1",
              "claim": ("bard_silver_unordered_evaluation" if silver else
                        "draft_only_not_valid_evaluation" if args.allow_draft else
                        "custom_bard_product_evaluation"),
              "protocol": ("BARD silver unordered player-event sets; no sequence claim" if silver else
                           "Q8 LCS formulas; color-number instead of official team-number; custom BARD clips"),
              "official_benchmark_reproduction": False,
              "manifest_sha256": sha256(args.bundle / "manifest.json"),
              "annotations_sha256": sha256(annotations_path),
              "sample_count": len(samples), "run_statuses": dict(statuses),
              "identity_statuses": dict(identity_counts), "failed_or_missing": failures,
              "q8_adapted": (None if silver else sequence_metrics(samples)),
              "unordered_player_event_sets": player_pair_metrics(samples),
              "per_game": per_game,
              "normalized_samples": samples,
              "temporal_evaluation": "not_computed: product editing ranges are not paper gate targets"}
    default_name = ("scores_silver.json" if silver else
                    "scores_draft.json" if args.allow_draft else "scores.json")
    output = args.output or args.run_root / default_name
    write_json(output, result)
    summary = {"output": str(output), **{key: result[key] for key in
            ("claim", "sample_count", "run_statuses", "identity_statuses")},
            "unordered_player_event_sets": result["unordered_player_event_sets"]}
    if result["q8_adapted"] is not None:
        summary.update(event_type=result["q8_adapted"]["event_type"],
                       full_event=result["q8_adapted"]["full_event"],
                       participant_accuracy=result["q8_adapted"]["participant_accuracy"])
    return summary


def score_tracks(args):
    """模型级人工轨迹标签评估；不依赖 Qwen，不能与产品端到端分数混报。

    输入 samples 每项包含 sample_id、prediction_json 和 references。
    reference.actor 是人工确认的 player_N；没追踪到的人用 missing_N，保留漏检。
    reference.interval 可缺省；存在时参与论文最高 gate 的 Hit 评估。
    """
    document = read_json(args.targets)
    if document.get("reviewed") is not True or not document.get("reviewer"):
        raise ValueError("轨迹 GT 必须经过人工核验并填写 reviewer。")
    samples, temporal, errors = [], [], []
    for sample in document["samples"]:
        references = sample["references"]
        for reference in references:
            reference["event"] = event_name(reference["event"])
            if not reference.get("actor") or reference["event"] == "blank":
                raise ValueError("轨迹 GT 必须给出 actor 且仅含非背景事件。")
        try:
            prediction = read_json(args.targets.parent / sample["prediction_json"])
        except (OSError, ValueError) as error:
            prediction = {"player_predictions": []}
            errors.append({"sample_id": sample["sample_id"], "error": str(error)})
        pred, ranked, by_track = [], [], {}
        for row in prediction["player_predictions"]:
            actor = row["player_id"]
            label = event_name(row["event"])
            if "class_probabilities" not in row or "paper_gate_segment" not in row:
                raise ValueError("旧缓存缺少完整概率/最高 gate 字段；须重新运行 PlayNet，不能伪造 Recall/Hit。")
            by_track[actor] = row
            if label != "blank":
                pred.append({"actor": actor, "event": label})
            for name, confidence in row["class_probabilities"].items():
                if event_name(name) != "blank":
                    ranked.append({"actor": actor, "event": event_name(name), "confidence": confidence})
        samples.append({"sample_id": sample["sample_id"], "references": references,
                        "predictions": pred, "ranked_pairs": ranked})
        for ref in references:
            if not ref.get("interval"):
                continue
            row = by_track.get(ref["actor"], {})
            gate = row.get("paper_gate_segment")
            if row and gate is None:
                raise ValueError("轨迹没有 gate 证据，无法按论文做时间评估。")
            temporal.append({"reference": ref, "prediction": None if not gate else
                             {"event": row["event"], "interval": [gate["start_time"], gate["end_time"]]}})
    result = {"protocol": "BasketEvent formulas; manually verified track actors; per-video global top-K",
              "official_benchmark_reproduction": False, "errors_counted_as_missing": errors,
              "player_event": player_pair_metrics(samples), "temporal": temporal_hits(temporal)}
    write_json(args.output, result)
    return result


def positive_integer(value):
    """CLI 中数量参数必须严格大于零。"""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main(argv=None):
    """保持构建、传输校验、运行和评分为独立步骤，便于逐步检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--games", nargs="+")
    build.add_argument("--game-count", type=positive_integer, default=6)
    build.add_argument("--per-class", type=positive_integer, default=3)
    build.add_argument("--seed", type=int, default=20260905)
    build.add_argument("--copy-videos", action="store_true")
    build.add_argument(
        "--all-clips", action="store_true",
        help="保留显式指定比赛的全部片段，包括没有映射事件的片段。",
    )
    copy_command = commands.add_parser("copy")
    copy_command.add_argument("--bundle", type=Path, required=True)
    copy_command.add_argument("--method", choices=("copy", "hardlink"), default="copy")
    commands.add_parser("verify").add_argument("--bundle", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument(
        "--gpu", choices=[str(i) for i in range(16)], default=None,
        help="兼容旧命令：未分别指定设备时，三个阶段共同使用这张逻辑 GPU。",
    )
    run.add_argument(
        "--sam3-gpus", default=None,
        help="SAM3 使用的逻辑 GPU 列表，例如 0,1。",
    )
    run.add_argument(
        "--playnet-gpu", type=int, choices=range(16), default=None,
        help="PlayNet 使用的单张逻辑 GPU。",
    )
    run.add_argument(
        "--identity-gpus", default=None,
        help="Qwen 身份阶段使用的逻辑 GPU 列表。",
    )
    run.add_argument("--ffmpeg-binary", default="ffmpeg")
    run.add_argument("--limit", type=positive_integer)
    run.add_argument("--pipeline-mode", choices=("product", "clip"), default="product")
    score = commands.add_parser("score")
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--run-root", type=Path, required=True)
    score.add_argument("--annotations", type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--allow-draft", action="store_true")
    score.add_argument(
        "--accept-bard-silver", action="store_true",
        help="用未人工核验的 BARD 映射做无序集合评估；不计算 Q8 顺序指标。",
    )
    tracks = commands.add_parser("score-tracks")
    tracks.add_argument("--targets", type=Path, required=True)
    tracks.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    handlers = {"build": build_bundle, "run": run_bundle, "score": score_bundle,
                "score-tracks": score_tracks,
                "copy": lambda a: materialize_bundle(a.bundle, a.method),
                "verify": lambda a: verify_bundle(a.bundle)}
    print(json.dumps(handlers[args.command](args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
