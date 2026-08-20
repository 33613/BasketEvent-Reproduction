import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.io import read_video

# # Ensure root path contains src module when running from NBA or NBA/src.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from settings import SETTINGS
from src.dataset import load_ball_from_json_resized, load_bbox_from_json_resized_onepid
from src.labels import LABEL_MAP
from src.model import PlayerEventModel

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for PlayerEventModel from raw video + trajectories"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(
            SETTINGS.project_root
            / "examples"
            / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4"
        ),
        help="Path to the basketball video file",
    )
    parser.add_argument(
        "--traj_json",
        type=str,
        default=str(
            SETTINGS.project_root
            / "examples"
            / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json"
        ),
        help="Path to the JSON file with player trajectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SETTINGS.event_checkpoint),
        help="Path to the BasketEvent model checkpoint (.pt).",
    )
    parser.add_argument(
        "--timesformer_model",
        type=str,
        default=str(SETTINGS.timesformer_model),
        help="Path to the local TimeSformer model directory.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=int(SETTINGS.gpu_ids.split(",")[0]),
        help="Visible CUDA device index used for inference.",
    )
    parser.add_argument(
        "--player_ids",
        type=str,
        default="",
        help="Comma-separated list of player IDs to use. If empty, all players in the JSON (except 'ball') are used in sorted order.",
    )
    parser.add_argument(
        "--starts",
        type=str,
        default="0",
        help="Comma-separated start indices for clip sampling. Default is 0.",
    )
    parser.add_argument(
        "--bag_clips",
        type=int,
        default=12,
        help="Number of clips (M). If starts is provided, it overrides bag_clips.",
    )
    parser.add_argument(
        "--clip_len",
        type=int,
        default=8,
        help="Number of frames per clip after sampling",
    )
    parser.add_argument(
        "--fps_in", type=int, default=25, help="Nominal original video FPS"
    )
    parser.add_argument(
        "--fps_out", type=int, default=4, help="Target effective FPS for model input"
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image size for the model"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Top-K predictions to print"
    )
    parser.add_argument(
        "--prediction_json_path",
        type=str,
        default="",
        help=(
            "Optional JSON output containing player predictions and temporal "
            "evidence windows for visualization."
        ),
    )
    parser.add_argument(
        "--timeline_topk",
        type=int,
        default=2,
        help="Number of strongest temporal evidence clips exported per event player.",
    )
    parser.add_argument(
        "--traj_format",
        type=str,
        default="xywh",
        choices=["xywh", "xyxy"],
        help="Trajectory format in JSON",
    )
    return parser.parse_args()


def normalize_frames(frames: torch.Tensor) -> torch.Tensor:
    # Accept frames either (T,H,W,C) or (T,C,H,W). Return (T,C,H,W) normalized.
    if frames.dim() != 4:
        raise ValueError(f"frames must be 4D tensor, got shape {tuple(frames.shape)}")

    # convert (T,H,W,C) -> (T,C,H,W)
    if frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2).contiguous()

    C = frames.shape[1]
    if C != 3:
        raise ValueError(f"expected 3 channels, got {C}")

    mean = torch.tensor(
        [0.45, 0.45, 0.45], dtype=frames.dtype, device=frames.device
    ).view(1, C, 1, 1)
    std = torch.tensor(
        [0.225, 0.225, 0.225], dtype=frames.dtype, device=frames.device
    ).view(1, C, 1, 1)
    return (frames - mean) / std


def build_clip_indices(
    start: int, clip_len: int, stride: int, nums: int
) -> torch.Tensor:
    idx = torch.arange(clip_len, dtype=torch.long) * stride + int(start)
    idx = torch.clamp(idx, 0, nums - 1)
    return idx


def load_trajectory_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Trajectory JSON must contain a dict at top level, got {type(data)}"
        )
    return data


def normalize_traj_entry(entry: Any, fmt: str = "xywh") -> Optional[List[float]]:
    if entry is None:
        return None
    if not isinstance(entry, list) or len(entry) != 4:
        return None
    if fmt == "xywh":
        return [float(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])]
    # xyxy -> xywh
    x1, y1, x2, y2 = map(float, entry)
    return [x1, y1, x2 - x1, y2 - y1]


def build_player_tracks(
    traj_data: Dict[str, Any],
    player_ids: List[str],
    kept_indices: List[int],
    scale_x: float,
    scale_y: float,
    fmt: str = "xywh",
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = len(player_ids)
    T = len(kept_indices)
    bboxes = torch.zeros((N, T, 4), dtype=torch.float32)
    mask = torch.zeros((N, T), dtype=torch.float32)

    for pi, pid in enumerate(player_ids):
        traj_item = traj_data.get(str(pid), {})
        traj_list = traj_item.get("trajectory") if isinstance(traj_item, dict) else None
        if traj_list is None:
            traj_list = []

        for ti, frame_index in enumerate(kept_indices):
            if frame_index < 0 or frame_index >= len(traj_list):
                continue
            raw_box = normalize_traj_entry(traj_list[frame_index], fmt)
            if raw_box is None:
                continue
            x, y, w, h = raw_box
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y
            bboxes[pi, ti] = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
            mask[pi, ti] = 1.0

    return bboxes, mask


def build_ball_track(
    traj_data: Dict[str, Any],
    kept_indices: List[int],
    scale_x: float,
    scale_y: float,
    fmt: str = "xywh",
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = len(kept_indices)
    bboxes = torch.zeros((T, 4), dtype=torch.float32)
    mask = torch.zeros((T,), dtype=torch.float32)

    ball_item = traj_data.get("ball", {})
    traj_list = ball_item.get("trajectory") if isinstance(ball_item, dict) else None
    if traj_list is None:
        return bboxes, mask

    for ti, frame_index in enumerate(kept_indices):
        if frame_index < 0 or frame_index >= len(traj_list):
            continue
        raw_box = normalize_traj_entry(traj_list[frame_index], fmt)
        if raw_box is None:
            continue
        x, y, w, h = raw_box
        x *= scale_x
        y *= scale_y
        w *= scale_x
        h *= scale_y
        bboxes[ti] = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)
        mask[ti] = 1.0

    return bboxes, mask


def build_clips_from_video(
    video_path: str,
    traj_data: Dict[str, Any],
    player_ids: List[str],
    starts: List[int],
    clip_len: int,
    fps_in: int,
    fps_out: int,
    size: int,
    fmt: str = "xywh",
) -> Dict[str, Any]:
    video_all, _, _ = read_video(video_path, pts_unit="sec")
    total_frames = int(video_all.shape[0])
    if total_frames == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    orig_h = int(video_all.shape[1])
    orig_w = int(video_all.shape[2])
    scale_x = float(size) / float(orig_w)
    scale_y = float(size) / float(orig_h)

    sample_stride_frames = max(1, int(round(float(fps_in) / float(fps_out))))
    clip_offsets = torch.arange(clip_len, dtype=torch.long) * sample_stride_frames

    M = len(starts)
    N = len(player_ids)
    T = clip_len

    clips_video = torch.zeros((M, 3, T, size, size), dtype=torch.float32)
    clips_bboxes = torch.zeros((M, N, T, 4), dtype=torch.float32)
    clips_bbox_mask = torch.zeros((M, N, T), dtype=torch.float32)
    clips_ball = torch.zeros((M, T, 4), dtype=torch.float32)
    clips_ball_mask = torch.zeros((M, T), dtype=torch.float32)
    idx = torch.zeros((M, T), dtype=torch.long)

    for mi, start in enumerate(starts):
        kept = torch.clamp(start + clip_offsets, 0, total_frames - 1)
        idx[mi] = kept

        frames = video_all[kept].float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(
            frames, size=(size, size), mode="bilinear", align_corners=False
        )
        frames = normalize_frames(frames)
        frames = frames.permute(1, 0, 2, 3).contiguous()
        clips_video[mi] = frames

        bboxes, mask = build_player_tracks(
            traj_data, player_ids, kept.tolist(), scale_x, scale_y, fmt
        )
        clips_bboxes[mi] = bboxes
        clips_bbox_mask[mi] = mask

        ball_bboxes, ball_mask = build_ball_track(
            traj_data, kept.tolist(), scale_x, scale_y, fmt
        )
        clips_ball[mi] = ball_bboxes
        clips_ball_mask[mi] = ball_mask

    nums = torch.tensor([total_frames] * M, dtype=torch.long)
    return {
        "clips_video": clips_video,
        "clips_bboxes": clips_bboxes,
        "clips_bbox_mask": clips_bbox_mask,
        "clips_ball": clips_ball,
        "clips_ball_mask": clips_ball_mask,
        "idx": idx,
        "nums": nums,
        "player_ids": player_ids,
        "fps_in": float(fps_in),
        "total_frames": total_frames,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    new_state_dict: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module.") :]] = v
        else:
            new_state_dict[k] = v
    incompatible = model.load_state_dict(new_state_dict, strict=False)
    if incompatible.missing_keys:
        print(
            f"[Checkpoint] missing keys ({len(incompatible.missing_keys)}): "
            f"{incompatible.missing_keys[:20]}"
        )
    if incompatible.unexpected_keys:
        print(
            f"[Checkpoint] unexpected keys ({len(incompatible.unexpected_keys)}): "
            f"{incompatible.unexpected_keys[:20]}"
        )

    metadata = ckpt if isinstance(ckpt, dict) else {}
    return metadata.get("epoch"), metadata.get("global_step")


@torch.no_grad()
def infer_one_video(
    model: PlayerEventModel,
    data: Dict[str, Any],
    device: torch.device,
    topk: int = 5,
    return_weights: bool = False,
) -> Dict[str, Any]:
    model.eval()
    clips_video = data["clips_video"].to(device)
    idx = data["idx"].to(device)
    nums = data["nums"].to(device)
    clips_bboxes = [
        data["clips_bboxes"][m].to(device) for m in range(data["clips_bboxes"].shape[0])
    ]
    clips_bbox_mask = [
        data["clips_bbox_mask"][m].to(device)
        for m in range(data["clips_bbox_mask"].shape[0])
    ]
    clips_ball = data["clips_ball"].to(device)
    clips_ball_mask = data["clips_ball_mask"].to(device)

    out = model.forward_mil_one_video(
        clips_video=clips_video,
        idx=idx,
        nums=nums,
        bboxes=clips_bboxes,
        bbox_masks=clips_bbox_mask,
        clips_ball=clips_ball,
        clips_ball_mask=clips_ball_mask,
        fps_in=data["fps_in"],
        topk=topk,
        return_weights=return_weights,
    )

    return out


def build_temporal_prediction_report(
    *,
    video_path: str,
    trajectory_data: Dict[str, Any],
    player_ids: List[str],
    data: Dict[str, Any],
    outputs: Dict[str, Any],
    timeline_topk: int,
) -> Dict[str, Any]:
    """Convert model outputs into video-level and time-aligned predictions.

    The event class comes from the final player-level head. Temporal evidence
    is ranked with the learned MIL gate weight multiplied by the clip-level
    probability of that final event class. This produces diagnostic evidence
    windows; it is not a separately supervised event-boundary detector.

    Args:
        video_path: Source video path recorded in the report.
        trajectory_data: Clean trajectory document containing jersey metadata.
        player_ids: Player IDs in the exact order passed to the model.
        data: Prepared inference data containing sampled source-frame indices.
        outputs: Model outputs returned by :func:`infer_one_video`.
        timeline_topk: Maximum evidence windows retained for each non-background
            player prediction.

    Returns:
        JSON-serializable prediction report consumed by the visualization
        script.

    Raises:
        ValueError: If ``timeline_topk`` is not positive.
    """
    if timeline_topk < 1:
        raise ValueError("timeline_topk must be at least 1")

    logits_person = outputs["logits_person"].detach().cpu()
    logits_clip = outputs["logits_clip"].detach().cpu()
    person_valid = outputs["person_valid"].detach().cpu().bool()
    person_clip_valid = outputs["person_clip_valid"].detach().cpu().bool()
    gate_weights = outputs.get("gate_weights")
    if gate_weights is not None:
        gate_weights = gate_weights.detach().cpu().squeeze(-1)

    person_probabilities = torch.softmax(logits_person, dim=-1)
    clip_probabilities = torch.softmax(logits_clip, dim=-1)
    sampled_indices = data["idx"].detach().cpu()
    fps = float(data["fps_in"])
    total_frames = int(data["total_frames"])
    duration_seconds = total_frames / fps

    player_predictions: List[Dict[str, Any]] = []
    temporal_events: List[Dict[str, Any]] = []
    for person_index, player_id in enumerate(player_ids):
        if not person_valid[person_index].item():
            continue

        player_probability = person_probabilities[person_index]
        label_id = int(player_probability.argmax().item())
        event_name = REVERSE_LABEL_MAP.get(label_id, str(label_id))
        confidence = float(player_probability[label_id].item())
        metadata = trajectory_data.get(player_id, {})
        if not isinstance(metadata, dict):
            metadata = {}
        jersey_number = metadata.get("jersey_number")
        jersey_color = metadata.get("jersey_color")
        player_predictions.append(
            {
                "player_id": player_id,
                "jersey_number": jersey_number,
                "jersey_color": jersey_color,
                "label_id": label_id,
                "event": event_name,
                "confidence": confidence,
            }
        )

        # Class zero is the background/blank event and has no timeline segment.
        if label_id == 0:
            continue

        clip_class_probability = clip_probabilities[:, person_index, label_id]
        if gate_weights is None:
            temporal_score = clip_class_probability.clone()
        else:
            temporal_score = gate_weights[:, person_index] * clip_class_probability
        temporal_score = temporal_score.masked_fill(
            ~person_clip_valid[:, person_index], float("-inf")
        )
        valid_count = int(person_clip_valid[:, person_index].sum().item())
        keep_count = min(timeline_topk, valid_count)
        if keep_count == 0:
            continue

        selected_clips = torch.topk(temporal_score, k=keep_count).indices.tolist()
        for clip_index in sorted(selected_clips):
            clip_frames = sampled_indices[clip_index]
            start_frame = int(clip_frames.min().item())
            end_frame = int(clip_frames.max().item()) + 1
            gate_weight = (
                float(gate_weights[clip_index, person_index].item())
                if gate_weights is not None
                else None
            )
            temporal_events.append(
                {
                    "player_id": player_id,
                    "jersey_number": jersey_number,
                    "jersey_color": jersey_color,
                    "label_id": label_id,
                    "event": event_name,
                    "confidence": confidence,
                    "clip_index": int(clip_index),
                    "clip_probability": float(
                        clip_class_probability[clip_index].item()
                    ),
                    "gate_weight": gate_weight,
                    "temporal_score": float(temporal_score[clip_index].item()),
                    "start_frame": start_frame,
                    "end_frame": min(end_frame, total_frames),
                    "start_time": start_frame / fps,
                    "end_time": min(end_frame, total_frames) / fps,
                }
            )

    return {
        "schema_version": "basketevent_temporal_predictions.v1",
        "video": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "localization_method": (
            "top-k clips ranked by MIL gate weight multiplied by the "
            "clip probability of the final player-level event"
        ),
        "localization_is_diagnostic": True,
        "player_predictions": player_predictions,
        "temporal_events": temporal_events,
    }


def write_prediction_report(path: str, report: Dict[str, Any]) -> None:
    """Write a UTF-8 temporal prediction report.

    Args:
        path: Destination JSON path.
        report: JSON-serializable result from
            :func:`build_temporal_prediction_report`.
    """
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    args.video = str(SETTINGS.require_file(args.video, "Input video"))
    args.traj_json = str(SETTINGS.require_file(args.traj_json, "Clean trajectory JSON"))
    args.checkpoint = str(
        SETTINGS.require_file(args.checkpoint, "BasketEvent checkpoint")
    )
    args.timesformer_model = str(
        SETTINGS.require_directory(
            args.timesformer_model,
            "TimeSformer model directory",
        )
    )

    traj_data = load_trajectory_json(args.traj_json)
    all_player_ids = [pid for pid in traj_data.keys() if pid != "ball"]
    if len(all_player_ids) == 0:
        raise ValueError("Trajectory JSON must contain at least one player trajectory")

    if args.player_ids:
        player_ids = [pid.strip() for pid in args.player_ids.split(",") if pid.strip()]
    else:
        player_ids = sorted(all_player_ids)

    starts: List[int]
    if args.starts.strip() == "":
        starts = [0]
    else:
        starts = [int(s) for s in args.starts.split(",") if s.strip()]
    if len(starts) == 0:
        starts = [0]

    # If user asks for multiple clips but only provides one start, spread starts evenly over the video.
    if len(starts) == 1 and args.bag_clips > 1:
        sample_stride_frames = max(
            1, int(round(float(args.fps_in) / float(args.fps_out)))
        )
        clip_span = (args.clip_len - 1) * sample_stride_frames + 1
        total_frames = int(read_video(args.video, pts_unit="sec")[0].shape[0])
        max_start = max(0, total_frames - clip_span)
        if args.bag_clips == 1:
            starts = [max_start // 2]
        else:
            starts = (
                torch.linspace(0, max_start, args.bag_clips).round().long().tolist()
            )
            starts = [
                int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in starts
            ]

    data = build_clips_from_video(
        video_path=args.video,
        traj_data=traj_data,
        player_ids=player_ids,
        starts=starts,
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        size=args.img_size,
        fmt=args.traj_format,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABEL_MAP)
    model = PlayerEventModel(
        num_classes=num_classes,
        pretrained_name=args.timesformer_model,
        local_files_only=SETTINGS.hf_local_files_only,
        image_size=args.img_size,
        roi_out_size=(1, 1),
        pooling_mode="gated",
    ).to(device)

    if args.checkpoint:
        epoch, gs = load_checkpoint(args.checkpoint, model, device)
        print(f"Loaded checkpoint {args.checkpoint} epoch={epoch} global_step={gs}")
    else:
        print("No checkpoint specified; using TimeSformer pretrained weights only.")

    out = infer_one_video(
        model,
        data,
        device=device,
        topk=args.topk,
        return_weights=bool(args.prediction_json_path),
    )

    logits_person = out["logits_person"].cpu()
    person_valid = out["person_valid"].cpu().bool()

    print(f"Inference result for video={args.video}")
    print("non-background player-event pairs:")
    result_pairs = []
    for i, pid in enumerate(player_ids):
        if not person_valid[i].item():
            continue
        pred_label = int(logits_person[i].argmax().item())
        if pred_label == 0:
            continue
        event_name = REVERSE_LABEL_MAP.get(pred_label, str(pred_label))
        result_pairs.append((pid, event_name))

    if len(result_pairs) == 0:
        print("  none")
    else:
        for pid, event_name in result_pairs:
            print(f"  player={pid}: event={event_name}")

    if args.prediction_json_path:
        report = build_temporal_prediction_report(
            video_path=args.video,
            trajectory_data=traj_data,
            player_ids=player_ids,
            data=data,
            outputs=out,
            timeline_topk=args.timeline_topk,
        )
        write_prediction_report(args.prediction_json_path, report)
        print(f"Temporal prediction report: {args.prediction_json_path}")


if __name__ == "__main__":
    main()
