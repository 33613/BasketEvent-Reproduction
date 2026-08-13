"""Minimal, checkpoint-free verification of BasketEvent feature extraction.

This script validates only the tensor path from one video clip to TimeSformer
features, ROIAlign features, bbox/type embeddings, global tokens, and enhanced
local tokens. It does not claim meaningful event predictions because the
BasketEvent-trained PlayerEventModel checkpoint is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import build_clips_from_video, load_trajectory_json
from src.dataset import LABEL_MAP
from src.model import PlayerEventModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify TimeSformer -> ROIAlign -> bbox embedding -> global/local tokens"
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "examples" / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4"),
    )
    parser.add_argument(
        "--traj_json",
        default=str(PROJECT_ROOT / "examples" / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json"),
    )
    parser.add_argument("--player_id", default="player_0")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--fps_in", type=int, default=25)
    parser.add_argument("--fps_out", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def report(name: str, tensor: torch.Tensor) -> None:
    finite = bool(torch.isfinite(tensor).all().item())
    print(
        f"{name:28s} shape={tuple(tensor.shape)!s:24s} "
        f"dtype={str(tensor.dtype):14s} finite={finite}"
    )
    if not finite:
        raise RuntimeError(f"{name} contains NaN or Inf")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test expects CUDA, but CUDA is unavailable")

    device = torch.device("cuda:0")
    traj_data = load_trajectory_json(args.traj_json)
    if args.player_id not in traj_data:
        raise KeyError(f"Unknown player_id {args.player_id!r}")

    data = build_clips_from_video(
        video_path=args.video,
        traj_data=traj_data,
        player_ids=[args.player_id],
        starts=[args.start],
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        size=args.img_size,
        fmt="xywh",
    )

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"sampled_frame_indices={data['idx'][0].tolist()}")
    report("input clip", data["clips_video"])
    report("player boxes", data["clips_bboxes"])
    report("ball boxes", data["clips_ball"])

    model = PlayerEventModel(
        num_classes=len(LABEL_MAP),
        image_size=args.img_size,
        roi_out_size=(1, 1),
        pooling_mode="gated",
    ).eval().to(device)

    clips_video = data["clips_video"].to(device)
    idx = data["idx"].to(device)
    nums = data["nums"].to(device)
    bboxes = [data["clips_bboxes"][0].to(device)]
    bbox_masks = [data["clips_bbox_mask"][0].to(device)]
    clips_ball = data["clips_ball"].to(device)
    clips_ball_mask = data["clips_ball_mask"].to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        extended_boxes, extended_masks = model._build_extended_boxes_with_ball(
            bboxes, bbox_masks, clips_ball, clips_ball_mask
        )

        backbone_out = model.backbone(
            clips_video, idx=idx, nums=nums, fps_in=float(args.fps_in)
        )
        featmap = backbone_out["featmap"]
        report("global feature map", featmap)

        roi_flat, roi_mask, _ = model.roi(featmap, extended_boxes, extended_masks)
        report("ROIAlign output", roi_flat)
        report("ROI valid mask", roi_mask)

        roi_projected = model.roi_proj(roi_flat)
        bbox_cat = torch.cat(extended_boxes, dim=0).to(torch.float32)
        mask_cat = torch.cat(extended_masks, dim=0).to(torch.float32)
        bbox_embedding = model.bbox_emb(bbox_cat, mask_cat)
        report("bbox embedding", bbox_embedding)

        local_flat = roi_projected + bbox_embedding
        type_ids = torch.zeros(local_flat.shape[:2], device=device, dtype=torch.long)
        type_ids[-1, :] = 1  # the final instance is the virtual ball token
        local_flat = model.type_emb(local_flat, type_ids)

        m, _, t, _, _ = featmap.shape
        n_plus_ball = int(local_flat.shape[0] // m)
        c = int(local_flat.shape[-1])
        local_tokens = local_flat.view(m, n_plus_ball, t, c)
        local_mask = roi_mask.view(m, n_plus_ball, t)
        report("local tokens", local_tokens)

        featmap_for_global = model.featmap_proj(featmap)
        _, c_global, _, hf, wf = featmap_for_global.shape
        global_tokens = (
            featmap_for_global.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(m * t, hf * wf, c_global)
        )
        report("global tokens", global_tokens)

        enhanced_local, attn = model.actor_global(
            person_feats=local_tokens,
            featmap=featmap_for_global,
            person_mask=local_mask,
            return_attn=True,
        )
        report("enhanced local tokens", enhanced_local)
        report("actor-global attention", attn)

        valid_attn = attn[local_mask.permute(0, 2, 1).bool()]
        if valid_attn.numel() > 0:
            sums = valid_attn.flatten(1).sum(dim=1)
            print(
                "attention_sum_range="
                f"[{sums.min().item():.6f}, {sums.max().item():.6f}]"
            )

    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"peak_cuda_memory_allocated={peak_gib:.2f} GiB")
    print("FEATURE PIPELINE SMOKE TEST PASSED")
    print(
        "Note: TimeSformer uses K400 pretrained weights, but bbox/type embeddings "
        "and actor-global attention remain randomly initialized without the "
        "authors' BasketEvent checkpoint. This validates execution and shapes, "
        "not paper-level event semantics."
    )


if __name__ == "__main__":
    main()
