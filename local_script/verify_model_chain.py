"""Checkpoint-free shape test for the complete BasketEvent method chain.

The test follows the paper notation from context-enhanced entity tokens Z_tilde
through entity interaction, temporal pooling, cross-clip interaction, gated
pooling, player representation h_i, and logits l_i. Randomly initialized
BasketEvent-specific layers make the values semantically meaningless; only
execution, tensor shapes, masks, finite values, and memory use are verified.
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
        description="Verify Z_tilde -> entity interaction -> pooling -> h_i -> logits"
    )
    parser.add_argument(
        "--video",
        default=str(PROJECT_ROOT / "examples" / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4"),
    )
    parser.add_argument(
        "--traj_json",
        default=str(PROJECT_ROOT / "examples" / "4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json"),
    )
    parser.add_argument(
        "--player_ids",
        default="",
        help="Comma-separated player IDs; empty uses all players in the clean JSON",
    )
    parser.add_argument("--bag_clips", type=int, default=2)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--fps_in", type=int, default=25)
    parser.add_argument("--fps_out", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def report(name: str, tensor: torch.Tensor) -> None:
    finite = bool(torch.isfinite(tensor).all().item())
    print(
        f"{name:30s} shape={tuple(tensor.shape)!s:24s} "
        f"dtype={str(tensor.dtype):14s} finite={finite}"
    )
    if not finite:
        raise RuntimeError(f"{name} contains NaN or Inf")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local smoke test")
    if args.bag_clips < 2:
        raise ValueError("bag_clips must be at least 2 to verify cross-clip interaction")

    device = torch.device("cuda:0")
    traj_data = load_trajectory_json(args.traj_json)
    all_player_ids = sorted(pid for pid in traj_data if pid != "ball")
    if args.player_ids.strip():
        player_ids = [x.strip() for x in args.player_ids.split(",") if x.strip()]
    else:
        player_ids = all_player_ids
    missing = [pid for pid in player_ids if pid not in traj_data]
    if missing:
        raise KeyError(f"Unknown player IDs: {missing}")

    first_traj = traj_data[player_ids[0]].get("trajectory", [])
    total_frames = len(first_traj)
    stride = max(1, int(round(float(args.fps_in) / float(args.fps_out))))
    clip_span = (args.clip_len - 1) * stride + 1
    max_start = max(0, total_frames - clip_span)
    starts = torch.linspace(0, max_start, args.bag_clips).round().long().tolist()

    data = build_clips_from_video(
        video_path=args.video,
        traj_data=traj_data,
        player_ids=player_ids,
        starts=[int(x) for x in starts],
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        size=args.img_size,
        fmt="xywh",
    )

    model = PlayerEventModel(
        num_classes=len(LABEL_MAP),
        image_size=args.img_size,
        roi_out_size=(1, 1),
        pooling_mode="gated",
    ).eval().to(device)

    clips_video = data["clips_video"].to(device)
    idx = data["idx"].to(device)
    nums = data["nums"].to(device)
    bboxes = [data["clips_bboxes"][m].to(device) for m in range(args.bag_clips)]
    bbox_masks = [data["clips_bbox_mask"][m].to(device) for m in range(args.bag_clips)]
    clips_ball = data["clips_ball"].to(device)
    clips_ball_mask = data["clips_ball_mask"].to(device)

    m = args.bag_clips
    n = len(player_ids)
    n_plus_ball = n + 1

    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"players={player_ids}")
    print(f"clip_starts={starts}")
    report("input clips", clips_video)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        extended_boxes, extended_masks = model._build_extended_boxes_with_ball(
            bboxes, bbox_masks, clips_ball, clips_ball_mask
        )
        featmap = model.backbone(
            clips_video, idx=idx, nums=nums, fps_in=float(args.fps_in)
        )["featmap"]

        roi_flat, mask_flat, _ = model.roi(featmap, extended_boxes, extended_masks)
        roi_flat = model.roi_proj(roi_flat)
        entity_flat = model._add_bbox_and_type_embedding(
            person_feats_flat=roi_flat,
            extended_bboxes=extended_boxes,
            extended_bbox_masks=extended_masks,
            M=m,
            N_plus_1=n_plus_ball,
            T=args.clip_len,
        )
        z = entity_flat.view(m, n_plus_ball, args.clip_len, model.mil_feat_dim)
        entity_mask = mask_flat.view(m, n_plus_ball, args.clip_len)

        featmap_for_global = model.featmap_proj(featmap)
        z_tilde = model.actor_global(
            person_feats=z,
            featmap=featmap_for_global,
            person_mask=entity_mask,
            return_attn=False,
        )
        report("Z_tilde after G-P attention", z_tilde)

        frame_tokens = z_tilde.permute(0, 2, 1, 3).contiguous().view(
            m * args.clip_len, n_plus_ball, model.mil_feat_dim
        )
        frame_mask = entity_mask.permute(0, 2, 1).contiguous().view(
            m * args.clip_len, n_plus_ball
        )
        frame_tokens = model.person_relation(frame_tokens, valid_mask=frame_mask)
        z_bar = frame_tokens.view(
            m, args.clip_len, n_plus_ball, model.mil_feat_dim
        ).permute(0, 2, 1, 3).contiguous()
        z_bar = z_bar * entity_mask.to(z_bar.dtype).unsqueeze(-1)
        report("Z_bar after entity attention", z_bar)

        c_flat = model.feature_pooler(
            z_bar.view(m * n_plus_ball, args.clip_len, model.mil_feat_dim),
            entity_mask.view(m * n_plus_ball, args.clip_len),
        )
        c = c_flat.view(m, n_plus_ball, model.mil_feat_dim)
        clip_valid = entity_mask.any(dim=2)
        report("c_mi after temporal pooling", c)

        c_hat = model.clip_relation(c, valid_mask=clip_valid)
        report("C_hat after cross-clip", c_hat)

        player_c_hat = c_hat[:, :n, :]
        player_clip_valid = clip_valid[:, :n]
        h_i, gate_logits, alpha = model.clip_pool(
            player_c_hat,
            valid_mask=player_clip_valid,
            return_weights=True,
        )
        report("h_i player representation", h_i)
        report("alpha gated clip weights", alpha)

        person_valid = player_clip_valid.any(dim=0)
        logits = model.person_head(h_i, person_valid)
        probabilities = torch.softmax(logits, dim=-1)
        report("l_i player logits", logits)
        report("pi_i class probabilities", probabilities)

        alpha_sum = alpha.squeeze(-1).sum(dim=0)
        probability_sum = probabilities.sum(dim=-1)
        if not torch.allclose(alpha_sum[person_valid], torch.ones_like(alpha_sum[person_valid]), atol=1e-5):
            raise RuntimeError("Valid-player gated clip weights do not sum to 1")
        if not torch.allclose(probability_sum, torch.ones_like(probability_sum), atol=1e-5):
            raise RuntimeError("Class probabilities do not sum to 1")
        print(
            f"alpha_sum_range=[{alpha_sum.min().item():.6f}, "
            f"{alpha_sum.max().item():.6f}]"
        )
        print(
            f"probability_sum_range=[{probability_sum.min().item():.6f}, "
            f"{probability_sum.max().item():.6f}]"
        )

    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"peak_cuda_memory_allocated={peak_gib:.2f} GiB")
    print("FULL METHOD CHAIN SHAPE TEST PASSED")
    print(
        "Checkpoint note: only TimeSformer has K400 pretrained weights. The "
        "BasketEvent-specific interaction, pooling, embedding, and classifier "
        "parameters are random, so logits are not event predictions."
    )


if __name__ == "__main__":
    main()
