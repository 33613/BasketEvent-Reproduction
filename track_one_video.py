"""Track basketball players and balls in one video with local SAM3.

The entry point supports optional CPU offloading for video frames and SAM3
inference state.  CPU offloading is useful on pre-Ampere GPUs, where Flash
Attention is unavailable and long clips can otherwise exhaust GPU memory.
"""

import os
import json
import argparse
import numpy as np
from sam3.sam3.model_builder import build_sam3_video_predictor
from sam3.sam3.visualization_utils import prepare_masks_for_visualization
import pandas as pd
import torch
import gc

from settings import SETTINGS
from src.sam3_compat import install_object_limit_request


# SAM3 worker processes use multiprocessing ``spawn`` and import this module.
# Installing the request hook here makes the same configuration request
# available in the parent process and every GPU worker before model creation.
install_object_limit_request(build_sam3_video_predictor)


def mask_to_bbox(mask: np.ndarray):
    """Convert a binary mask to an ``[x, y, width, height]`` box.

    Args:
        mask: Two-dimensional binary or score mask.

    Returns:
        The smallest integer bounding box containing nonzero pixels, or
        ``None`` when the mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def get_frame_dict(outputs_per_frame, frame_idx):
    """Return one frame output while accepting integer or string JSON keys.

    Args:
        outputs_per_frame: Mapping from frame identifiers to object masks.
        frame_idx: Zero-based frame index.

    Returns:
        Object-mask mapping for the requested frame, or an empty mapping.
    """
    if frame_idx in outputs_per_frame:
        return outputs_per_frame[frame_idx]
    if str(frame_idx) in outputs_per_frame:
        return outputs_per_frame[str(frame_idx)]
    return {}


def collect_object_ids(outputs_per_frame):
    """Collect and numerically sort every object ID in frame outputs.

    Args:
        outputs_per_frame: Mapping from frame identifiers to object masks.

    Returns:
        Sorted integer object identifiers.
    """
    obj_ids = set()
    for _, obj_dict in outputs_per_frame.items():
        obj_ids.update(obj_dict.keys())
    return sorted(int(x) for x in obj_ids)


def build_trajectory_json(player_outputs, ball_outputs, json_path, num_frames=None):
    """Convert SAM3 player and ball masks into trajectory JSON.

    Args:
        player_outputs: Per-frame masks produced by the player text prompt.
        ball_outputs: Per-frame masks produced by the basketball text prompt.
        json_path: Destination JSON path.
        num_frames: Optional authoritative frame count. When omitted, the
            largest observed frame index determines the count.

    The output maps ``player_N`` and ``ball_N`` IDs to trajectories whose
    entries are ``[x, y, width, height]`` boxes or ``None``.
    """

    player_ids = collect_object_ids(player_outputs)
    ball_ids = collect_object_ids(ball_outputs)

    all_frame_ids = set()
    all_frame_ids.update(int(k) for k in player_outputs.keys())
    all_frame_ids.update(int(k) for k in ball_outputs.keys())

    if num_frames is None:
        if len(all_frame_ids) == 0:
            num_frames = 0
        else:
            num_frames = max(all_frame_ids) + 1

    result = {}

    # 1. 保存 player trajectory
    # 这里把 SAM3 的原始 player id 重新映射成 player_0, player_1, ...
    for new_pid, raw_pid in enumerate(player_ids):
        object_name = f"player_{new_pid}"
        trajectory = []

        for frame_idx in range(num_frames):
            frame_dict = get_frame_dict(player_outputs, frame_idx)
            mask = frame_dict.get(raw_pid, None)

            if mask is None:
                mask = frame_dict.get(str(raw_pid), None)

            bbox = mask_to_bbox(mask) if mask is not None else None
            trajectory.append(bbox)

        result[object_name] = {"trajectory": trajectory}

    # 2. 保存 ball trajectory
    # 这里把 ball 命名为 ball_1, ball_2, ...
    for new_bid, raw_bid in enumerate(ball_ids, start=1):
        object_name = f"ball_{new_bid}"
        trajectory = []

        for frame_idx in range(num_frames):
            frame_dict = get_frame_dict(ball_outputs, frame_idx)
            mask = frame_dict.get(raw_bid, None)

            if mask is None:
                mask = frame_dict.get(str(raw_bid), None)

            bbox = mask_to_bbox(mask) if mask is not None else None
            trajectory.append(bbox)

        result[object_name] = {"trajectory": trajectory}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def propagate_in_video(predictor, session_id):
    """Collect streamed SAM3 propagation responses for one session.

    Args:
        predictor: SAM3 video predictor instance.
        session_id: Active SAM3 session identifier.

    Returns:
        Mapping from frame indices to raw SAM3 outputs.
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def prune_prompt_objects(
    predictor,
    session_id,
    frame_index,
    prompt_outputs,
    max_objects,
    prompt_name,
):
    """Remove low-confidence objects created by the initial text prompt.

    SAM3 applies ``max_num_objects`` when associating new detections during
    propagation, but the first ``add_prompt`` call can seed more objects before
    that limit is consulted. Those initial objects form the tracker attention
    batch, so they must be pruned explicitly on memory-limited GPUs.

    Args:
        predictor: SAM3 video predictor instance.
        session_id: Active SAM3 session identifier.
        frame_index: Frame on which the prompt was introduced.
        prompt_outputs: Output dictionary returned by ``add_prompt``.
        max_objects: Maximum number of prompt objects to keep.
        prompt_name: Human-readable prompt label used in diagnostics.

    Returns:
        Kept object IDs ordered by descending prompt confidence.

    Raises:
        RuntimeError: If SAM3 returns object IDs without matching confidence
            scores, making deterministic pruning impossible.
    """
    object_ids = np.asarray(prompt_outputs.get("out_obj_ids", [])).reshape(-1)
    probabilities = np.asarray(prompt_outputs.get("out_probs", [])).reshape(-1)

    if object_ids.size == 0:
        print(f"SAM3 prompt candidates ({prompt_name}): detected=0, kept=0")
        return []
    if probabilities.size != object_ids.size:
        raise RuntimeError(
            f"SAM3 returned {object_ids.size} object IDs but "
            f"{probabilities.size} confidence scores for {prompt_name}"
        )

    ranked_indices = np.argsort(-probabilities, kind="stable")
    keep_indices = set(ranked_indices[:max_objects].tolist())
    kept_object_ids = [int(object_ids[index]) for index in ranked_indices[:max_objects]]
    removed_objects = []

    for index, (object_id, probability) in enumerate(zip(object_ids, probabilities)):
        if index in keep_indices:
            continue
        predictor.handle_request(
            request=dict(
                type="remove_object",
                session_id=session_id,
                frame_index=frame_index,
                obj_id=int(object_id),
            )
        )
        removed_objects.append(
            {"object_id": int(object_id), "score": float(probability)}
        )

    print(
        f"SAM3 prompt candidates ({prompt_name}): "
        f"detected={object_ids.size}, kept={len(kept_object_ids)}, "
        f"removed={removed_objects}"
    )
    return kept_object_ids


def run_text_prompt(
    predictor,
    session_id,
    prompt_text,
    max_objects,
    prompt_name,
    frame_index=0,
):
    """Reset a session and propagate one text prompt through the video.

    Args:
        predictor: SAM3 video predictor instance.
        session_id: Active SAM3 session identifier.
        prompt_text: Open-vocabulary object description.
        max_objects: Maximum number of initial prompt objects to retain.
        prompt_name: Human-readable prompt label used in diagnostics.
        frame_index: Frame on which the prompt is introduced.

    Returns:
        Per-frame masks prepared as NumPy arrays for trajectory conversion.
    """
    predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    prompt_response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_index,
            text=prompt_text,
        )
    )
    prune_prompt_objects(
        predictor=predictor,
        session_id=session_id,
        frame_index=frame_index,
        prompt_outputs=prompt_response["outputs"],
        max_objects=max_objects,
        prompt_name=prompt_name,
    )

    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    return outputs_per_frame


def configure_object_limit(predictor, max_num_objects, prompt_name):
    """Broadcast SAM3 masklet and compiled-slot limits to every GPU rank.

    SAM3 treats a non-positive model limit as effectively unlimited and expands
    it to 10,000 objects while reserving 16 object slots for optional compile
    warm-up. ``max_num_objects`` limits the global active masklet batch that
    drives eager-mode attention memory. The compatibility request derives
    ``num_obj_for_compile`` per GPU and is dispatched to all worker processes.

    Args:
        predictor: Initialized ``Sam3VideoPredictorMultiGPU`` instance.
        max_num_objects: Positive global maximum number of active masklets.
        prompt_name: Human-readable prompt label used in runtime diagnostics.

    Raises:
        ValueError: If the limit is not positive.
        RuntimeError: If a GPU rank has no configurable SAM3 object limit.
    """
    if max_num_objects <= 0:
        raise ValueError("--max-num-objects must be a positive integer")

    response = predictor.handle_request(
        request=dict(
            type="configure_object_limit",
            max_num_objects=max_num_objects,
        )
    )
    print(
        f"SAM3 object limits ({prompt_name}): "
        f"global={response['max_num_objects']}, "
        f"per_gpu_slots={response['num_obj_for_compile']}, "
        f"world_size={response['world_size']}"
    )


def configure_tracker_memory(predictor, num_maskmem, max_cond_frames):
    """Broadcast reduced temporal-memory settings to every SAM3 GPU rank.

    SAM3 normally retains seven mask-memory frames and lets four conditioning
    frames participate in memory attention. Those defaults produce increasing
    attention pressure around reconditioning frames on Turing GPUs. This
    compatibility profile retains the most recent temporal context while
    bounding the attention key/value sequence on every rank.

    Args:
        predictor: Initialized ``Sam3VideoPredictorMultiGPU`` instance.
        num_maskmem: Total mask-memory slots, including the input frame.
        max_cond_frames: Maximum conditioning frames used by memory attention.

    Raises:
        ValueError: If either limit is less than one.
        RuntimeError: If a GPU rank lacks configurable tracker attributes.
    """
    if num_maskmem < 1:
        raise ValueError("--sam3-num-maskmem must be at least one")
    if max_cond_frames < 1:
        raise ValueError("--sam3-max-cond-frames must be at least one")

    response = predictor.handle_request(
        request=dict(
            type="configure_tracker_memory",
            num_maskmem=num_maskmem,
            max_cond_frames_in_attn=max_cond_frames,
        )
    )
    print(
        "SAM3 tracker memory limits: "
        f"num_maskmem={response['num_maskmem']}, "
        f"max_cond_frames_in_attn={response['max_cond_frames_in_attn']}"
    )


def parse_args():
    """Parse command-line options for SAM3 tracking.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run SAM3 video segmentation and export bbox jsons"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720_raw.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--gpus_to_use",
        type=str,
        default=SETTINGS.gpu_ids,
        help="GPU ids, e.g. '0' or '0,1,2'",
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=str(SETTINGS.sam3_checkpoint),
        help="Path to the local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--sam3_bpe",
        type=str,
        default=str(SETTINGS.sam3_bpe),
        help="Path to the SAM3 BPE vocabulary.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store decoded and resized video frames in CPU memory instead of "
            "GPU memory. Recommended for long clips on memory-limited GPUs."
        ),
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Store reusable SAM3 inference state in CPU memory. This saves "
            "additional GPU memory at the cost of lower tracking throughput."
        ),
    )
    parser.add_argument(
        "--max-num-objects",
        type=int,
        default=10,
        help=(
            "Maximum simultaneous player masklets and compiled object slots. "
            "The default of 10 matches the number of on-court players and "
            "reduces attention memory on pre-Ampere GPUs. In multi-GPU mode, "
            "the global limit is synchronized and divided across all ranks."
        ),
    )
    parser.add_argument(
        "--max-ball-objects",
        type=int,
        default=2,
        help=(
            "Maximum simultaneous basketball masklets and compiled object "
            "slots after the player session is reset. Defaults to 2 to allow "
            "one false positive without using the player-sized batch."
        ),
    )
    parser.add_argument(
        "--sam3-num-maskmem",
        type=int,
        default=3,
        help=(
            "Number of SAM3 mask-memory frames retained by each tracker. "
            "Defaults to 3 instead of the upstream 7 for Turing GPUs."
        ),
    )
    parser.add_argument(
        "--sam3-max-cond-frames",
        type=int,
        default=1,
        help=(
            "Maximum conditioning frames participating in memory attention. "
            "Defaults to 1 instead of the upstream 4 for Turing GPUs."
        ),
    )
    return parser.parse_args()


def main():
    """Run SAM3 tracking and write raw player and ball trajectories."""
    args = parse_args()
    gpus_to_use = [int(x) for x in args.gpus_to_use.split(",")]
    video_path = args.video_path
    json_save_path = args.json_save_path

    # Validate before model construction so invalid arguments cannot leave
    # spawned SAM3 worker processes behind.
    if args.max_num_objects <= 0:
        raise ValueError("--max-num-objects must be a positive integer")
    if args.max_ball_objects <= 0:
        raise ValueError("--max-ball-objects must be a positive integer")
    if args.sam3_num_maskmem < 1:
        raise ValueError("--sam3-num-maskmem must be at least one")
    if args.sam3_max_cond_frames < 1:
        raise ValueError("--sam3-max-cond-frames must be at least one")

    video_path = str(SETTINGS.require_file(video_path, "Input video"))
    sam3_checkpoint = str(
        SETTINGS.require_file(args.sam3_checkpoint, "SAM3 checkpoint")
    )
    sam3_bpe = str(SETTINGS.require_file(args.sam3_bpe, "SAM3 BPE vocabulary"))
    SETTINGS.create_parent(json_save_path)

    predictor = build_sam3_video_predictor(
        checkpoint_path=sam3_checkpoint,
        bpe_path=sam3_bpe,
        gpus_to_use=gpus_to_use,
    )
    configure_object_limit(
        predictor,
        max_num_objects=args.max_num_objects,
        prompt_name="players",
    )
    configure_tracker_memory(
        predictor,
        num_maskmem=args.sam3_num_maskmem,
        max_cond_frames=args.sam3_max_cond_frames,
    )

    print(
        "SAM3 CPU offload: "
        f"video={args.offload_video_to_cpu}, "
        f"state={args.offload_state_to_cpu}"
    )

    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        )
        session_id = response["session_id"]

        player_outputs = run_text_prompt(
            predictor,
            session_id,
            prompt_text="basketball player on the court",
            max_objects=args.max_num_objects,
            prompt_name="players",
            frame_index=0,
        )

        # Player outputs are already materialized as NumPy arrays. The next
        # ``run_text_prompt`` reset can therefore clear tracker state and use a
        # much smaller compiled batch for basketball candidates.
        configure_object_limit(
            predictor,
            max_num_objects=args.max_ball_objects,
            prompt_name="basketball",
        )
        ball_outputs = run_text_prompt(
            predictor,
            session_id,
            prompt_text="basketball",
            max_objects=args.max_ball_objects,
            prompt_name="basketball",
            frame_index=0,
        )

        build_trajectory_json(
            player_outputs=player_outputs,
            ball_outputs=ball_outputs,
            json_path=json_save_path,
        )

    except RuntimeError as error:
        err_msg = str(error).lower()
        if isinstance(error, torch.OutOfMemoryError) or "out of memory" in err_msg:
            print(f"[OOM] skip video due to CUDA OOM: {video_path}")
        else:
            print(f"[ERROR] failed to process video {video_path}: {error}")
        # A missing output must never be reported to batch runners as success.
        raise

    finally:
        # 一定要 close session
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass

        # 强制释放显存 & Python 对象
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
