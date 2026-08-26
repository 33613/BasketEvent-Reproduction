"""Command-line entry point for distributed BasketEvent training.

This module translates command-line options into concrete dependencies.  The
training lifecycle itself is implemented by
:class:`src.modules.event_recognition.solver.Solver` and is
started through its public ``run()`` method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from src.core.config import SETTINGS
from src.modules.event_recognition.dataset import VideoBagClipsDataset, bag_collate_fn
from src.modules.event_recognition.labels import LABEL_MAP
from src.modules.event_recognition.playnet.model import PlayerEventModel
from src.modules.event_recognition.solver import (
    DistributedContext,
    Solver,
    SolverConfig,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    """Parse the existing training command-line interface.

    Returns:
        Training options supplied by the user or their compatible defaults.
    """
    parser = argparse.ArgumentParser(
        description="Train the BasketEvent player-event recognition model."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--bag_clips", type=int, default=4)
    data.add_argument("--clip_len", type=int, default=8)
    data.add_argument("--fps_in", type=int, default=25)
    data.add_argument("--fps_out", type=int, default=4)
    data.add_argument("--img_size", type=int, default=224)
    data.add_argument("--rebuild_cache", action="store_true")
    data.add_argument(
        "--bbox_dir",
        type=str,
        default=str(SETTINGS.train_annotations_dir),
    )
    data.add_argument(
        "--video_dir",
        type=str,
        default=str(SETTINGS.videos_dir),
    )
    data.add_argument(
        "--cache_dir",
        type=str,
        default=str(SETTINGS.cache_dir),
    )
    data.add_argument(
        "--timesformer_model",
        type=str,
        default=str(SETTINGS.timesformer_model),
        help="Path to the local TimeSformer model directory.",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--epochs", type=int, default=10)
    optimization.add_argument("--batch_size", type=int, default=1)
    optimization.add_argument("--num_workers", type=int, default=4)
    optimization.add_argument("--lr", type=float, default=5e-5)
    optimization.add_argument("--weight_decay", type=float, default=0.05)
    optimization.add_argument("--topk", type=int, default=3)
    optimization.add_argument("--grad_clip", type=float, default=1.0)
    optimization.add_argument("--seed", type=int, default=123)
    optimization.add_argument("--clip_aux_weight", type=float, default=0.2)
    optimization.add_argument("--clip_soft_tau", type=float, default=0.5)

    output = parser.add_argument_group("output and resume")
    output.add_argument(
        "--save_dir",
        type=str,
        default=str(SETTINGS.trained_checkpoints_dir),
    )
    output.add_argument("--resume", action="store_true")
    output.add_argument("--resume_ckpt", type=str, default="")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Validate external inputs and create writable runtime directories.

    Args:
        args: Parsed training arguments.

    Returns:
        The same namespace with normalized path strings.
    """
    args.bbox_dir = str(
        SETTINGS.require_directory(args.bbox_dir, "Training annotation directory")
    )
    args.video_dir = str(
        SETTINGS.require_directory(args.video_dir, "Runtime video directory")
    )
    args.timesformer_model = str(
        SETTINGS.require_directory(
            args.timesformer_model, "TimeSformer model directory"
        )
    )
    SETTINGS.create_directories([args.cache_dir, args.save_dir])
    return args


def build_training_loader(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler]:
    """Build the dataset, distributed sampler, and collating data loader.

    Args:
        args: Validated training options.
        context: Distributed worker metadata.

    Returns:
        Data loader and its epoch-aware distributed sampler.
    """
    cache_path = Path(args.cache_dir) / (
        f"bag_clip{args.clip_len}_fps{args.fps_out}_" f"M{args.bag_clips}_train.pkl"
    )
    dataset = VideoBagClipsDataset(
        bbox_dir=args.bbox_dir,
        video_dir=args.video_dir,
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        bag_clips=args.bag_clips,
        size=args.img_size,
        cache_path=str(cache_path),
        rebuild_cache=args.rebuild_cache,
        add_blank=True,
        require_ball=True,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        drop_last=False,
    )

    loader_options = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": bag_collate_fn,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_options)
    return loader, sampler


def build_distributed_model(
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[DistributedDataParallel, AdamW, int]:
    """Assemble the event model, optimizer, and DDP wrapper.

    Args:
        args: Validated training options.
        context: Distributed worker metadata.

    Returns:
        DDP-wrapped model, optimizer, and number of event classes.
    """
    num_classes = len(LABEL_MAP)
    model = PlayerEventModel(
        num_classes=num_classes,
        pretrained_name=args.timesformer_model,
        local_files_only=SETTINGS.hf_local_files_only,
        roi_out_size=(1, 1),
        use_clip_relation=True,
        use_actor_global=True,
        use_person_relation=True,
    ).to(context.device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        find_unused_parameters=True,
    )
    return distributed_model, optimizer, num_classes


def build_solver_config(args: argparse.Namespace) -> SolverConfig:
    """Convert CLI values into an argparse-independent solver config."""
    return SolverConfig(
        epochs=args.epochs,
        fps_in=float(args.fps_in),
        topk=args.topk,
        grad_clip=args.grad_clip,
        clip_aux_weight=args.clip_aux_weight,
        clip_soft_tau=args.clip_soft_tau,
        save_dir=args.save_dir,
        resume=args.resume,
        resume_ckpt=args.resume_ckpt,
    )


def main() -> None:
    """Construct distributed training dependencies and run the solver."""
    args = resolve_paths(parse_args())
    context = DistributedContext.initialize()
    try:
        seed_everything(args.seed, context.rank)
        if context.is_main_process:
            print(
                "[DDP] "
                f"world_size={context.world_size} rank={context.rank} "
                f"local_rank={context.local_rank} device={context.device}"
            )

        loader, sampler = build_training_loader(args, context)
        model, optimizer, num_classes = build_distributed_model(args, context)
        solver = Solver(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            train_sampler=sampler,
            context=context,
            config=build_solver_config(args),
            num_classes=num_classes,
        )
        solver.run()
    finally:
        context.close()


if __name__ == "__main__":
    main()
