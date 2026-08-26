"""Training orchestration for the BasketEvent event-recognition model.

The solver owns optimization, loss computation, checkpoint lifecycle, and the
epoch loop.  Dataset construction and model assembly stay outside this module,
which keeps the solver reusable with lightweight test doubles and future model
variants that preserve the same input/output contract.
"""

from __future__ import annotations

import datetime
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


class EpochSampler(Protocol):
    """Describe the sampler capability required by :class:`Solver`."""

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle order for one epoch."""


@dataclass(frozen=True)
class DistributedContext:
    """Store the process-local state of one distributed training worker.

    Attributes:
        rank: Global worker rank.
        world_size: Total number of workers.
        local_rank: GPU index assigned to this process.
        device: CUDA device used by this process.
    """

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @classmethod
    def initialize(cls, timeout_minutes: int = 2) -> "DistributedContext":
        """Initialize NCCL from the environment created by ``torchrun``.

        Args:
            timeout_minutes: Maximum time allowed for a collective operation.

        Returns:
            The distributed context for the current process.
        """
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=torch.device("cuda", local_rank),
        )

    @property
    def is_main_process(self) -> bool:
        """Return whether this worker may write shared output files."""
        return self.rank == 0

    def reduce_mean(self, value: float) -> float:
        """Average one scalar across workers.

        Args:
            value: Process-local scalar value.

        Returns:
            Mean value across the distributed group.
        """
        if self.world_size == 1 or not dist.is_initialized():
            return float(value)
        tensor = torch.tensor([value], device=self.device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return float(tensor.item())

    def barrier(self) -> None:
        """Synchronize all workers when distributed training is active."""
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

    def all_true(self, value: bool) -> bool:
        """Return whether every worker reports a true condition.

        Args:
            value: Process-local condition.

        Returns:
            ``True`` only when all workers report ``True``.  A single-process
            context returns ``value`` unchanged.
        """
        if self.world_size == 1 or not dist.is_initialized():
            return bool(value)
        flag = torch.tensor([int(value)], device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def close(self) -> None:
        """Destroy the distributed process group if it is active."""
        if dist.is_initialized():
            dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch for one distributed worker.

    Args:
        seed: Base experiment seed.
        rank: Worker rank added to the base seed.
    """
    worker_seed = int(seed) + int(rank)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


@dataclass(frozen=True)
class SolverConfig:
    """Configure optimization without depending on command-line parsing.

    Attributes:
        epochs: Last epoch number to execute, starting at one.
        fps_in: Source-video frame rate passed to the model.
        topk: Number of clips used by top-k MIL pooling.
        grad_clip: Maximum gradient norm; non-positive disables clipping.
        clip_aux_weight: Weight of the clip-level auxiliary objective.
        clip_soft_tau: Temperature for positive clip weighting.
        save_dir: Directory for epoch and latest checkpoints.
        resume: Whether to attempt checkpoint restoration.
        resume_ckpt: Optional explicit checkpoint path.
        label_smoothing: Smoothing applied to classification targets.
        bootstrap_alpha: Weight of labels in bootstrapped targets.
        ignore_threshold: Confidence threshold for ignoring suspicious blank GT.
        blank_class: Integer index of the background event class.
        blank_class_weight: Relative loss weight assigned to background.
    """

    epochs: int
    fps_in: float
    topk: int
    grad_clip: float
    clip_aux_weight: float
    clip_soft_tau: float
    save_dir: str
    resume: bool = False
    resume_ckpt: str = ""
    label_smoothing: float = 0.1
    bootstrap_alpha: float = 0.8
    ignore_threshold: float = 0.8
    blank_class: int = 0
    blank_class_weight: float = 0.1

    def __post_init__(self) -> None:
        """Validate values that would otherwise produce silent bad training."""
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.fps_in <= 0:
            raise ValueError("fps_in must be positive")
        if self.topk < 1:
            raise ValueError("topk must be at least 1")
        if self.clip_soft_tau <= 0:
            raise ValueError("clip_soft_tau must be positive")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.bootstrap_alpha <= 1.0:
            raise ValueError("bootstrap_alpha must be in [0, 1]")
        if not 0.0 <= self.ignore_threshold <= 1.0:
            raise ValueError("ignore_threshold must be in [0, 1]")
        if self.blank_class < 0:
            raise ValueError("blank_class must be non-negative")
        if self.blank_class_weight < 0:
            raise ValueError("blank_class_weight must be non-negative")


@dataclass(frozen=True)
class SolverState:
    """Record the next epoch and completed optimizer steps."""

    start_epoch: int = 1
    global_step: int = 0


@dataclass(frozen=True)
class LossBreakdown:
    """Expose total and component losses for logging and tests."""

    total: torch.Tensor
    person: torch.Tensor
    clip: torch.Tensor


@dataclass(frozen=True)
class VideoBatch:
    """Hold one video's tensors sliced from a collated training batch."""

    clips_video: torch.Tensor
    idx: torch.Tensor
    nums: torch.Tensor
    bboxes: list[torch.Tensor]
    bbox_masks: list[torch.Tensor]
    clips_ball: torch.Tensor
    clips_ball_mask: torch.Tensor
    labels: torch.Tensor
    person_ids: Sequence[str]


def move_collated_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move every tensor in a collated batch exactly once.

    Args:
        batch: Output of ``bag_collate_fn``.
        device: Target training device.

    Returns:
        A shallow batch copy whose tensor values live on ``device``.
    """
    moved = dict(batch)
    for key in ("clips_video", "idx", "nums", "clips_ball", "clips_ball_mask"):
        moved[key] = batch[key].to(device, non_blocking=True)
    moved["bboxes"] = [value.to(device, non_blocking=True) for value in batch["bboxes"]]
    moved["bbox_masks"] = [
        value.to(device, non_blocking=True) for value in batch["bbox_masks"]
    ]
    moved["labels"] = [value.to(device, non_blocking=True) for value in batch["labels"]]
    return moved


def slice_video_from_collated_batch(batch: Mapping[str, Any], index: int) -> VideoBatch:
    """Extract one variable-person video from a prepared batch.

    Args:
        batch: Device-ready output of :func:`move_collated_batch_to_device`.
        index: Video index inside the collated batch.

    Returns:
        Tensors for one model call.
    """
    batch_size, bag_size = batch["clips_video"].shape[:2]
    if not 0 <= index < batch_size:
        raise IndexError(f"video index {index} is outside batch size {batch_size}")

    base = index * bag_size
    return VideoBatch(
        clips_video=batch["clips_video"][index],
        idx=batch["idx"][index],
        nums=batch["nums"][index].repeat(bag_size),
        bboxes=[batch["bboxes"][base + clip] for clip in range(bag_size)],
        bbox_masks=[batch["bbox_masks"][base + clip] for clip in range(bag_size)],
        clips_ball=batch["clips_ball"][index],
        clips_ball_mask=batch["clips_ball_mask"][index],
        labels=batch["labels"][index],
        person_ids=batch["person_ids"][index],
    )


class PlayerEventObjective:
    """Compute the person-level and clip-level bootstrapped objectives."""

    def __init__(self, config: SolverConfig) -> None:
        """Store the loss hyperparameters.

        Args:
            config: Solver configuration containing all objective constants.
        """
        self.config = config

    def __call__(
        self,
        outputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        class_weight: torch.Tensor,
    ) -> LossBreakdown | None:
        """Compute one video's training objective.

        Args:
            outputs: Model output containing ``logits_person`` and
                ``logits_clip``.
            labels: Event class for every person trajectory.
            class_weight: Per-class loss weights.

        Returns:
            Component losses, or ``None`` when every sample is filtered out.
        """
        logits_person = outputs["logits_person"]
        logits_clip = outputs["logits_clip"]
        if logits_person.numel() == 0:
            return None
        if logits_person.shape[0] != labels.numel():
            raise ValueError(
                "person logits and labels disagree: "
                f"{logits_person.shape[0]} != {labels.numel()}"
            )

        person_loss = self._person_loss(logits_person, labels, class_weight)
        if person_loss is None:
            return None
        clip_loss = self._clip_loss(logits_clip, labels, class_weight)
        total = person_loss + self.config.clip_aux_weight * clip_loss
        return LossBreakdown(total=total, person=person_loss, clip=clip_loss)

    def _smooth_targets(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Return label-smoothed one-hot targets."""
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        smoothing = self.config.label_smoothing
        return one_hot * (1.0 - smoothing) + smoothing / num_classes

    def _person_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weight: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute the filtered person-level bootstrap loss."""
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        num_classes = logits.shape[-1]

        with torch.no_grad():
            smooth_target = self._smooth_targets(labels, num_classes)
            bootstrap_target = (
                self.config.bootstrap_alpha * smooth_target
                + (1.0 - self.config.bootstrap_alpha) * probabilities
            )
            max_probability, prediction = probabilities.max(dim=-1)
            include = torch.ones_like(labels, dtype=torch.bool)
            blank = labels == self.config.blank_class
            confident_non_blank = (max_probability > self.config.ignore_threshold) & (
                prediction != self.config.blank_class
            )
            include[blank & confident_non_blank] = False

        loss = -(bootstrap_target * log_probabilities).sum(dim=-1)
        loss = loss * class_weight[labels]
        loss = loss[include]
        return loss.mean() if loss.numel() else None

    def _clip_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the positive/negative clip-level auxiliary loss."""
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        num_clips, _, num_classes = logits.shape
        blank = labels == self.config.blank_class
        terms: list[torch.Tensor] = []

        positive = ~blank
        if positive.any():
            indices = torch.nonzero(positive, as_tuple=False).squeeze(1)
            positive_labels = labels[indices]
            positive_probabilities = probabilities[:, indices, :]
            positive_log_probabilities = log_probabilities[:, indices, :]

            with torch.no_grad():
                gt_scores = positive_probabilities.gather(
                    dim=-1,
                    index=positive_labels.view(1, -1, 1).expand(num_clips, -1, 1),
                ).squeeze(-1)
                clip_weights = torch.softmax(
                    gt_scores / self.config.clip_soft_tau, dim=0
                )
                smooth_target = (
                    self._smooth_targets(positive_labels, num_classes)
                    .unsqueeze(0)
                    .expand(num_clips, -1, -1)
                )
                bootstrap_target = (
                    self.config.bootstrap_alpha * smooth_target
                    + (1.0 - self.config.bootstrap_alpha) * positive_probabilities
                )

            loss = -(bootstrap_target * positive_log_probabilities).sum(dim=-1)
            loss = loss * class_weight[positive_labels].unsqueeze(0)
            loss = loss * clip_weights
            terms.append(loss.sum(dim=0).mean())

        if blank.any():
            indices = torch.nonzero(blank, as_tuple=False).squeeze(1)
            negative_probabilities = probabilities[:, indices, :]
            negative_log_probabilities = log_probabilities[:, indices, :]
            with torch.no_grad():
                blank_targets = torch.full(
                    (num_clips, indices.numel()),
                    self.config.blank_class,
                    device=logits.device,
                    dtype=torch.long,
                )
                smooth_target = self._smooth_targets(blank_targets, num_classes)
                bootstrap_target = (
                    self.config.bootstrap_alpha * smooth_target
                    + (1.0 - self.config.bootstrap_alpha) * negative_probabilities
                )
            loss = -(bootstrap_target * negative_log_probabilities).sum(dim=-1)
            loss = loss * class_weight[self.config.blank_class]
            terms.append(loss.mean())

        if not terms:
            return torch.zeros((), device=logits.device)
        return torch.stack(terms).mean()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from DDP or a plain module.

    Args:
        model: Possibly distributed model wrapper.

    Returns:
        The module whose parameter keys must be written to checkpoints.
    """
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


class CheckpointManager:
    """Persist and restore the stable BasketEvent checkpoint schema."""

    def __init__(self, save_dir: str | Path) -> None:
        """Create a checkpoint manager.

        Args:
            save_dir: Directory receiving epoch and ``latest.pt`` files.
        """
        self.save_dir = Path(save_dir).expanduser()

    @property
    def latest_path(self) -> Path:
        """Return the default resume checkpoint path."""
        return self.save_dir / "latest.pt"

    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: Optimizer,
    ) -> tuple[Path, Path]:
        """Write an epoch checkpoint and update ``latest.pt``.

        Args:
            epoch: Completed epoch number.
            global_step: Completed optimizer-step count.
            model: Plain or DDP-wrapped model.
            optimizer: Optimizer whose state should be resumable.

        Returns:
            Paths of the epoch checkpoint and latest checkpoint.
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        epoch_path = self.save_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, epoch_path)
        torch.save(state, self.latest_path)
        return epoch_path, self.latest_path

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None,
        device: torch.device | str,
    ) -> SolverState:
        """Restore model, optimizer, and progress using strict model keys.

        Args:
            checkpoint_path: Checkpoint written by :meth:`save`.
            model: Plain or DDP-wrapped destination model.
            optimizer: Optional optimizer to restore.
            device: Device used while deserializing tensors.

        Returns:
            State pointing to the next epoch.
        """
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=True
        )
        unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return SolverState(
            start_epoch=int(checkpoint.get("epoch", 0)) + 1,
            global_step=int(checkpoint.get("global_step", 0)),
        )


class Solver:
    """Train one event model through a public :meth:`run` lifecycle."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: Iterable[Mapping[str, Any]],
        train_sampler: EpochSampler,
        context: DistributedContext,
        config: SolverConfig,
        num_classes: int,
        objective: PlayerEventObjective | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        """Initialize the training dependencies.

        Args:
            model: Plain or DDP-wrapped event model.
            optimizer: Optimizer for ``model``.
            train_loader: Iterable of collated video bags.
            train_sampler: Sampler supporting deterministic ``set_epoch``.
            context: Distributed process metadata.
            config: Optimization and checkpoint configuration.
            num_classes: Event-class vocabulary size.
            objective: Optional custom objective for tests or experiments.
            checkpoint_manager: Optional custom checkpoint backend.
        """
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if config.blank_class >= num_classes:
            raise ValueError("blank_class must be smaller than num_classes")
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.context = context
        self.config = config
        self.num_classes = int(num_classes)
        self.objective = objective or PlayerEventObjective(config)
        self.checkpoints = checkpoint_manager or CheckpointManager(config.save_dir)
        self.class_weight = torch.ones(num_classes, device=context.device)
        self.class_weight[config.blank_class] = config.blank_class_weight

    def run(self) -> SolverState:
        """Restore optional state, train all requested epochs, and checkpoint.

        Returns:
            Final solver state with the next epoch and global step.
        """
        state = self._resume_if_requested()
        for epoch in range(state.start_epoch, self.config.epochs + 1):
            epoch_loss, global_step = self._train_epoch(epoch, state.global_step)
            state = SolverState(start_epoch=epoch + 1, global_step=global_step)
            mean_loss = self.context.reduce_mean(epoch_loss)
            if self.context.is_main_process:
                epoch_path, latest_path = self.checkpoints.save(
                    epoch, state.global_step, self.model, self.optimizer
                )
                print(f"[Epoch {epoch}] loss={mean_loss:.6f}")
                print(f"[CKPT] saved: {epoch_path}")
                print(f"[CKPT] saved: {latest_path}")
            self.context.barrier()
        if self.context.is_main_process:
            print("[Done]")
        return state

    def _resume_if_requested(self) -> SolverState:
        """Load the same checkpoint independently on every worker."""
        state = SolverState()
        if not self.config.resume:
            return state

        checkpoint_path = Path(
            self.config.resume_ckpt
            if self.config.resume_ckpt
            else self.checkpoints.latest_path
        ).expanduser()
        # All ranks must choose the same branch.  Otherwise a worker that sees
        # the shared checkpoint can wait at the load barrier while another
        # worker has already entered the training loop.
        checkpoint_available = self.context.all_true(checkpoint_path.is_file())
        if not checkpoint_available:
            if self.context.is_main_process:
                print(
                    f"[Resume] ckpt is not accessible to every worker: "
                    f"{checkpoint_path}; start from scratch"
                )
            return state

        # Every rank restores model and optimizer state. Broadcasting only the
        # counters would leave non-zero ranks with stale parameters.
        state = self.checkpoints.load(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.context.device,
        )
        self.context.barrier()
        if self.context.is_main_process:
            print(
                f"[Resume] loaded: {checkpoint_path} "
                f"(start_epoch={state.start_epoch}, "
                f"global_step={state.global_step})"
            )
        return state

    def _train_epoch(self, epoch: int, global_step: int) -> tuple[float, int]:
        """Train one epoch and return its local mean loss and updated step."""
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        loss_sum = 0.0
        steps = 0
        started = time.time()

        iterator: Iterable[Mapping[str, Any]] = self.train_loader
        if self.context.is_main_process:
            iterator = tqdm(
                self.train_loader,
                desc=f"Epoch [{epoch}/{self.config.epochs}]",
                ncols=120,
            )

        for batch in iterator:
            batch_loss = self._train_batch(batch)
            global_step += 1
            loss_sum += float(batch_loss.item())
            steps += 1
            if self.context.is_main_process and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    loss=f"{loss_sum / max(1, steps):.4f}",
                    B=batch["clips_video"].shape[0],
                    M=batch["clips_video"].shape[1],
                    topk=self.config.topk,
                )

        mean_loss = loss_sum / max(1, steps)
        if self.context.is_main_process:
            print(
                f"[Epoch {epoch}] local_loss={mean_loss:.6f} "
                f"time={time.time() - started:.1f}s"
            )
        return mean_loss, global_step

    def _train_batch(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Optimize one collated batch while forwarding videos individually."""
        prepared = move_collated_batch_to_device(batch, self.context.device)
        batch_size = prepared["clips_video"].shape[0]
        self.optimizer.zero_grad(set_to_none=True)
        total_loss: torch.Tensor | None = None
        valid_videos = 0

        for index in range(batch_size):
            video = slice_video_from_collated_batch(prepared, index)
            if video.labels.numel() == 0:
                continue
            outputs = self.model(
                clips_video=video.clips_video,
                idx=video.idx,
                nums=video.nums,
                bboxes=video.bboxes,
                bbox_masks=video.bbox_masks,
                clips_ball=video.clips_ball,
                clips_ball_mask=video.clips_ball_mask,
                fps_in=float(self.config.fps_in),
                topk=int(self.config.topk),
            )
            breakdown = self.objective(outputs, video.labels, self.class_weight)
            if breakdown is None:
                continue
            total_loss = (
                breakdown.total if total_loss is None else total_loss + breakdown.total
            )
            valid_videos += 1

        if total_loss is None:
            # Preserve the original training behavior for an entirely filtered
            # batch. A two-GPU smoke test should still monitor this rare branch.
            total_loss = torch.zeros((), device=self.context.device, requires_grad=True)
        else:
            total_loss = total_loss / valid_videos

        total_loss.backward()
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        self.optimizer.step()
        return total_loss.detach()
