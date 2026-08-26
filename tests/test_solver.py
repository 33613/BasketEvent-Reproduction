"""Unit and smoke tests for the object-oriented training solver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD

from src.modules.event_recognition.solver import (
    CheckpointManager,
    DistributedContext,
    PlayerEventObjective,
    Solver,
    SolverConfig,
    move_collated_batch_to_device,
    slice_video_from_collated_batch,
)


def make_config(save_dir: str, **overrides) -> SolverConfig:
    """Build a compact valid solver configuration for tests."""
    values = {
        "epochs": 1,
        "fps_in": 4.0,
        "topk": 1,
        "grad_clip": 1.0,
        "clip_aux_weight": 0.2,
        "clip_soft_tau": 0.5,
        "save_dir": save_dir,
    }
    values.update(overrides)
    return SolverConfig(**values)


class FakeSampler:
    """Record epochs supplied by the solver."""

    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        """Remember one epoch selection."""
        self.epochs.append(epoch)


class DummyEventModel(nn.Module):
    """Small model implementing the PlayerEventModel output contract."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, **inputs) -> dict[str, torch.Tensor]:
        """Create person and clip logits from the video mean."""
        clips = inputs["clips_video"]
        num_clips = clips.shape[0]
        num_people = inputs["bboxes"][0].shape[0]
        scalar = clips.mean().reshape(1, 1).expand(num_people, 1)
        person_logits = self.classifier(scalar)
        clip_logits = person_logits.unsqueeze(0).expand(num_clips, -1, -1)
        return {
            "logits_person": person_logits,
            "logits_clip": clip_logits,
        }


def synthetic_batch(batch_size: int = 1, bag_size: int = 2) -> dict:
    """Return a minimal batch with one person in every video."""
    return {
        "clips_video": torch.ones(batch_size, bag_size, 1, 1, 1, 1),
        "idx": torch.zeros(batch_size, bag_size, 1, dtype=torch.long),
        "nums": torch.full((batch_size,), 2, dtype=torch.long),
        "bboxes": [torch.ones(1, 1, 4) for _ in range(batch_size * bag_size)],
        "bbox_masks": [torch.ones(1, 1) for _ in range(batch_size * bag_size)],
        "clips_ball": torch.ones(batch_size, bag_size, 1, 4),
        "clips_ball_mask": torch.ones(batch_size, bag_size, 1),
        "labels": [torch.tensor([1], dtype=torch.long) for _ in range(batch_size)],
        "person_ids": [[f"person_{index}"] for index in range(batch_size)],
    }


class PlayerEventObjectiveTest(unittest.TestCase):
    """Exercise filtering and both levels of the training objective."""

    def test_mixed_positive_and_blank_loss_is_finite(self) -> None:
        """Positive and background actors should contribute a finite loss."""
        with tempfile.TemporaryDirectory() as directory:
            objective = PlayerEventObjective(make_config(directory))
            outputs = {
                "logits_person": torch.tensor(
                    [[0.1, 1.2, -0.4], [1.0, 0.2, -0.1]],
                    requires_grad=True,
                ),
                "logits_clip": torch.tensor(
                    [
                        [[0.0, 1.0, -0.2], [1.2, 0.0, -0.3]],
                        [[0.2, 0.8, -0.1], [0.9, 0.1, -0.2]],
                    ],
                    requires_grad=True,
                ),
            }
            labels = torch.tensor([1, 0])
            class_weight = torch.tensor([0.1, 1.0, 1.0])

            result = objective(outputs, labels, class_weight)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(torch.isfinite(result.total))
            self.assertAlmostEqual(
                result.total.item(),
                (result.person + 0.2 * result.clip).item(),
                places=6,
            )

    def test_confident_non_blank_prediction_can_filter_blank_gt(self) -> None:
        """An entirely ignored background video should return no objective."""
        with tempfile.TemporaryDirectory() as directory:
            objective = PlayerEventObjective(make_config(directory))
            outputs = {
                "logits_person": torch.tensor([[0.0, 10.0, 0.0]]),
                "logits_clip": torch.tensor([[[0.0, 10.0, 0.0]]]),
            }
            result = objective(
                outputs,
                torch.tensor([0]),
                torch.tensor([0.1, 1.0, 1.0]),
            )
            self.assertIsNone(result)


class BatchAndCheckpointTest(unittest.TestCase):
    """Verify variable-person slicing and stable checkpoint persistence."""

    def test_batch_slice_uses_flat_clip_offset(self) -> None:
        """Video index two must select its own contiguous clip metadata."""
        batch = synthetic_batch(batch_size=2, bag_size=3)
        for index, boxes in enumerate(batch["bboxes"]):
            boxes.fill_(float(index))
        prepared = move_collated_batch_to_device(batch, torch.device("cpu"))

        video = slice_video_from_collated_batch(prepared, 1)

        self.assertEqual(tuple(video.clips_video.shape), (3, 1, 1, 1, 1))
        self.assertEqual([item.item() for item in video.nums], [2, 2, 2])
        self.assertEqual([item[0, 0, 0].item() for item in video.bboxes], [3, 4, 5])

    def test_checkpoint_round_trip_uses_legacy_schema(self) -> None:
        """Saved checkpoints must retain epoch/global_step/model/optimizer keys."""
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory)
            model = nn.Linear(2, 3)
            optimizer = AdamW(model.parameters(), lr=0.01)
            before = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

            epoch_path, latest_path = manager.save(
                epoch=4,
                global_step=19,
                model=model,
                optimizer=optimizer,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(5.0)
            state = manager.load(latest_path, model, optimizer, torch.device("cpu"))

            document = torch.load(epoch_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(document), {"epoch", "global_step", "model", "optimizer"}
            )
            self.assertEqual(state.start_epoch, 5)
            self.assertEqual(state.global_step, 19)
            self.assertTrue(epoch_path.is_file())
            self.assertTrue(latest_path.is_file())
            for key, expected in before.items():
                self.assertTrue(torch.equal(model.state_dict()[key], expected))


class SolverRunTest(unittest.TestCase):
    """Smoke-test the public Solver.run() lifecycle on CPU."""

    def test_run_updates_model_and_writes_checkpoint(self) -> None:
        """One batch should produce one optimizer step and two checkpoint files."""
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(3)
            model = DummyEventModel()
            optimizer = SGD(model.parameters(), lr=0.1)
            initial = model.classifier.weight.detach().clone()
            sampler = FakeSampler()
            context = DistributedContext(
                rank=0,
                world_size=1,
                local_rank=0,
                device=torch.device("cpu"),
            )
            solver = Solver(
                model=model,
                optimizer=optimizer,
                train_loader=[synthetic_batch()],
                train_sampler=sampler,
                context=context,
                config=make_config(directory),
                num_classes=3,
            )

            state = solver.run()

            self.assertEqual(state.start_epoch, 2)
            self.assertEqual(state.global_step, 1)
            self.assertEqual(sampler.epochs, [1])
            self.assertFalse(torch.equal(model.classifier.weight, initial))
            self.assertTrue((Path(directory) / "epoch_001.pt").is_file())
            self.assertTrue((Path(directory) / "latest.pt").is_file())


if __name__ == "__main__":
    unittest.main()
