"""Tests for reusable layers and the assembled player-event model."""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.model as legacy_model_module
import src.modules.event_recognition.playnet.model as model_module
from src.modules.event_recognition.playnet.layers import BBoxEmbedding, GatedClipPooling
from src.modules.event_recognition.playnet.model import PlayerEventModel


class FakeBackbone(nn.Module):
    """Small deterministic backbone used instead of a downloaded TimeSformer."""

    def __init__(self, *args, **kwargs) -> None:
        """Create a three-channel projection with a 2-by-2 output grid."""
        super().__init__()
        del args, kwargs
        self.hidden_dim = 8
        self.grid_h = 2
        self.grid_w = 2
        self.projection = nn.Conv3d(3, self.hidden_dim, kernel_size=1)

    def forward(self, video: torch.Tensor, **kwargs) -> dict[str, object]:
        """Project a channel-first clip and pool only its spatial dimensions."""
        del kwargs
        if video.shape[1] != 3:
            video = video.permute(0, 2, 1, 3, 4).contiguous()
        features = self.projection(video)
        features = F.adaptive_avg_pool3d(
            features, (video.shape[2], self.grid_h, self.grid_w)
        )
        return {
            "featmap": features,
            "aux": {
                "hidden_dim": self.hidden_dim,
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
            },
        }


class LayerContractTest(unittest.TestCase):
    """Verify masking and numerical behavior of standalone layers."""

    def test_bbox_embedding_masks_invalid_frames(self) -> None:
        """Invalid boxes must produce an exactly zero embedding."""
        layer = BBoxEmbedding(out_dim=8, image_size=16, dropout=0.0)
        boxes = torch.tensor([[[1.0, 1.0, 8.0, 12.0], [2.0, 2.0, 9.0, 13.0]]])
        mask = torch.tensor([[1.0, 0.0]])

        output = layer(boxes, mask)

        self.assertEqual(tuple(output.shape), (1, 2, 8))
        self.assertTrue(torch.equal(output[:, 1], torch.zeros_like(output[:, 1])))

    def test_gated_pooling_handles_all_invalid_person(self) -> None:
        """The all-invalid guard must keep pooling finite instead of producing NaN."""
        layer = GatedClipPooling(dim=8)
        features = torch.randn(3, 2, 8)
        valid = torch.tensor([[1, 0], [1, 0], [0, 0]], dtype=torch.bool)

        pooled, logits, weights = layer(features, valid_mask=valid, return_weights=True)

        self.assertEqual(tuple(pooled.shape), (2, 8))
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertEqual(tuple(weights.shape), (3, 2, 1))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(weights).all())


class PlayerEventModelAssemblyTest(unittest.TestCase):
    """Verify that model assembly preserves tensor and checkpoint contracts."""

    def _build_model(self) -> PlayerEventModel:
        """Construct the complete topology with a lightweight backbone."""
        with mock.patch.object(model_module, "TimeSformerBackbone", FakeBackbone):
            return PlayerEventModel(
                num_classes=4,
                pretrained_name="unused",
                local_files_only=True,
                image_size=16,
                mil_attn_heads=2,
                mil_attn_dropout=0.0,
            )

    def test_layer_symbols_remain_import_compatible(self) -> None:
        """Existing imports from src.model should resolve to src.layer classes."""
        self.assertIs(legacy_model_module.BBoxEmbedding, BBoxEmbedding)

    def test_forward_contract_and_strict_checkpoint_round_trip(self) -> None:
        """The assembled model must keep output shapes and strict state keys."""
        torch.manual_seed(7)
        model = self._build_model().eval()
        clips = torch.randn(2, 3, 2, 16, 16)
        indices = torch.tensor([[0, 1], [2, 3]])
        frame_counts = torch.tensor([4, 4])
        boxes = [
            torch.tensor(
                [
                    [[1.0, 1.0, 8.0, 12.0], [1.0, 1.0, 8.0, 12.0]],
                    [[7.0, 2.0, 15.0, 15.0], [7.0, 2.0, 15.0, 15.0]],
                ]
            )
            for _ in range(2)
        ]
        masks = [torch.ones(2, 2) for _ in range(2)]
        ball = torch.tensor(
            [
                [[6.0, 6.0, 8.0, 8.0], [7.0, 6.0, 9.0, 8.0]],
                [[8.0, 5.0, 10.0, 7.0], [9.0, 5.0, 11.0, 7.0]],
            ]
        )
        ball_mask = torch.ones(2, 2)

        with torch.no_grad():
            output = model(
                clips_video=clips,
                idx=indices,
                nums=frame_counts,
                bboxes=boxes,
                bbox_masks=masks,
                clips_ball=ball,
                clips_ball_mask=ball_mask,
                fps_in=4.0,
                topk=1,
            )

        self.assertEqual(tuple(output["logits_person"].shape), (2, 4))
        self.assertEqual(tuple(output["logits_clip"].shape), (2, 2, 4))
        self.assertTrue(torch.isfinite(output["logits_person"]).all())
        keys = set(model.state_dict())
        self.assertIn("bbox_emb.mlp.0.weight", keys)
        self.assertIn("actor_global.cross_attn.in_proj_weight", keys)
        self.assertIn("person_head.mlp.3.weight", keys)

        restored = self._build_model()
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
