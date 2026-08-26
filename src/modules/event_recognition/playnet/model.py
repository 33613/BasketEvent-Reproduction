"""Assemble the complete BasketEvent player-event recognition network.

The reusable building blocks live in :mod:`src.layer`.  This module owns only
the high-level model topology so checkpoint parameter names remain stable.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.core.config import SETTINGS
from src.modules.event_recognition.playnet.layers import (
    ActorGlobalCrossAttention,
    BBoxEmbedding,
    ClipRelationBlock,
    GatedClipPooling,
    MLPBlock,
    PersonEventClassifierHead,
    PersonFeaturePooler,
    PersonRelationBlock,
    TemporalROIAlign,
    TimeScaleEmbedding,
    TimeSformerBackbone,
    TopKClipPooling,
    TypeEmbedding,
)

__all__ = [
    "ActorGlobalCrossAttention",
    "BBoxEmbedding",
    "ClipRelationBlock",
    "GatedClipPooling",
    "MLPBlock",
    "PersonEventClassifierHead",
    "PlayerEventModel",
    "PersonFeaturePooler",
    "PersonRelationBlock",
    "TemporalROIAlign",
    "TimeScaleEmbedding",
    "TimeSformerBackbone",
    "TopKClipPooling",
    "TypeEmbedding",
]


class PlayerEventModel(nn.Module):
    """Recognize one event class per tracked player across a bag of clips.

    The model combines dense TimeSformer features, frame-level ROI features,
    actor-to-scene attention, within-frame entity relations, cross-clip
    relations, and multiple-instance pooling.  The ball is appended as the
    final virtual entity in every clip so it can participate in relation
    modeling, but the classification heads only return player predictions.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_name: str = str(SETTINGS.timesformer_model),
        local_files_only: bool = SETTINGS.hf_local_files_only,
        roi_out_size: Tuple[int, int] = (1, 1),
        roi_out_dim: Optional[int] = None,
        image_size: int = 224,
        add_bbox_embedding: bool = True,
        add_type_embedding: bool = True,
        mil_attn_heads: int = 8,
        mil_attn_dropout: float = 0.1,
        use_actor_global: bool = True,
        use_person_relation: bool = True,
        use_clip_relation: bool = True,
        pooling_mode: str = "gated",  # "gated" or "topk"
    ):
        """Initializes the player-event recognition network.

        Args:
            num_classes: Number of event classes, including background.
            pretrained_name: Hugging Face identifier or local TimeSformer path.
            local_files_only: Whether backbone loading is restricted to local
                files.
            roi_out_size: Spatial output size of temporal ROIAlign.
            roi_out_dim: Feature dimension after ROI projection.  When omitted,
                the TimeSformer hidden dimension is used.
            image_size: Width and height of the square input frames.
            add_bbox_embedding: Whether to add normalized box and motion
                embeddings to entity features.
            add_type_embedding: Whether to distinguish player and ball tokens.
            mil_attn_heads: Number of attention heads in relation modules.
            mil_attn_dropout: Dropout used by attention modules.
            use_actor_global: Whether entities attend to the dense scene map.
            use_person_relation: Whether entities interact within each frame.
            use_clip_relation: Whether the same entity interacts across clips.
            pooling_mode: Cross-clip pooling strategy, either ``"gated"`` or
                ``"topk"``.

        Raises:
            ValueError: If ``pooling_mode`` is unsupported.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.add_bbox_embedding = bool(add_bbox_embedding)
        self.add_type_embedding = bool(add_type_embedding)
        self.use_actor_global = bool(use_actor_global)
        self.use_person_relation = bool(use_person_relation)
        self.use_clip_relation = bool(use_clip_relation)
        self.pooling_mode = str(pooling_mode)

        self.backbone = TimeSformerBackbone(
            pretrained_name=pretrained_name,
            use_pretrained=True,
            local_files_only=local_files_only,
        )

        C = self.backbone.hidden_dim
        self.backbone_dim = C

        self.roi = TemporalROIAlign(
            image_size=image_size,
            output_size=roi_out_size,
            aligned=True,
        )

        roi_flat_dim = C * roi_out_size[0] * roi_out_size[1]
        self.mil_feat_dim = roi_out_dim if roi_out_dim is not None else C

        # 如果 ROIAlign 不是 1x1，先投影回 mil_feat_dim。
        if roi_flat_dim != self.mil_feat_dim:
            self.roi_proj = nn.Sequential(
                nn.LayerNorm(roi_flat_dim),
                nn.Linear(roi_flat_dim, self.mil_feat_dim),
            )
        else:
            self.roi_proj = nn.Identity()

        if self.add_bbox_embedding:
            self.bbox_emb = BBoxEmbedding(
                out_dim=self.mil_feat_dim,
                image_size=image_size,
                use_motion=True,
                hidden_dim=256,
                dropout=0.1,
            )

        if self.add_type_embedding:
            self.type_emb = TypeEmbedding(dim=self.mil_feat_dim, num_types=2)

        if self.use_actor_global:
            # 若 mil_feat_dim 与 backbone_dim 不同，需要把 featmap 投影到 mil_feat_dim。
            if C != self.mil_feat_dim:
                self.featmap_proj = nn.Conv3d(
                    C, self.mil_feat_dim, kernel_size=1, bias=False
                )
            else:
                self.featmap_proj = nn.Identity()

            self.actor_global = ActorGlobalCrossAttention(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                add_spatial_pos=True,
                grid_h=self.backbone.grid_h,
                grid_w=self.backbone.grid_w,
                mlp_ratio=4.0,
            )
        else:
            self.featmap_proj = nn.Identity()
            self.actor_global = None

        self.feature_pooler = PersonFeaturePooler(
            in_dim=self.mil_feat_dim,
            num_heads=mil_attn_heads,
            dropout=mil_attn_dropout,
        )

        if self.use_person_relation:
            self.person_relation = PersonRelationBlock(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                mlp_ratio=4.0,
            )
        else:
            self.person_relation = None

        if self.use_clip_relation:
            self.clip_relation = ClipRelationBlock(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                mlp_ratio=4.0,
            )
        else:
            self.clip_relation = None

        if self.pooling_mode == "gated":
            self.clip_pool = GatedClipPooling(self.mil_feat_dim)
        elif self.pooling_mode == "topk":
            self.clip_pool = TopKClipPooling(self.mil_feat_dim)
        else:
            raise ValueError(
                f"pooling_mode must be 'gated' or 'topk', got {pooling_mode}"
            )

        self.clip_head = PersonEventClassifierHead(
            in_dim=self.mil_feat_dim,
            num_classes=num_classes,
            dropout=0.1,
            hidden_dim=512,
        )
        self.person_head = PersonEventClassifierHead(
            in_dim=self.mil_feat_dim,
            num_classes=num_classes,
            dropout=0.1,
            hidden_dim=512,
        )

    @torch.no_grad()
    def _assert_aligned_person_count(self, bboxes: List[torch.Tensor]) -> int:
        """Validates that every clip contains the same tracked-player count.

        Args:
            bboxes: Per-clip boxes with shape ``(N, T, 4)``.

        Returns:
            The shared number of tracked players.  An empty clip list returns
            zero.

        Raises:
            RuntimeError: If clips contain different player counts.
        """
        if len(bboxes) == 0:
            return 0
        N = int(bboxes[0].shape[0])
        for bb in bboxes:
            if int(bb.shape[0]) != N:
                raise RuntimeError(
                    f"MIL requires aligned persons across clips, but got person counts "
                    f"{[int(x.shape[0]) for x in bboxes]}"
                )
        return N

    @staticmethod
    def _build_extended_boxes_with_ball(
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        clips_ball: torch.Tensor,
        clips_ball_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Appends one ball trajectory to the entity boxes of every clip.

        Args:
            bboxes: Per-clip player boxes, each shaped ``(N, T, 4)``.
            bbox_masks: Per-clip player-validity masks shaped ``(N, T)``.
            clips_ball: Ball boxes shaped ``(M, T, 4)``.
            clips_ball_mask: Ball-validity masks shaped ``(M, T)``.

        Returns:
            A pair containing boxes and masks whose final entity is the ball.
        """
        M = len(bboxes)
        extended_bboxes = []
        extended_bbox_masks = []
        for m in range(M):
            person_bb = bboxes[m]
            person_mask = bbox_masks[m]

            ball_bb = clips_ball[m : m + 1, :, :].to(
                device=person_bb.device, dtype=person_bb.dtype
            )
            ball_mask = clips_ball_mask[m : m + 1, :].to(
                device=person_mask.device, dtype=person_mask.dtype
            )

            extended_bboxes.append(torch.cat([person_bb, ball_bb], dim=0))
            extended_bbox_masks.append(torch.cat([person_mask, ball_mask], dim=0))

        return extended_bboxes, extended_bbox_masks

    def _add_bbox_and_type_embedding(
        self,
        person_feats_flat: torch.Tensor,
        extended_bboxes: List[torch.Tensor],
        extended_bbox_masks: List[torch.Tensor],
        M: int,
        N_plus_1: int,
        T: int,
    ) -> torch.Tensor:
        """Adds optional geometry and entity-type embeddings.

        Args:
            person_feats_flat: Flattened entity features with shape
                ``(M * (N + 1), T, C)``.
            extended_bboxes: Per-clip player and ball boxes.
            extended_bbox_masks: Per-clip validity masks for those boxes.
            M: Number of clips in the video bag.
            N_plus_1: Number of players plus the ball token per clip.
            T: Number of sampled frames per clip.

        Returns:
            Enriched features with the same shape as ``person_feats_flat``.
        """
        device = person_feats_flat.device

        if self.add_bbox_embedding and person_feats_flat.numel() > 0:
            bbox_cat = torch.cat(
                [bb.to(device=device, dtype=torch.float32) for bb in extended_bboxes],
                dim=0,
            )  # (M*(N+1),T,4)
            mask_cat = torch.cat(
                [
                    mk.to(device=device, dtype=torch.float32)
                    for mk in extended_bbox_masks
                ],
                dim=0,
            )  # (M*(N+1),T)
            bbox_feat = self.bbox_emb(bbox_cat, mask_cat)
            person_feats_flat = person_feats_flat + bbox_feat

        if self.add_type_embedding and person_feats_flat.numel() > 0:
            # 对每个 clip: 前 N 个是 person，最后 1 个是 ball。
            type_ids_one_clip = torch.zeros(
                (N_plus_1,), device=device, dtype=torch.long
            )
            type_ids_one_clip[-1] = 1
            type_ids = type_ids_one_clip.repeat(M)  # (M*(N+1),)
            type_ids = type_ids[:, None].expand(M * N_plus_1, T)  # (M*(N+1),T)
            person_feats_flat = self.type_emb(person_feats_flat, type_ids)

        return person_feats_flat

    def extract_roi_debug_features(
        self,
        clips_video: torch.Tensor,
        idx: torch.Tensor,
        nums: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        fps_in: float = 25.0,
    ) -> Dict[str, torch.Tensor]:
        """Extracts backbone and ROI features for debugging or visualization.

        Args:
            clips_video: Video clips shaped ``(M, C, T, H, W)``.
            idx: Source-frame indices shaped ``(M, T)``.
            nums: Valid sampled-frame counts shaped ``(M,)``.
            bboxes: Per-clip player boxes shaped ``(N, T, 4)``.
            bbox_masks: Per-clip player-validity masks shaped ``(N, T)``.
            fps_in: Frame rate of the source video.

        Returns:
            A dictionary containing the dense backbone feature map, its
            optional global-attention projection, raw and projected player ROI
            features, player masks, and the aligned player count.
        """
        device = clips_video.device
        M = int(clips_video.shape[0])
        N = self._assert_aligned_person_count(bboxes)

        backbone_out = self.backbone(
            clips_video,
            idx=idx,
            nums=nums,
            fps_in=fps_in,
        )
        featmap = backbone_out["featmap"]  # (M,C,T,Hf,Wf)

        person_feats_flat, person_mask_flat, _ = self.roi(
            featmap,
            bboxes,
            bbox_masks,
        )

        _, _, T, _, _ = featmap.shape
        roi_flat_dim = (
            int(person_feats_flat.shape[-1])
            if person_feats_flat.numel() > 0
            else self.backbone_dim
        )
        person_feats_raw = person_feats_flat.view(M, N, T, roi_flat_dim)
        person_mask = person_mask_flat.view(M, N, T)

        person_feats_projected = self.roi_proj(person_feats_flat)
        person_feats_projected = person_feats_projected.view(M, N, T, self.mil_feat_dim)

        featmap_for_global = (
            self.featmap_proj(featmap) if self.use_actor_global else featmap
        )

        return {
            "featmap": featmap,
            "featmap_for_global": featmap_for_global,
            "person_feats_raw": person_feats_raw,
            "person_feats_projected": person_feats_projected,
            "person_mask": person_mask,
            "num_persons": torch.tensor([N], device=device, dtype=torch.long),
        }

    def forward_mil_one_video(
        self,
        clips_video: torch.Tensor,  # (M,C,T,H,W)
        idx: torch.Tensor,  # (M,T)
        nums: torch.Tensor,  # (M,)
        bboxes: List[torch.Tensor],  # list length M; each (N,T,4)
        bbox_masks: List[torch.Tensor],  # list length M; each (N,T)
        clips_ball: torch.Tensor,  # (M,T,4)
        clips_ball_mask: torch.Tensor,  # (M,T)
        fps_in: float = 25.0,
        topk: int = 2,
        return_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Runs multiple-instance event recognition for one video bag.

        Args:
            clips_video: Video clips shaped ``(M, C, T, H, W)``.
            idx: Source-frame indices shaped ``(M, T)``.
            nums: Valid sampled-frame counts shaped ``(M,)``.
            bboxes: Per-clip player boxes, each shaped ``(N, T, 4)``.
            bbox_masks: Per-clip player-validity masks shaped ``(N, T)``.
            clips_ball: Ball boxes shaped ``(M, T, 4)``.
            clips_ball_mask: Ball-validity masks shaped ``(M, T)``.
            fps_in: Frame rate of the source video.
            topk: Number of clips selected by top-k pooling.
            return_weights: Whether to return pooling and actor-global
                attention weights.

        Returns:
            A dictionary containing clip-level and video-level player features,
            logits, validity masks, and optional attention weights.
        """
        device = clips_video.device
        M = int(clips_video.shape[0])

        if M == 0:
            return {
                "pooled_feats_flat": torch.zeros((0, self.mil_feat_dim), device=device),
                "pooled_feats_clip": torch.zeros(
                    (0, 0, self.mil_feat_dim), device=device
                ),
                "pooled_feats_person": torch.zeros(
                    (0, self.mil_feat_dim), device=device
                ),
                "logits_person": torch.zeros((0, self.num_classes), device=device),
                "logits_clip": torch.zeros((0, 0, self.num_classes), device=device),
            }

        N = self._assert_aligned_person_count(bboxes)
        N_plus_1 = N + 1

        # person_valid: (N,), 某个球员只要任一 clip/frame 有 bbox 即有效。
        if N > 0:
            person_valid = torch.stack(
                [mask.to(device=device).any(dim=1) for mask in bbox_masks], dim=0
            ).any(
                dim=0
            )  # (N,)
        else:
            person_valid = torch.zeros((0,), device=device, dtype=torch.bool)

        # 加 ball token。
        extended_bboxes, extended_bbox_masks = self._build_extended_boxes_with_ball(
            bboxes=bboxes,
            bbox_masks=bbox_masks,
            clips_ball=clips_ball,
            clips_ball_mask=clips_ball_mask,
        )

        # 1. backbone global featmap。
        backbone_out = self.backbone(
            clips_video,
            idx=idx,
            nums=nums,
            fps_in=fps_in,
        )
        featmap = backbone_out["featmap"]  # (M,C,T,Hf,Wf)
        _, _, T, _, _ = featmap.shape

        # 2. ROIAlign 得到每个 token 的帧级特征。
        person_feats_flat, person_mask_flat, person_splits = self.roi(
            featmap,
            extended_bboxes,
            extended_bbox_masks,
        )  # (M*(N+1),T,roi_dim), (M*(N+1),T)

        if N == 0:
            return {
                "pooled_feats_flat": torch.zeros((0, self.mil_feat_dim), device=device),
                "pooled_feats_clip": torch.zeros(
                    (M, 0, self.mil_feat_dim), device=device
                ),
                "pooled_feats_person": torch.zeros(
                    (0, self.mil_feat_dim), device=device
                ),
                "logits_person": torch.zeros((0, self.num_classes), device=device),
                "logits_clip": torch.zeros((M, 0, self.num_classes), device=device),
            }

        # ROI projection。
        person_feats_flat = self.roi_proj(person_feats_flat)

        # 3. bbox embedding + type embedding。
        person_feats_flat = self._add_bbox_and_type_embedding(
            person_feats_flat=person_feats_flat,
            extended_bboxes=extended_bboxes,
            extended_bbox_masks=extended_bbox_masks,
            M=M,
            N_plus_1=N_plus_1,
            T=T,
        )

        # reshape to (M,N+1,T,C)
        C = self.mil_feat_dim
        person_feats = person_feats_flat.view(M, N_plus_1, T, C)
        person_mask = person_mask_flat.view(M, N_plus_1, T)

        # 4. Actor-Global Cross Attention：每个人物/球 token 与全局特征交互。
        attn_weights_actor_global = None
        if self.use_actor_global:
            featmap_for_global = self.featmap_proj(featmap)
            forward_result = self.actor_global(
                person_feats=person_feats,
                featmap=featmap_for_global,
                person_mask=person_mask,
                return_attn=return_weights,
            )
            if return_weights and isinstance(forward_result, tuple):
                person_feats, attn_weights_actor_global = forward_result
            else:
                person_feats = forward_result  # (M,N+1,T,C)

        # 5. Person-Person Interaction：先在每一帧上做球员/ball 之间交互，再沿时间聚合。
        if self.use_person_relation:
            person_feats_frame = (
                person_feats.permute(0, 2, 1, 3).contiguous().view(M * T, N_plus_1, C)
            )  # (M*T,N+1,C)
            person_mask_frame = (
                person_mask.permute(0, 2, 1).contiguous().view(M * T, N_plus_1)
            )  # (M*T,N+1)
            person_feats_frame = self.person_relation(
                person_feats_frame,
                valid_mask=person_mask_frame,
            )
            person_feats = (
                person_feats_frame.view(M, T, N_plus_1, C)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            person_feats = person_feats * person_mask.to(
                dtype=person_feats.dtype
            ).unsqueeze(-1)

        # 6. clip 内 temporal pooling：每个 clip 内，T 帧 -> 1 个 token。
        pooled_feats_flat = self.feature_pooler(
            person_feats.view(M * N_plus_1, T, C),
            person_mask.view(M * N_plus_1, T),
        )  # (M*(N+1),C)
        pooled_feats_clip = pooled_feats_flat.view(M, N_plus_1, C)  # (M,N+1,C)

        # clip-token valid mask: (M,N+1)，只要该 token 在该 clip 任一帧有效。
        clip_token_valid = person_mask.any(dim=2)  # (M,N+1)

        # 7. Clip-Clip Interaction：同一球员跨 clip 交互。
        if self.use_clip_relation:
            pooled_feats_clip = self.clip_relation(
                pooled_feats_clip,
                valid_mask=clip_token_valid,
            )

        # 8. clip-level logits：只对 person 分类，ball 不分类。
        logits_clip_all = self.clip_head(pooled_feats_clip.view(M * N_plus_1, C)).view(
            M, N_plus_1, self.num_classes
        )
        logits_clip = logits_clip_all[:, :N, :]

        # 9. 去掉 ball，只对 person 做 MIL pooling。
        pooled_feats_clip_persons = pooled_feats_clip[:, :N, :]  # (M,N,C)
        person_clip_valid = clip_token_valid[:, :N]  # (M,N)

        if return_weights:
            if self.pooling_mode == "topk":
                pooled_feats_person, gate_logits, gate_weights = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    topk=topk,
                    return_weights=True,
                )
            else:
                pooled_feats_person, gate_logits, gate_weights = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    return_weights=True,
                )
        else:
            if self.pooling_mode == "topk":
                pooled_feats_person = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    topk=topk,
                    return_weights=False,
                )
            else:
                pooled_feats_person = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    return_weights=False,
                )

        # 10. person-level classification。
        logits_person = self.person_head(pooled_feats_person, person_valid)

        out = {
            "pooled_feats_flat": pooled_feats_flat,  # (M*(N+1),C), includes ball
            "pooled_feats_clip_all": pooled_feats_clip,  # (M,N+1,C), includes ball
            "pooled_feats_clip": pooled_feats_clip_persons,  # (M,N,C), persons only
            "pooled_feats_person": pooled_feats_person,  # (N,C)
            "logits_person": logits_person,  # (N,num_classes)
            "logits_clip": logits_clip,  # (M,N,num_classes)
            "person_valid": person_valid,  # (N,)
            "person_clip_valid": person_clip_valid,  # (M,N)
        }

        if return_weights:
            out["gate_logits"] = gate_logits  # (M,N)
            out["gate_weights"] = gate_weights  # (M,N,1)
            if attn_weights_actor_global is not None:
                out["attn_weights_actor_global"] = (
                    attn_weights_actor_global  # (M,T,N,Hf,Wf)
                )

        return out

    def forward(
        self,
        clips_video: torch.Tensor,
        idx: torch.Tensor,
        nums: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        clips_ball: torch.Tensor,
        clips_ball_mask: torch.Tensor,
        fps_in: float,
        topk: int,
        return_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Delegates the standard module call to one-video MIL inference.

        Args:
            clips_video: Video clips shaped ``(M, C, T, H, W)``.
            idx: Source-frame indices shaped ``(M, T)``.
            nums: Valid sampled-frame counts shaped ``(M,)``.
            bboxes: Per-clip player boxes shaped ``(N, T, 4)``.
            bbox_masks: Per-clip player-validity masks shaped ``(N, T)``.
            clips_ball: Ball boxes shaped ``(M, T, 4)``.
            clips_ball_mask: Ball-validity masks shaped ``(M, T)``.
            fps_in: Frame rate of the source video.
            topk: Number of clips selected by top-k pooling.
            return_weights: Whether to include attention diagnostics.

        Returns:
            The same output dictionary as :meth:`forward_mil_one_video`.
        """
        return self.forward_mil_one_video(
            clips_video=clips_video,
            idx=idx,
            nums=nums,
            bboxes=bboxes,
            bbox_masks=bbox_masks,
            clips_ball=clips_ball,
            clips_ball_mask=clips_ball_mask,
            fps_in=fps_in,
            topk=topk,
            return_weights=return_weights,
        )
