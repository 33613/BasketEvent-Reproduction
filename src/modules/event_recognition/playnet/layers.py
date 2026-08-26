"""Reusable neural-network layers for BasketEvent.

This module contains the feature extraction, geometric encoding, interaction,
pooling, and classification building blocks used by :class:`PlayerEventModel`.
Keeping these components separate makes their tensor contracts independently
testable while preserving the registered attribute names of the assembled model.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from src.core.config import SETTINGS

__all__ = [
    "ActorGlobalCrossAttention",
    "BBoxEmbedding",
    "ClipRelationBlock",
    "GatedClipPooling",
    "MLPBlock",
    "PersonEventClassifierHead",
    "PersonFeaturePooler",
    "PersonRelationBlock",
    "TemporalROIAlign",
    "TimeScaleEmbedding",
    "TimeSformerBackbone",
    "TopKClipPooling",
    "TypeEmbedding",
]


# =========================================================
# 1) BBox embedding
# =========================================================
class BBoxEmbedding(nn.Module):
    """
    bbox 序列 (N,T,4) + mask (N,T) -> (N,T,C)
    输入 bbox: xyxy in [0, image_size]
    特征包含：
      - 几何位置: x1,y1,x2,y2,w,h,cx,cy,area,aspect_ratio
      - 运动信息: dx,dy,dw,dh
    """

    def __init__(
        self,
        out_dim: int,
        image_size: int = 224,
        use_motion: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize the geometric encoder.

        Args:
            out_dim: Output feature dimension.
            image_size: Coordinate system used by input boxes.
            use_motion: Whether to append frame-to-frame geometry deltas.
            hidden_dim: Hidden dimension of the projection MLP.
            dropout: Dropout probability inside the projection MLP.
        """
        super().__init__()
        self.image_size = float(image_size)
        self.use_motion = use_motion

        in_dim = 10 + (4 if use_motion else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self, bboxes_xyxy: torch.Tensor, bbox_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode spatial and optional motion features.

        Args:
            bboxes_xyxy: Boxes shaped ``(N, T, 4)`` in ``xyxy`` format.
            bbox_mask: Valid-frame mask shaped ``(N, T)``.

        Returns:
            Masked embeddings shaped ``(N, T, out_dim)``.
        """
        b = bboxes_xyxy.to(torch.float32)
        b = torch.clamp(b, 0.0, self.image_size)

        x1, y1, x2, y2 = b.unbind(dim=-1)
        w = (x2 - x1).clamp_min(1.0)
        h = (y2 - y1).clamp_min(1.0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        area = (w * h) / (self.image_size * self.image_size)
        ar = (w / h).clamp(0.0, 10.0)

        s = self.image_size
        x1n, y1n, x2n, y2n = x1 / s, y1 / s, x2 / s, y2 / s
        wn, hn = w / s, h / s
        cxn, cyn = cx / s, cy / s

        feats = [x1n, y1n, x2n, y2n, wn, hn, cxn, cyn, area, ar]

        if self.use_motion:
            dx = torch.zeros_like(cxn)
            dy = torch.zeros_like(cyn)
            dw = torch.zeros_like(wn)
            dh = torch.zeros_like(hn)
            dx[:, 1:] = cxn[:, 1:] - cxn[:, :-1]
            dy[:, 1:] = cyn[:, 1:] - cyn[:, :-1]
            dw[:, 1:] = wn[:, 1:] - wn[:, :-1]
            dh[:, 1:] = hn[:, 1:] - hn[:, :-1]
            feats += [dx, dy, dw, dh]

        feat = torch.stack(feats, dim=-1)  # (N,T,in_dim)
        m = bbox_mask.to(dtype=feat.dtype).unsqueeze(-1)
        feat = feat * m

        emb = self.mlp(feat)
        emb = emb * m
        return emb


# =========================================================
# 2) Time scale embedding
# =========================================================
class TimeScaleEmbedding(nn.Module):
    """
    idx + nums -> (B,T,D)
    feats: t_norm, t_sec, dt_sec
    """

    def __init__(self, d_model: int, use_dt: bool = True):
        """Initialize the continuous time projection.

        Args:
            d_model: Output embedding dimension.
            use_dt: Whether to include the interval since the previous frame.
        """
        super().__init__()
        self.use_dt = use_dt
        in_dim = 2 + (1 if use_dt else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self, idx: torch.Tensor, nums: torch.Tensor, fps: float = 25.0
    ) -> torch.Tensor:
        """Embed normalized and absolute frame times.

        Args:
            idx: Sampled frame indices shaped ``(B, T)``.
            nums: Original frame counts shaped ``(B,)``.
            fps: Source-video frame rate.

        Returns:
            Time embeddings shaped ``(B, T, d_model)``.
        """
        idx_f = idx.to(torch.float32)
        nums_f = nums.to(idx.device).to(torch.float32).clamp_min(2.0)

        t_norm = idx_f / (nums_f[:, None] - 1.0)
        t_sec = idx_f / float(fps)

        feats = [t_norm, t_sec]
        if self.use_dt:
            dt = torch.zeros_like(t_sec)
            dt[:, 1:] = t_sec[:, 1:] - t_sec[:, :-1]
            feats.append(dt)

        feat = torch.stack(feats, dim=-1)
        return self.mlp(feat)


# =========================================================
# 3) TimeSformer backbone
# =========================================================
class TimeSformerBackbone(nn.Module):
    """
    输入 video: (B,C,T,H,W) 或 (B,T,C,H,W)
    输出 featmap: (B,D,T,H',W')

    注意：HF TimeSformer 默认输出 token 序列。
    这里去掉 CLS token 后 reshape 成时空 feature map。
    """

    def __init__(
        self,
        pretrained_name: str = str(SETTINGS.timesformer_model),
        use_pretrained: bool = False,
        local_files_only: bool = SETTINGS.hf_local_files_only,
    ):
        """Load a pretrained TimeSformer or initialize it from configuration.

        Args:
            pretrained_name: Local model directory or Hugging Face identifier.
            use_pretrained: Whether to load model weights instead of config only.
            local_files_only: Whether network access is forbidden.

        Raises:
            ImportError: If the installed Transformers package lacks TimeSformer.
        """
        super().__init__()
        try:
            from transformers import TimesformerModel, TimesformerConfig
        except Exception as e:
            raise ImportError(
                "请先 pip install transformers，且版本需包含 TimesformerModel。原始错误: "
                + str(e)
            )

        if use_pretrained:
            self.model = TimesformerModel.from_pretrained(
                pretrained_name,
                local_files_only=local_files_only,
            )
        else:
            # 如果 pretrained_name 是本地 config 或模型名，这里仍可用 config 初始化。
            cfg = TimesformerConfig.from_pretrained(
                pretrained_name,
                local_files_only=local_files_only,
            )
            self.model = TimesformerModel(cfg)

        self.hidden_dim = self.model.config.hidden_size

        image_size = getattr(self.model.config, "image_size", 224)
        patch_size = getattr(self.model.config, "patch_size", 16)
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size

    @staticmethod
    def _ensure_b_t_c_h_w(video: torch.Tensor) -> torch.Tensor:
        """Normalize channel-first or time-first video to ``(B, T, C, H, W)``."""
        if video.dim() != 5:
            raise ValueError(f"video must be 5D, got {tuple(video.shape)}")
        # Accept both (B,C,T,H,W) and (B,T,C,H,W).
        # Prefer interpreting ambiguous small-T inputs as channel-first because the
        # training pipeline in this repo constructs clips as (B,C,T,H,W).
        if video.shape[1] in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def forward(
        self,
        video: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        nums: Optional[torch.Tensor] = None,
        fps_in: float = 25.0,
    ) -> Dict[str, Any]:
        """Extract a dense spatiotemporal feature map.

        Args:
            video: Clips in ``(B, C, T, H, W)`` or ``(B, T, C, H, W)`` format.
            idx: Optional sampled frame indices retained for API compatibility.
            nums: Optional source frame counts retained for API compatibility.
            fps_in: Source frame rate retained for API compatibility.

        Returns:
            Mapping containing ``featmap`` shaped ``(B, D, T, H', W')`` and
            backbone geometry metadata under ``aux``.

        Raises:
            ValueError: If TimeSformer returns an unexpected token count.
        """
        x = self._ensure_b_t_c_h_w(video)  # (B,T,C,H,W)
        B, T, C, H, W = x.shape

        # HF TimeSformer layers read config.num_frames during forward, so keep it
        # in sync with the current clip length. Its embedding table is resized
        # internally when T differs from the pretrained checkpoint setting.
        self.model.config.num_frames = T
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state  # (B,1+T*P,D)
        tokens = tokens[:, 1:, :]  # (B,T*P,D)

        P = self.grid_h * self.grid_w
        if tokens.shape[1] != T * P:
            raise ValueError(
                f"Token length mismatch: got {tokens.shape[1]}, expected {T}*{P}={T*P}. "
                f"Check image_size/patch_size or input size."
            )

        fmap = tokens.view(B, T, P, self.hidden_dim).view(
            B, T, self.grid_h, self.grid_w, self.hidden_dim
        )  # (B,T,H',W',D)

        fmap = fmap.permute(0, 4, 1, 2, 3).contiguous()  # (B,D,T,H',W')

        return {
            "featmap": fmap,
            "aux": {
                "hidden_dim": self.hidden_dim,
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
            },
        }


# =========================================================
# 4) Temporal ROIAlign
# =========================================================
class TemporalROIAlign(nn.Module):
    """
    对每一帧 feature map 做 ROIAlign。

    Inputs:
      featmap:    (B, C, T, Hf, Wf)
      bboxes:     list length B; each (Ni, T, 4) xyxy@image_size
      bbox_mask:  list length B; each (Ni, T) 1/0

    Outputs:
      person_feats:  (sum Ni, T, C) when output_size=(1,1)
                     (sum Ni, T, C*oh*ow) when output_size != (1,1)
      person_mask:   (sum Ni, T)
      person_splits: list length B, each = Ni
    """

    def __init__(
        self,
        image_size: int = 224,
        output_size: Tuple[int, int] = (1, 1),
        aligned: bool = True,
    ):
        """Initialize frame-wise ROIAlign.

        Args:
            image_size: Coordinate system used by input boxes.
            output_size: Spatial ROIAlign output height and width.
            aligned: Whether to use the half-pixel aligned ROIAlign variant.
        """
        super().__init__()
        self.image_size = float(image_size)
        self.output_size = output_size
        self.aligned = aligned

    def forward(
        self,
        featmap: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_mask: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Pool every valid actor box from every feature-map frame.

        Args:
            featmap: Feature map shaped ``(B, C, T, Hf, Wf)``.
            bboxes: Per-video boxes, each shaped ``(Ni, T, 4)``.
            bbox_mask: Per-video validity masks, each shaped ``(Ni, T)``.

        Returns:
            Flattened actor features, their masks, and actor counts per video.
        """
        device = featmap.device
        dtype = featmap.dtype

        if featmap.dim() != 5:
            raise ValueError(
                f"featmap must be (B,C,T,Hf,Wf), got {tuple(featmap.shape)}"
            )

        B, C, T, Hf, Wf = featmap.shape
        if len(bboxes) != B or len(bbox_mask) != B:
            raise ValueError(
                f"Length of bboxes/mask list must equal B={B}, got {len(bboxes)} and {len(bbox_mask)}"
            )

        out_h, out_w = self.output_size
        pooled_dim = C * out_h * out_w

        person_splits = [int(bb.shape[0]) for bb in bboxes]
        total_persons = sum(person_splits)

        if total_persons == 0:
            return (
                torch.zeros((0, T, pooled_dim), device=device, dtype=dtype),
                torch.zeros((0, T), device=device, dtype=torch.float32),
                person_splits,
            )

        person_feats = torch.zeros(
            (total_persons, T, pooled_dim), device=device, dtype=dtype
        )
        person_mask = torch.zeros(
            (total_persons, T), device=device, dtype=torch.float32
        )

        feat_2d = featmap.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, Hf, Wf)

        sx = float(Wf) / self.image_size
        sy = float(Hf) / self.image_size

        rois = []
        roi_person_index = []
        roi_t_index = []

        cursor = 0
        for b in range(B):
            bb = bboxes[b].to(device=device, dtype=torch.float32)
            mk = bbox_mask[b].to(device=device, dtype=torch.float32)
            Ni = int(bb.shape[0])
            if Ni == 0:
                continue

            person_mask[cursor : cursor + Ni] = mk

            for pi in range(Ni):
                for t in range(T):
                    if mk[pi, t].item() <= 0:
                        continue
                    x1, y1, x2, y2 = bb[pi, t].tolist()

                    x1 *= sx
                    x2 *= sx
                    y1 *= sy
                    y2 *= sy

                    x1 = max(0.0, min(x1, Wf - 1.0))
                    x2 = max(0.0, min(x2, Wf - 1.0))
                    y1 = max(0.0, min(y1, Hf - 1.0))
                    y2 = max(0.0, min(y2, Hf - 1.0))

                    if x2 <= x1:
                        x2 = min(Wf - 1.0, x1 + 1.0)
                    if y2 <= y1:
                        y2 = min(Hf - 1.0, y1 + 1.0)

                    frame_index = b * T + t
                    rois.append([frame_index, x1, y1, x2, y2])
                    roi_person_index.append(cursor + pi)
                    roi_t_index.append(t)

            cursor += Ni

        if len(rois) == 0:
            return person_feats, person_mask, person_splits

        rois = torch.tensor(rois, device=device, dtype=torch.float32)
        pooled = roi_align(
            input=feat_2d,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=self.aligned,
        )  # (R,C,oh,ow)

        pooled = pooled.flatten(1)  # (R,C*oh*ow)

        roi_person_index = torch.tensor(
            roi_person_index, device=device, dtype=torch.long
        )
        roi_t_index = torch.tensor(roi_t_index, device=device, dtype=torch.long)
        person_feats[roi_person_index, roi_t_index] = pooled.to(dtype)

        return person_feats, person_mask, person_splits


# =========================================================
# 5) Utility blocks
# =========================================================
class MLPBlock(nn.Module):
    """Apply a pre-normalized feed-forward residual block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """Initialize the residual MLP.

        Args:
            dim: Input and output feature dimension.
            mlp_ratio: Hidden expansion ratio.
            dropout: Dropout probability after each linear projection.
        """
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the residual sum of ``x`` and its feed-forward transform."""
        return x + self.net(x)


class TypeEmbedding(nn.Module):
    """
    给 person / ball 加不同类型 embedding。
    type_id: 0=person, 1=ball
    """

    def __init__(self, dim: int, num_types: int = 2):
        """Initialize learned embeddings for actor categories.

        Args:
            dim: Token feature dimension.
            num_types: Number of entity categories.
        """
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)

    def forward(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """
        x:        (..., C)
        type_ids: broadcastable to x.shape[:-1]
        """
        return x + self.emb(type_ids.to(x.device).long())


# =========================================================
# 6) Actor-Global Interaction
# =========================================================
class ActorGlobalCrossAttention(nn.Module):
    """
    第 1 阶段：每个球员 token 与对应帧的全局 feature map 交互。

    输入：
      person_feats: (M, N, T, C)
      featmap:      (M, C, T, Hf, Wf)
      person_mask:  (M, N, T), 1=valid

    输出：
      enhanced person_feats: (M, N, T, C)

    实现逻辑：
      对每个 (m,t)，有 N 个 person queries；
      对应全局 feature map F[m,:,t,:,:] 展成 Hf*Wf 个 global tokens；
      Q = person token, K/V = global tokens。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_spatial_pos: bool = True,
        grid_h: int = 14,
        grid_w: int = 14,
        mlp_ratio: float = 4.0,
    ):
        """Initialize actor-to-global cross-attention.

        Args:
            dim: Shared actor and feature-map token dimension.
            num_heads: Number of attention heads.
            dropout: Attention and residual dropout probability.
            add_spatial_pos: Whether to learn feature-map positions.
            grid_h: Reference feature-map height.
            grid_w: Reference feature-map width.
            mlp_ratio: Feed-forward expansion ratio.
        """
        super().__init__()
        self.dim = dim
        self.add_spatial_pos = add_spatial_pos
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        if add_spatial_pos:
            self.spatial_pos = nn.Parameter(torch.zeros(1, grid_h * grid_w, dim))
            nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(
        self,
        person_feats: torch.Tensor,
        featmap: torch.Tensor,
        person_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        """Enhance actor tokens with the corresponding frame's global tokens.

        Args:
            person_feats: Actor tokens shaped ``(M, N, T, C)``.
            featmap: Global features shaped ``(M, C, T, Hf, Wf)``.
            person_mask: Optional valid-token mask shaped ``(M, N, T)``.
            return_attn: Whether to return spatial attention maps.

        Returns:
            Enhanced tokens, optionally paired with attention maps.
        """
        if person_feats.dim() != 4:
            raise ValueError(
                f"person_feats must be (M,N,T,C), got {tuple(person_feats.shape)}"
            )
        if featmap.dim() != 5:
            raise ValueError(f"featmap must be (M,C,T,H,W), got {tuple(featmap.shape)}")

        M, N, T, C = person_feats.shape
        Mf, Cf, Tf, Hf, Wf = featmap.shape
        if (M, T, C) != (Mf, Tf, Cf):
            raise ValueError(
                f"Shape mismatch: person_feats={(M,N,T,C)}, featmap={(Mf,Cf,Tf,Hf,Wf)}"
            )

        # global tokens: (M,T,H*W,C) -> (M*T, H*W, C)
        global_tokens = (
            featmap.permute(0, 2, 3, 4, 1).contiguous().view(M * T, Hf * Wf, C)
        )

        if self.add_spatial_pos:
            if Hf == self.grid_h and Wf == self.grid_w:
                pos = self.spatial_pos
            else:
                # 若输入尺寸改变，插值 spatial pos。
                pos_2d = self.spatial_pos.view(1, self.grid_h, self.grid_w, C).permute(
                    0, 3, 1, 2
                )
                pos_2d = F.interpolate(
                    pos_2d, size=(Hf, Wf), mode="bilinear", align_corners=False
                )
                pos = pos_2d.permute(0, 2, 3, 1).contiguous().view(1, Hf * Wf, C)
            global_tokens = global_tokens + pos.to(global_tokens.dtype)

        # queries: (M,T,N,C) -> (M*T,N,C)
        q = person_feats.permute(0, 2, 1, 3).contiguous().view(M * T, N, C)

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(global_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=return_attn,
        )

        out = q + self.dropout(attn_out)
        out = self.out_norm(out)
        out = self.ffn(out)

        out = out.view(M, T, N, C).permute(0, 2, 1, 3).contiguous()  # (M,N,T,C)

        if person_mask is not None:
            out = out * person_mask.to(dtype=out.dtype).unsqueeze(-1)

        if return_attn:
            # attn_weights: (M*T, N, Hf*Wf) -> reshape to (M, T, N, Hf, Wf)
            attn_weights_reshaped = attn_weights.view(M, T, N, Hf, Wf)
            return out, attn_weights_reshaped
        return out


# =========================================================
# 7) Temporal pooler inside each clip
# =========================================================
class PersonFeaturePooler(nn.Module):
    """
    对每个 person 在一个 clip 内的 T 帧特征做 temporal attention pooling。
    输入:  (N,T,C)
    输出:  (N,C)
    """

    def __init__(self, in_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """Initialize temporal self-attention pooling.

        Args:
            in_dim: Actor feature dimension.
            num_heads: Number of temporal attention heads.
            dropout: Attention and feed-forward dropout probability.
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(in_dim)
        self.ffn = MLPBlock(in_dim, mlp_ratio=4.0, dropout=dropout)
        self.out_norm = nn.LayerNorm(in_dim)

    def forward(
        self, person_feats: torch.Tensor, person_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool each actor's valid frame tokens into one clip token.

        Args:
            person_feats: Frame tokens shaped ``(N, T, C)``.
            person_mask: Valid-frame mask shaped ``(N, T)``.

        Returns:
            Clip-level actor tokens shaped ``(N, C)``.
        """
        if person_feats.numel() == 0:
            return torch.zeros(
                (0, person_feats.shape[-1]),
                device=person_feats.device,
                dtype=person_feats.dtype,
            )

        n, _, c = person_feats.shape
        valid_mask = person_mask > 0

        cls_token = self.cls_token.expand(n, -1, -1)
        temporal_tokens = torch.cat([cls_token, person_feats], dim=1)  # (N,1+T,C)

        cls_valid = torch.ones((n, 1), dtype=torch.bool, device=person_feats.device)
        attn_valid_mask = torch.cat([cls_valid, valid_mask], dim=1)
        key_padding_mask = ~attn_valid_mask

        attn_out, _ = self.temporal_attn(
            temporal_tokens,
            temporal_tokens,
            temporal_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        temporal_tokens = self.temporal_norm(temporal_tokens + attn_out)
        temporal_tokens = self.ffn(temporal_tokens)

        pooled = self.out_norm(temporal_tokens[:, 0, :])
        return pooled


# =========================================================
# 8) Person-Person Interaction within each clip
# =========================================================
class PersonRelationBlock(nn.Module):
    """
    第 2 阶段：同一 clip 内，球员之间、球员与 ball token 之间做 self-attention。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=valid
    输出：
      x:          (M,N,C)
    """

    def __init__(
        self, dim: int, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: float = 4.0
    ):
        """Initialize within-clip entity self-attention.

        Args:
            dim: Entity token dimension.
            num_heads: Number of attention heads.
            dropout: Attention and residual dropout probability.
            mlp_ratio: Feed-forward expansion ratio.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Exchange information among people and the ball within each frame."""
        if x.numel() == 0:
            return x

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~(valid_mask > 0)
            # 防止某一个 clip 内所有 token 都无效导致 attention NaN。
            all_invalid = key_padding_mask.all(dim=1)
            if all_invalid.any():
                key_padding_mask[all_invalid, 0] = False

        h = self.norm1(x)
        ctx, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(ctx))
        x = self.ffn(x)

        if valid_mask is not None:
            x = x * valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        return x


# =========================================================
# 9) Clip-Clip Interaction across clips for each player
# =========================================================
class ClipRelationBlock(nn.Module):
    """
    第 3 阶段：同一个球员/ball 在不同 clip 之间做 self-attention。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=该 clip 内该 token 有效
    输出：
      x:          (M,N,C)
    """

    def __init__(
        self, dim: int, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: float = 4.0
    ):
        """Initialize cross-clip self-attention for aligned entities.

        Args:
            dim: Entity token dimension.
            num_heads: Number of attention heads.
            dropout: Attention and residual dropout probability.
            mlp_ratio: Feed-forward expansion ratio.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Exchange information across clips for each aligned entity."""
        if x.numel() == 0:
            return x

        # (M,N,C) -> (N,M,C): 对每个球员沿 clip 维度交互
        x_in = x.permute(1, 0, 2).contiguous()

        key_padding_mask = None
        if valid_mask is not None:
            mask_in = valid_mask.permute(1, 0).contiguous()  # (N,M)
            key_padding_mask = ~(mask_in > 0)
            all_invalid = key_padding_mask.all(dim=1)
            if all_invalid.any():
                key_padding_mask[all_invalid, 0] = False
        else:
            mask_in = None

        h = self.norm1(x_in)
        ctx, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x_out = self.norm2(x_in + self.dropout(ctx))
        x_out = self.ffn(x_out)

        if mask_in is not None:
            x_out = x_out * mask_in.to(dtype=x_out.dtype).unsqueeze(-1)

        return x_out.permute(1, 0, 2).contiguous()


# =========================================================
# 10) Heads and pooling
# =========================================================
class PersonEventClassifierHead(nn.Module):
    """Map actor features to the event-class vocabulary."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: int = 512,
    ):
        """Initialize the normalized classification MLP.

        Args:
            in_dim: Actor feature dimension.
            num_classes: Number of event classes.
            dropout: Hidden dropout probability.
            hidden_dim: Hidden classifier dimension.
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, person_feats: torch.Tensor, person_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return event logits, zeroing actors masked as invalid."""
        person_feats = self.norm(person_feats)
        logits = self.mlp(person_feats)
        if person_mask is not None:
            person_mask = person_mask.to(dtype=logits.dtype).unsqueeze(-1)
            logits = logits * person_mask
        return logits


class GatedClipPooling(nn.Module):
    """
    对每个 person 在 M 个 clip 上做 gated pooling。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=valid
    输出：
      pooled:     (N,C)
      gate_logits:(M,N)
      gate_weights:(M,N,1)
    """

    def __init__(self, dim: int):
        """Initialize a learned clip-scoring gate.

        Args:
            dim: Clip-token feature dimension.
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        """Pool clips with a normalized learned score for each person."""
        gate_logits = self.gate(x).squeeze(-1)  # (M,N)

        if valid_mask is not None:
            mask = valid_mask > 0
            # 对无效 clip 置为极小值，避免参与 softmax。
            gate_logits = gate_logits.masked_fill(~mask, -1e4)
            # 若某 person 所有 clip 都无效，给第一个 clip 一个安全位置，避免 NaN。
            all_invalid = (~mask).all(dim=0)
            if all_invalid.any():
                gate_logits[:, all_invalid] = -1e4
                gate_logits[0, all_invalid] = 0.0

        gate_weights = torch.softmax(gate_logits, dim=0).unsqueeze(-1)  # (M,N,1)
        pooled = torch.sum(x * gate_weights, dim=0)  # (N,C)

        if return_weights:
            return pooled, gate_logits, gate_weights
        return pooled


class TopKClipPooling(nn.Module):
    """
    备用：对每个 person 在 clip 维度做 top-k pooling。
    注意：top-k 根据每个 clip 的 gate score 选 clip，再平均特征。
    """

    def __init__(self, dim: int):
        """Initialize learned scores used to select top-k clips.

        Args:
            dim: Clip-token feature dimension.
        """
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        topk: int = 2,
        return_weights: bool = False,
    ):
        """Average the highest-scoring valid clips for each person."""
        M, N, C = x.shape
        scores = self.score(x).squeeze(-1)  # (M,N)
        if valid_mask is not None:
            scores = scores.masked_fill(~(valid_mask > 0), -1e4)

        k = max(1, min(int(topk), M))
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=0)  # (k,N)

        gather_idx = topk_idx.unsqueeze(-1).expand(k, N, C)
        selected = torch.gather(x, dim=0, index=gather_idx)  # (k,N,C)
        pooled = selected.mean(dim=0)  # (N,C)

        if return_weights:
            # 构造一个稀疏 weights，方便和 gated pooling 输出接口统一。
            weights = torch.zeros((M, N, 1), device=x.device, dtype=x.dtype)
            weights.scatter_(0, topk_idx.unsqueeze(-1), 1.0 / float(k))
            return pooled, scores, weights
        return pooled
