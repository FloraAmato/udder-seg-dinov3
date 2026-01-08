import torch
import torch.nn as nn
from typing import Sequence, List, Optional

from .dpt_decoder import DPTDecoder, DPTSegmentationHead


class TimmViTEncoder(nn.Module):
    """
    Wrapper for timm ViT model using forward_intermediates() for feature extraction.

    timm handles input channels natively via the in_chans parameter,
    so no external input adapter is needed.
    """
    def __init__(self, dino_vit, out_indices: Sequence[int] = (-1, -3, -6, -9)):
        """
        Args:
            dino_vit: timm ViT model created with timm.create_model()
            out_indices: Which transformer blocks to extract features from.
                        Supports both positive and negative indexing.
                        Default: last 4 blocks at indices -1, -3, -6, -9
        """
        super().__init__()
        self.backbone = dino_vit

        # Infer patch size & embed dim from the model
        if hasattr(self.backbone, 'patch_embed'):
            proj = self.backbone.patch_embed.proj
            self.patch_size = int(getattr(proj, "kernel_size", (16, 16))[0])
            self.embed_dim = int(proj.out_channels)
        else:
            # Fallback for different model structures
            self.patch_size = 16
            self.embed_dim = self.backbone.embed_dim

        # Get total number of blocks
        self.total_blocks = len(self.backbone.blocks)

        # Resolve indices: convert negatives to absolute 0..L-1 and validate
        self.out_indices = self._resolve_indices(out_indices)

    def _resolve_indices(self, indices: Sequence[int]) -> List[int]:
        """Resolve negative indices to positive ones and validate."""
        resolved = []
        for i in indices:
            j = i if i >= 0 else self.total_blocks + i
            if not (0 <= j < self.total_blocks):
                raise ValueError(
                    f"Layer index {i} (→ {j}) out of range [0, {self.total_blocks-1}]"
                )
            resolved.append(int(j))
        # Keep them in ascending depth (shallower → deeper)
        resolved = sorted(dict.fromkeys(resolved))
        return resolved

    def forward(self, x: torch.Tensor):
        """
        Forward pass using timm's forward_intermediates().

        Args:
            x: Input tensor of shape (B, C, H, W) where C matches in_chans

        Returns:
            feats: List of feature tensors from specified transformer blocks
            prefixes: List of None (no CLS tokens returned)
        """
        # Use timm's forward_intermediates() to get intermediate features
        # This returns features in [B, N, C] format, we need to reshape to [B, C, H, W]
        _, intermediates = self.backbone.forward_intermediates(
            x,
            indices=self.out_indices,
            norm=True,
        )

        # Reshape intermediates to spatial format [B, C, H, W]
        B = x.shape[0]
        H, W = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        feats = []
        for feat in intermediates:
            # feat shape: [B, N, C] where N = H*W (+ maybe cls token)
            if feat.shape[1] == H * W + 1:
                # Has CLS token, remove it
                feat = feat[:, 1:, :]
            elif feat.shape[1] == H * W:
                pass  # No CLS token
            else:
                # Handle other cases (e.g., register tokens)
                # Assume extra tokens are at the beginning
                expected_patches = H * W
                feat = feat[:, -expected_patches:, :]

            # Reshape to [B, C, H, W]
            feat = feat.transpose(1, 2).reshape(B, -1, H, W)
            feats.append(feat)

        prefixes: List[Optional[torch.Tensor]] = [None] * len(feats)
        return feats, prefixes


class DinoViTDPT(nn.Module):
    """
    DINOv3 ViT encoder with DPT decoder for segmentation.

    Uses timm's ViT with forward_intermediates() for multi-block feature extraction.
    Input channel handling is done natively by timm via the in_chans parameter.
    """
    def __init__(
        self,
        dino_vit,
        out_indices: Sequence[int] = (-1, -3, -6, -9),
        num_classes: int = 1,
        readout: str = "ignore",   # no prefix tokens available
        fusion_channels: int = 256,
    ):
        super().__init__()
        self.encoder = TimmViTEncoder(dino_vit, out_indices=out_indices)

        enc_out_ch = (self.encoder.embed_dim,) * 4
        enc_strides = (self.encoder.patch_size,) * 4

        self.decoder = DPTDecoder(
            encoder_out_channels=enc_out_ch,
            encoder_output_strides=enc_strides,
            encoder_has_prefix_tokens=False,  # No CLS tokens in our feature extraction
            readout=readout,
            fusion_channels=fusion_channels,
        )
        self.seg_head = DPTSegmentationHead(
            in_channels=fusion_channels,
            out_channels=num_classes,
            activation=None,
            upsampling=2.0,  # decoder gives H/2 → upsample to H
        )

    @property
    def patch_size(self):
        return self.encoder.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, prefixes = self.encoder(x)
        fused = self.decoder(feats, prefixes)  # [B, C, H/2, W/2]
        return self.seg_head(fused)            # [B, K, H, W]
