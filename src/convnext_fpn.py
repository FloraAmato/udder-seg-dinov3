import torch.nn as nn
import torch.nn.functional as F

from .fpn_decoder import FPNDecoder


class TimmConvNeXtFeatures(nn.Module):
    """
    Wrapper for timm ConvNeXt model with features_only=True.

    timm handles input channels natively via the in_chans parameter,
    so no external input adapter is needed.
    """
    def __init__(self, backbone, feature_info):
        """
        Args:
            backbone: timm ConvNeXt model created with features_only=True
            feature_info: backbone.feature_info from timm containing channel info
        """
        super().__init__()
        self.backbone = backbone
        self.feature_info = feature_info

    def forward(self, x):
        """
        Forward pass returning multi-scale feature maps.

        Args:
            x: Input tensor of shape (B, C, H, W) where C matches the in_chans
               used when creating the backbone (e.g., C=1 for grayscale)

        Returns:
            List of feature maps [c2, c3, c4, c5] at strides [4, 8, 16, 32]
        """
        features = self.backbone(x)
        return features  # List of [c2, c3, c4, c5]


class DinoConvNeXtFPN(nn.Module):
    """
    DINOv3 ConvNeXt encoder with FPN decoder for segmentation.

    Uses timm's ConvNeXt with features_only=True for multi-scale feature extraction.
    Input channel handling is done natively by timm via the in_chans parameter.
    """
    def __init__(
        self,
        convnext_backbone,          # timm ConvNeXt with features_only=True
        feature_info,               # backbone.feature_info from timm
        pyramid_channels=256,
        segmentation_channels=128,
        merge_policy="cat",         # "add" or "cat"
        dropout=0.1
    ):
        super().__init__()
        self.encoder = TimmConvNeXtFeatures(convnext_backbone, feature_info)

        # Get encoder channel dimensions from timm's feature_info
        enc_ch = [info['num_chs'] for info in feature_info]

        self.decoder = FPNDecoder(
            encoder_channels=enc_ch,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            dropout=dropout,
            merge_policy=merge_policy,
            interpolation_mode="nearest",
        )
        self.seg_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)                                                         # [c2, c3, c4, c5]
        x = self.decoder(feats)                                                         # FPN features at 1/4 resolution
        x = self.seg_head(x)                                                            # Masks as (B, 1, H/4, W/4) tensor
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)    # Rescaling to input size
        return x
