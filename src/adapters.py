"""
Input adapters for image preprocessing.

NOTE: When using timm for DINOv3 models, these adapters are NOT needed:
- Input channels: timm handles this natively via the `in_chans` parameter.
  Simply pass in_chans=1 when calling timm.create_model() for grayscale images.
- Normalization: timm models include their own normalization based on the
  pretrained weights configuration.

These adapters are kept for legacy compatibility and potential custom use cases.
"""

import torch
import torch.nn as nn


class InputAdapter1to3(nn.Module):
    """
    Convert grayscale (1-channel) to RGB (3-channel) via learned projection.

    NOTE: This adapter is NOT needed when using timm models. timm handles
    input channel adaptation natively via the `in_chans` parameter.

    Example with timm (recommended):
        model = timm.create_model('convnext_base.dinov3_lvd1689m', in_chans=1)
        # No adapter needed - timm adapts the first conv layer automatically
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight[:] = 1/3

    def forward(self, x1):
        return self.proj(x1)


class ImageNetNormalizer(nn.Module):
    """
    ImageNet normalization (required for ImageNet-pretrained models).

    NOTE: This normalizer may NOT be needed when using timm models, as timm
    applies normalization internally based on the model's pretrained config.
    Check your specific timm model's requirements.
    """
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, x3):
        return (x3 - self.mean) / self.std
