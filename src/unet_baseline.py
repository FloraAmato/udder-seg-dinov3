import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetBaseline(nn.Module):
    """
    UNet baseline with ResNet-50 encoder for comparison.
    Uses segmentation_models_pytorch for implementation.
    """
    def __init__(self, in_channels=1, classes=1, encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)


def get_unet_baseline(in_channels=1):
    """Create a UNet baseline model with ResNet-50 encoder."""
    return UNetBaseline(in_channels=in_channels)
