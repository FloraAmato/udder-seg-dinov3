import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class LightweightSegModel(nn.Module):
    """
    Lightweight segmentation model using various efficient encoders.
    Uses segmentation_models_pytorch with FPN decoder for fair comparison.
    """
    def __init__(self, encoder_name: str, in_channels: int = 1, classes: int = 1, encoder_weights: str = "imagenet"):
        super().__init__()
        self.encoder_name = encoder_name
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x):
        return self.model(x)


def get_mobilenetv3_baseline(in_channels: int = 1):
    """MobileNetV3-Large with FPN decoder."""
    return LightweightSegModel(
        encoder_name="timm-mobilenetv3_large_100",
        in_channels=in_channels
    )


def get_efficientvit_baseline(in_channels: int = 1):
    """EfficientViT-M2 with FPN decoder."""
    return LightweightSegModel(
        encoder_name="timm-efficientvit_m2",
        in_channels=in_channels
    )


def get_mobilenetv2_baseline(in_channels: int = 1):
    """MobileNetV2 with FPN decoder."""
    return LightweightSegModel(
        encoder_name="mobilenet_v2",
        in_channels=in_channels
    )
