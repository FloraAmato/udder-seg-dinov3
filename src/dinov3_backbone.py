import timm


def get_backbone(model_name: str, in_chans: int = 1, pretrained: bool = True):
    """
    Load a DINOv3 backbone from timm with automatic weight download.

    timm handles input channels (in_chans) natively - when in_chans != 3,
    it adapts the first convolutional layer automatically.

    Args:
        model_name: timm model name (e.g., 'convnext_base.dinov3_lvd1689m',
                    'vit_base_patch16_dinov3.lvd1689m')
        in_chans: Number of input channels (default: 1 for grayscale thermal images).
                  timm automatically adapts the model's input layer.
        pretrained: Whether to load pretrained weights (default: True)

    Returns:
        backbone: The timm model configured as a feature extractor

    Available DINOv3 models in timm:
        ConvNeXt variants:
            - convnext_tiny.dinov3_lvd1689m
            - convnext_small.dinov3_lvd1689m
            - convnext_base.dinov3_lvd1689m

        ViT variants:
            - vit_small_patch16_dinov3.lvd1689m
            - vit_base_patch16_dinov3.lvd1689m

    Note:
        Use `timm.list_models('*dinov3*')` to see all available DINOv3 models.
    """
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    return backbone


def get_convnext_features_backbone(model_name: str, in_chans: int = 1, pretrained: bool = True):
    """
    Load a DINOv3 ConvNeXt backbone configured for multi-scale feature extraction.

    Uses timm's features_only=True to return feature maps from each stage.

    Args:
        model_name: timm ConvNeXt model name (e.g., 'convnext_base.dinov3_lvd1689m')
        in_chans: Number of input channels (default: 1 for grayscale thermal images)
        pretrained: Whether to load pretrained weights (default: True)

    Returns:
        backbone: The timm model configured for multi-scale feature extraction
        feature_info: Information about output feature channels at each stage
    """
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        features_only=True,
        out_indices=(0, 1, 2, 3),  # Return features from all 4 stages
    )
    return backbone, backbone.feature_info


def get_vit_backbone(model_name: str, in_chans: int = 1, pretrained: bool = True):
    """
    Load a DINOv3 ViT backbone for feature extraction.

    For ViT models, use forward_intermediates() to get intermediate layer features.

    Args:
        model_name: timm ViT model name (e.g., 'vit_base_patch16_dinov3.lvd1689m')
        in_chans: Number of input channels (default: 1 for grayscale thermal images)
        pretrained: Whether to load pretrained weights (default: True)

    Returns:
        backbone: The timm ViT model
    """
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
    )
    return backbone
