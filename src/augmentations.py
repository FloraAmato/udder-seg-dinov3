import cv2
import albumentations as A

def get_training_augmentation(use_augmentation=True):
    """
    Data augmentation pipeline for training.

    Args:
        use_augmentation: If False, only apply minimal transforms (pad + normalize)
    """
    if not use_augmentation:
        # No augmentation - only pad and normalize
        train_transform = [
            A.PadIfNeeded(
                min_height=480,
                min_width=640,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=1.0
            ),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ]
        return A.Compose(train_transform, additional_targets={"mask": "mask"})

    train_transform = [
        # 1) RandomResizedCrop directly to 480×640, but vary scale and aspect-ratio
        A.RandomResizedCrop(
            size=(480, 640),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),

        # 2) Random horizontal flip
        A.HorizontalFlip(p=0.5),

        # 3) Random brightness/contrast adjustment
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),

        # 4) Blur/sharpen: apply one of GaussianBlur, MotionBlur, MedianBlur, or Sharpen
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.Sharpen(p=1.0)
        ], p=0.5),

        # 5) Add Gaussian noise
        A.GaussNoise(std_range=(0.1, 0.5), p=0.5),

        # 6) Final pad to ensure exactly 480×640 (just in case)
        A.PadIfNeeded(
            min_height=480,
            min_width=640,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1.0
        ),

        # 7) Normalize to [0, 1] range
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ]

    return A.Compose(train_transform, additional_targets={"mask": "mask"})


def get_validation_augmentation():
    """
    Data augmentation pipeline for validation (just padding and normalization).
    """
    test_transform = [
        A.PadIfNeeded(
            min_height=480,
            min_width=640,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1.0
        ),

        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
    ]
    return A.Compose(test_transform, additional_targets={"mask": "mask"})
