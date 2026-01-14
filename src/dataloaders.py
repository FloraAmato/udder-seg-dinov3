from torch.utils.data import DataLoader

from .augmentations import get_training_augmentation, get_validation_augmentation
from .dataset import UdderSegmentationDataset


def get_dataloaders(train_images, train_masks, val_images, val_masks, test_images, test_masks, batch_size, use_augmentation=True):
    train_ds = UdderSegmentationDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        augmentation=get_training_augmentation(use_augmentation=use_augmentation),
    )
    val_ds = UdderSegmentationDataset(
        images_dir=val_images,
        masks_dir=val_masks,
        augmentation=get_validation_augmentation(),
    )
    test_ds = UdderSegmentationDataset(
        images_dir=test_images,
        masks_dir=test_masks,
        augmentation=get_validation_augmentation(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
