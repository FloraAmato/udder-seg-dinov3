#!/usr/bin/env python3
"""
Ablation study runner for udder segmentation.

Ablation studies:
1. Augmentation ablation (Small pruned): on vs off
2. Loss ablation (Small pruned): BCE, Dice, BCE+Dice, BCE+Dice+Focal
3. Pruning ratio ablation (Small): r ∈ {0.1, 0.3, 0.5}
4. Pretraining ablation (Small): DINOv3 vs DINOv2 vs ImageNet
5. UNet baseline (ResNet-50)
"""

import os
import torch
import torch_pruning as tp
import pytorch_lightning as pl
import wandb

from src.callbacks import get_callbacks
from src.config import get_config
from src.dataloaders import get_dataloaders
from src.dinov3_backbone import get_convnext_features_backbone, get_vit_backbone
from src.lit_dino import LitDinoModule
from src.lit_unet import LitUNetModule
from src.logger import get_loggers, log_macs_params
from src.loss import get_segmentation_loss
from src.onnx_export import export_onnx

# Base config
cfg = get_config()
SEED = cfg["GENERAL"]["SEED"]
IN_CHANS = cfg["GENERAL"]["IN_CHANS"]
TRAIN_IMAGES = cfg["DATA"]["TRAIN_IMAGES"]
TRAIN_MASKS = cfg["DATA"]["TRAIN_MASKS"]
VAL_IMAGES = cfg["DATA"]["VAL_IMAGES"]
VAL_MASKS = cfg["DATA"]["VAL_MASKS"]
TEST_IMAGES = cfg["DATA"]["TEST_IMAGES"]
TEST_MASKS = cfg["DATA"]["TEST_MASKS"]
BATCH_SIZE = cfg["TRAINING"]["BATCH_SIZE"]
MAX_EPOCHS = cfg["TRAINING"]["MAX_EPOCHS"]
WEIGHT_DECAY = cfg["TRAINING"]["WEIGHT_DECAY"]
WARMUP_RATIO = cfg["TRAINING"]["WARMUP_RATIO"]
ROUND_TO = cfg["PRUNING"]["ROUND_TO"]
WANDB_OFFLINE = cfg['WANDB']['OFFLINE']
PROJECT_NAME = cfg['WANDB']['PROJECT_NAME']

pl.seed_everything(SEED)


def train_and_prune(
    model_name,
    encoder,
    lr,
    loss_config,
    pruning_ratio,
    use_augmentation=True,
    experiment_name=None,
    pretrained_source="dinov3",  # "dinov3", "dinov2", "imagenet"
):
    """Train and prune a model with given configuration."""
    exp_name = experiment_name or f"{model_name}_{pretrained_source}"
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  Model: {model_name}, Encoder: {encoder}, LR: {lr}")
    print(f"  Loss: {loss_config}, Pruning ratio: {pruning_ratio}")
    print(f"  Augmentation: {use_augmentation}, Pretrained: {pretrained_source}")
    print(f"{'='*60}\n")

    # Get loss
    loss = get_segmentation_loss(
        bce=loss_config.get("bce", 0.0),
        dice=loss_config.get("dice", 0.0),
        tversky=loss_config.get("tversky", 0.0),
        lovasz=loss_config.get("lovasz", 0.0),
        focal=loss_config.get("focal", 0.0),
        ignore_index=None
    )

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        TRAIN_IMAGES, TRAIN_MASKS,
        VAL_IMAGES, VAL_MASKS,
        TEST_IMAGES, TEST_MASKS,
        BATCH_SIZE,
        use_augmentation=use_augmentation
    )

    # Get backbone based on pretrained source
    if pretrained_source == "dinov3":
        timm_model_name = model_name
    elif pretrained_source == "dinov2":
        # DINOv2 models in timm (if available)
        if "convnext_small" in model_name:
            timm_model_name = "convnext_small.fb_in22k_ft_in1k"  # Fallback to IN22k
        else:
            timm_model_name = model_name.replace(".dinov3_lvd1689m", ".fb_in22k_ft_in1k")
    elif pretrained_source == "imagenet":
        # ImageNet pretrained
        if "convnext_small" in model_name:
            timm_model_name = "convnext_small.fb_in1k"
        else:
            timm_model_name = model_name.replace(".dinov3_lvd1689m", ".fb_in1k")
    else:
        timm_model_name = model_name

    # Setup loggers
    train_logger, prune_logger = get_loggers(
        model_name=exp_name,
        project_name=PROJECT_NAME + "_ablations",
        wandb_offline=WANDB_OFFLINE
    )

    # Create model
    if encoder.lower() == "convnext":
        backbone, feature_info = get_convnext_features_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
        lit_train = LitDinoModule(
            backbone, encoder=encoder, loss=loss, feature_info=feature_info,
            lr=lr, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, freeze_backbone=True
        )
    else:
        backbone = get_vit_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
        lit_train = LitDinoModule(
            backbone, encoder=encoder, loss=loss, feature_info=None,
            lr=lr, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO, freeze_backbone=True
        )

    # Train
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=get_callbacks(model_name=exp_name, max_epochs=MAX_EPOCHS, mode="train"),
        logger=train_logger,
    )
    trainer.fit(lit_train, train_loader, val_loader)

    example_inputs = torch.randn(1, IN_CHANS, 480, 640)
    log_macs_params(lit_train.model, example_inputs, train_logger)

    # Test before pruning
    ckpt_cb = trainer.checkpoint_callback
    best_ckpt = ckpt_cb.best_model_path

    if encoder.lower() == "convnext":
        backbone, feature_info = get_convnext_features_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
        lit_test = LitDinoModule.load_from_checkpoint(
            best_ckpt, encoder=encoder, backbone=backbone, feature_info=feature_info, loss=loss
        )
    else:
        backbone = get_vit_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
        lit_test = LitDinoModule.load_from_checkpoint(
            best_ckpt, encoder=encoder, backbone=backbone, feature_info=None, loss=loss
        )

    torch.save(lit_test.model.state_dict(), f"checkpoints/{exp_name.lower()}/best.pt")
    export_onnx(lit_test.model.cpu(), example_inputs, exp_name.lower())
    trainer.test(lit_test, test_loader)
    wandb.finish()

    # Pruning
    if pruning_ratio > 0:
        if encoder.lower() == "convnext":
            backbone, feature_info = get_convnext_features_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
            lit_prune = LitDinoModule.load_from_checkpoint(
                best_ckpt, encoder=encoder, backbone=backbone, feature_info=feature_info, loss=loss
            )
        else:
            backbone = get_vit_backbone(timm_model_name, in_chans=IN_CHANS, pretrained=True)
            lit_prune = LitDinoModule.load_from_checkpoint(
                best_ckpt, encoder=encoder, backbone=backbone, feature_info=None, loss=loss
            )

        model = lit_prune.model
        model.eval().cpu()

        for p in model.parameters():
            p.requires_grad = True

        imp = tp.importance.GroupMagnitudeImportance(p=2)
        ignored_layers = [model.seg_head]

        if encoder.lower() == "vit":
            for m in model.modules():
                if m.__class__.__name__ == "EvaAttention":
                    ignored_layers += [m.qkv, m.proj]

        pruner = tp.pruner.BasePruner(
            model, example_inputs, importance=imp,
            global_pruning=True, isomorphic=True,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            round_to=ROUND_TO,
        )
        pruner.step()
        log_macs_params(model, example_inputs, prune_logger)

        lit_prune.model = model

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=-1,
            precision=32,
            gradient_clip_val=1.0,
            callbacks=get_callbacks(model_name=exp_name, max_epochs=MAX_EPOCHS, mode="prune"),
            logger=prune_logger,
        )
        trainer.fit(lit_prune, train_loader, val_loader)

        pruned_ckpt = trainer.checkpoint_callback.best_model_path
        ckpt = torch.load(pruned_ckpt, map_location="cpu")
        lit_prune.load_state_dict(ckpt["state_dict"], strict=True)

        torch.save(lit_prune.model.state_dict(), f"checkpoints/{exp_name.lower()}_pruned/best.pt")
        export_onnx(lit_prune.model.cpu(), example_inputs, exp_name.lower() + "_pruned")
        trainer.test(lit_prune, test_loader)
        wandb.finish()


def train_unet_baseline():
    """Train UNet baseline with ResNet-50 encoder."""
    print(f"\n{'='*60}")
    print("Running: UNet ResNet-50 Baseline")
    print(f"{'='*60}\n")

    loss = get_segmentation_loss(
        bce=1.0, dice=1.0, tversky=0.0, lovasz=0.0, focal=1.0, ignore_index=None
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        TRAIN_IMAGES, TRAIN_MASKS,
        VAL_IMAGES, VAL_MASKS,
        TEST_IMAGES, TEST_MASKS,
        BATCH_SIZE,
        use_augmentation=True
    )

    train_logger, _ = get_loggers(
        model_name="unet_resnet50",
        project_name=PROJECT_NAME + "_ablations",
        wandb_offline=WANDB_OFFLINE
    )

    lit_model = LitUNetModule(
        loss=loss, in_channels=IN_CHANS,
        lr=1e-3, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=get_callbacks(model_name="unet_resnet50", max_epochs=MAX_EPOCHS, mode="train"),
        logger=train_logger,
    )
    trainer.fit(lit_model, train_loader, val_loader)

    example_inputs = torch.randn(1, IN_CHANS, 480, 640)
    log_macs_params(lit_model.model, example_inputs, train_logger)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    lit_test = LitUNetModule.load_from_checkpoint(best_ckpt, loss=loss, in_channels=IN_CHANS)

    torch.save(lit_test.model.state_dict(), "checkpoints/unet_resnet50/best.pt")
    export_onnx(lit_test.model.cpu(), example_inputs, "unet_resnet50")
    trainer.test(lit_test, test_loader)
    wandb.finish()


if __name__ == "__main__":
    # Base model for ablations
    BASE_MODEL = "convnext_small.dinov3_lvd1689m"
    BASE_ENCODER = "convnext"
    BASE_LR = 0.005
    BASE_LOSS = {"bce": 1.0, "dice": 1.0, "tversky": 0.0, "lovasz": 0.0, "focal": 1.0}
    BASE_PRUNING = 0.30

    print("\n" + "="*80)
    print("ABLATION STUDIES")
    print("="*80)

    # =========================================================================
    # 1. Augmentation ablation (Small pruned): on vs off
    # =========================================================================
    print("\n>>> ABLATION 1: Augmentation (on vs off)")

    train_and_prune(
        model_name=BASE_MODEL, encoder=BASE_ENCODER, lr=BASE_LR,
        loss_config=BASE_LOSS, pruning_ratio=BASE_PRUNING,
        use_augmentation=True,
        experiment_name="ablation_aug_on"
    )

    train_and_prune(
        model_name=BASE_MODEL, encoder=BASE_ENCODER, lr=BASE_LR,
        loss_config=BASE_LOSS, pruning_ratio=BASE_PRUNING,
        use_augmentation=False,
        experiment_name="ablation_aug_off"
    )

    # =========================================================================
    # 2. Loss ablation (Small pruned): BCE, Dice, BCE+Dice, BCE+Dice+Focal
    # =========================================================================
    print("\n>>> ABLATION 2: Loss functions")

    loss_configs = {
        "loss_bce": {"bce": 1.0, "dice": 0.0, "tversky": 0.0, "lovasz": 0.0, "focal": 0.0},
        "loss_dice": {"bce": 0.0, "dice": 1.0, "tversky": 0.0, "lovasz": 0.0, "focal": 0.0},
        "loss_bce_dice": {"bce": 1.0, "dice": 1.0, "tversky": 0.0, "lovasz": 0.0, "focal": 0.0},
        "loss_bce_dice_focal": {"bce": 1.0, "dice": 1.0, "tversky": 0.0, "lovasz": 0.0, "focal": 1.0},
    }

    for exp_name, loss_cfg in loss_configs.items():
        train_and_prune(
            model_name=BASE_MODEL, encoder=BASE_ENCODER, lr=BASE_LR,
            loss_config=loss_cfg, pruning_ratio=BASE_PRUNING,
            use_augmentation=True,
            experiment_name=f"ablation_{exp_name}"
        )

    # =========================================================================
    # 3. Pruning ratio ablation (Small): r ∈ {0.1, 0.3, 0.5}
    # =========================================================================
    print("\n>>> ABLATION 3: Pruning ratio")

    for prune_ratio in [0.1, 0.3, 0.5]:
        train_and_prune(
            model_name=BASE_MODEL, encoder=BASE_ENCODER, lr=BASE_LR,
            loss_config=BASE_LOSS, pruning_ratio=prune_ratio,
            use_augmentation=True,
            experiment_name=f"ablation_prune_{int(prune_ratio*100)}"
        )

    # =========================================================================
    # 4. Pretraining ablation (Small): DINOv3 vs DINOv2 vs ImageNet
    # =========================================================================
    print("\n>>> ABLATION 4: Pretraining source")

    for pretrained in ["dinov3", "imagenet"]:  # DINOv2 requires specific model names
        train_and_prune(
            model_name=BASE_MODEL, encoder=BASE_ENCODER, lr=BASE_LR,
            loss_config=BASE_LOSS, pruning_ratio=BASE_PRUNING,
            use_augmentation=True,
            experiment_name=f"ablation_pretrain_{pretrained}",
            pretrained_source=pretrained
        )

    # =========================================================================
    # 5. UNet baseline (ResNet-50)
    # =========================================================================
    print("\n>>> BASELINE: UNet ResNet-50")
    train_unet_baseline()

    print("\n" + "="*80)
    print("ALL ABLATION STUDIES COMPLETED!")
    print("="*80)
