import torch
import torch_pruning as tp
import pytorch_lightning as pl
import wandb

from src.callbacks import get_callbacks
from src.config import get_config
from src.dataloaders import get_dataloaders
from src.dinov3_backbone import get_convnext_features_backbone, get_vit_backbone
from src.lit_dino import LitDinoModule
from src.logger import get_loggers, log_macs_params
from src.loss import get_segmentation_loss
from src.onnx_export import export_onnx


# Get config
cfg = get_config()
SEED = cfg["GENERAL"]["SEED"]
IN_CHANS = cfg["GENERAL"]["IN_CHANS"]
MODEL_ENCODER = cfg["MODEL"]["MODEL_ENCODER"]
MODEL_NAME = cfg["MODEL"]["MODEL_NAME"]
TRAIN_IMAGES = cfg["DATA"]["TRAIN_IMAGES"]
TRAIN_MASKS = cfg["DATA"]["TRAIN_MASKS"]
VAL_IMAGES = cfg["DATA"]["VAL_IMAGES"]
VAL_MASKS = cfg["DATA"]["VAL_MASKS"]
TEST_IMAGES = cfg["DATA"]["TEST_IMAGES"]
TEST_MASKS = cfg["DATA"]["TEST_MASKS"]
BATCH_SIZE = cfg["TRAINING"]["BATCH_SIZE"]
MAX_EPOCHS = cfg["TRAINING"]["MAX_EPOCHS"]
LR = cfg["TRAINING"]["LR"]
WEIGHT_DECAY = cfg["TRAINING"]["WEIGHT_DECAY"]
WARMPUP_RATIO = cfg["TRAINING"]["WARMUP_RATIO"]
LOSS_BCE = cfg["LOSS"]["BCE"]
LOSS_DICE = cfg["LOSS"]["DICE"]
LOSS_TVERSKY = cfg["LOSS"]["TVERSKY"]
LOSS_LOVASZ = cfg["LOSS"]["LOVASZ"]
LOSS_FOCAL = cfg["LOSS"]["FOCAL"]
LOSS_IGNORE_INDEX = cfg["LOSS"]["IGNORE_INDEX"]
PRUNING_RATIO = cfg["PRUNING"]["PRUNING_RATIO"]
ROUND_TO = cfg["PRUNING"]["ROUND_TO"]
WANDB_OFFLINE = cfg['WANDB']['OFFLINE']
PROJECT_NAME = cfg['WANDB']['PROJECT_NAME']

# Seed everything rng related
pl.seed_everything(SEED)

train_logger, prune_logger = get_loggers(
    model_name=MODEL_NAME,
    project_name=PROJECT_NAME,
    wandb_offline=WANDB_OFFLINE
)

trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision=32,
    gradient_clip_val=1.0,
    callbacks=get_callbacks(model_name=MODEL_NAME, max_epochs=MAX_EPOCHS, mode="train"),
    logger=train_logger,
)

train_loader, val_loader, test_loader = get_dataloaders(
    TRAIN_IMAGES, TRAIN_MASKS,
    VAL_IMAGES, VAL_MASKS,
    TEST_IMAGES, TEST_MASKS,
    BATCH_SIZE
)

# Load backbone from timm (weights are automatically downloaded)
# timm handles input channels natively via the in_chans parameter
loss = get_segmentation_loss(LOSS_BCE, LOSS_DICE, LOSS_TVERSKY, LOSS_LOVASZ, LOSS_FOCAL, ignore_index=LOSS_IGNORE_INDEX)

if MODEL_ENCODER.lower() == "convnext":
    # ConvNeXt: use features_only mode for multi-scale feature extraction
    backbone, feature_info = get_convnext_features_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    lit_train = LitDinoModule(
        backbone, encoder=MODEL_ENCODER, loss=loss, feature_info=feature_info,
        lr=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMPUP_RATIO, freeze_backbone=True
    )
elif MODEL_ENCODER.lower() == "vit":
    # ViT: use standard model with forward_intermediates() for feature extraction
    backbone = get_vit_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    feature_info = None
    lit_train = LitDinoModule(
        backbone, encoder=MODEL_ENCODER, loss=loss, feature_info=feature_info,
        lr=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMPUP_RATIO, freeze_backbone=True
    )
else:
    raise ValueError(f"Unknown encoder: {MODEL_ENCODER}. Use 'convnext' or 'vit'.")

# Training loop
trainer.fit(lit_train, train_loader, val_loader)
example_inputs = torch.randn(1, IN_CHANS, 480, 640)
log_macs_params(lit_train.model, example_inputs, train_logger)

# Testing loop
ckpt_cb = trainer.checkpoint_callback
best_ckpt = ckpt_cb.best_model_path

# Reload model for testing
if MODEL_ENCODER.lower() == "convnext":
    backbone, feature_info = get_convnext_features_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    lit_test = LitDinoModule.load_from_checkpoint(
        best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, feature_info=feature_info, loss=loss
    )
else:
    backbone = get_vit_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    lit_test = LitDinoModule.load_from_checkpoint(
        best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, feature_info=None, loss=loss
    )

torch.save(lit_test.model.state_dict(), f"checkpoints/{MODEL_NAME.lower()}/best.pt")
onnx_model = lit_test.model.cpu()
export_onnx(onnx_model, example_inputs, MODEL_NAME.lower())
trainer.test(lit_test, test_loader)
wandb.finish()

# Pruning
if MODEL_ENCODER.lower() == "convnext":
    backbone, feature_info = get_convnext_features_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    lit_prune = LitDinoModule.load_from_checkpoint(
        best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, feature_info=feature_info, loss=loss
    )
else:
    backbone = get_vit_backbone(MODEL_NAME, in_chans=IN_CHANS, pretrained=True)
    lit_prune = LitDinoModule.load_from_checkpoint(
        best_ckpt, encoder=MODEL_ENCODER, backbone=backbone, feature_info=None, loss=loss
    )

model = lit_prune.model
model.eval().cpu()

# IMPORTANT: Temporarily enable gradients on ALL parameters before pruning.
# torch_pruning needs a traceable dependency graph to prune the encoder.
# If encoder params have requires_grad=False, TP treats it as opaque and skips it.
for p in model.parameters():
    p.requires_grad = True

imp = tp.importance.GroupMagnitudeImportance(p=2)  # L2 over grouped weights
ignored_layers = [
    model.seg_head,
]

# Pruner (global + isomorphic) with configurable pruning ratio and channel rounding
pruner = tp.pruner.BasePruner(
    model,
    example_inputs,
    importance=imp,
    global_pruning=True,
    isomorphic=True,
    pruning_ratio=PRUNING_RATIO,
    ignored_layers=ignored_layers,
    round_to=ROUND_TO,
)

pruner.step()  # Prune the graph
log_macs_params(model, example_inputs, prune_logger)

# Put the pruned module back (gradients already enabled for fine-tuning)
lit_prune.model = model

# Recreate the optimizer (shapes changed!)
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=-1,
    precision=32,
    gradient_clip_val=1.0,
    strategy="ddp_find_unused_parameters_true",
    callbacks=get_callbacks(model_name=MODEL_NAME, max_epochs=MAX_EPOCHS, mode="prune"),
    logger=prune_logger,
)
trainer.fit(lit_prune, train_loader, val_loader)
pruned_ckpt_cb = trainer.checkpoint_callback
pruned_best_ckpt = pruned_ckpt_cb.best_model_path

# Load the state_dict from the checkpoint into the already-pruned LightningModule
ckpt = torch.load(pruned_best_ckpt, map_location="cpu")
lit_prune.load_state_dict(ckpt["state_dict"], strict=True)

# Save/export weights and test
torch.save(lit_prune.model.state_dict(), f"checkpoints/{MODEL_NAME.lower()}_pruned/best.pt")
onnx_model = lit_prune.model.cpu()
export_onnx(onnx_model, example_inputs, MODEL_NAME.lower() + "_pruned")
trainer.test(lit_prune, test_loader)
wandb.finish()
