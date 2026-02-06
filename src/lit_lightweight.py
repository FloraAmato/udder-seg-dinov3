import pytorch_lightning as pl
import torch

from torchmetrics.segmentation import MeanIoU, DiceScore

from .loss import WeightedSumLoss
from .lightweight_baselines import (
    get_mobilenetv3_baseline,
    get_efficientvit_baseline,
    get_mobilenetv2_baseline,
)


class LitLightweightModule(pl.LightningModule):
    """Lightning module for lightweight segmentation baselines."""

    def __init__(
        self,
        encoder_type: str,
        loss: WeightedSumLoss,
        in_channels: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 5e-2,
        warmup_ratio: float = 0.05
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "mobilenetv3":
            self.model = get_mobilenetv3_baseline(in_channels=in_channels)
        elif encoder_type == "efficientvit":
            self.model = get_efficientvit_baseline(in_channels=in_channels)
        elif encoder_type == "mobilenetv2":
            self.model = get_mobilenetv2_baseline(in_channels=in_channels)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.loss = loss

        # Metrics
        self.train_iou = MeanIoU(num_classes=2, include_background=False)
        self.val_iou = MeanIoU(num_classes=2, include_background=False)
        self.test_iou = MeanIoU(num_classes=2, include_background=False)
        self.test_dice = DiceScore(num_classes=2, include_background=False)

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y):
        total, parts = self.loss(logits, y)
        for k, v in parts.items():
            self.log(f"loss/{k}", v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    def _shared_step(self, batch, tta=False):
        x, y = batch
        logits = self(x)
        if tta:
            x_flip = torch.flip(x, dims=[-1])
            logits = (logits + torch.flip(self(x_flip), dims=[-1])) / 2.0
        loss = self._loss(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).int()
        return loss, preds.int(), y.int()

    def training_step(self, batch, _):
        loss, preds, target = self._shared_step(batch, tta=False)
        self.train_iou(preds, target)
        self.log_dict(
            {"train_loss": loss, "train_iou": self.train_iou},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, _):
        loss, preds, target = self._shared_step(batch, tta=False)
        self.val_iou(preds, target)
        self.log_dict(
            {"val_loss": loss, "val_iou": self.val_iou},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def test_step(self, batch, _):
        _, preds, target = self._shared_step(batch, tta=True)
        self.test_iou(preds, target)
        self.test_dice(preds, target)
        self.log_dict(
            {"test_iou": self.test_iou, "test_dice": self.test_dice},
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if getattr(self.trainer, "max_steps", None) and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            total_steps = int(self.trainer.estimated_stepping_batches)

        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=self.lr * 1e-3
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_warmup",
            },
        }
