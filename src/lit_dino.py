import pytorch_lightning as pl
import torch

from torchmetrics.segmentation import MeanIoU, DiceScore

from .loss import WeightedSumLoss


class LitDinoModule(pl.LightningModule):
    def __init__(
        self,
        backbone,
        encoder: str,
        loss: WeightedSumLoss,
        feature_info=None,  # Required for ConvNeXt (from timm's backbone.feature_info)
        freeze_backbone=True,
        lr=4e-3,
        weight_decay=5e-2,
        warmup_ratio=0.05
    ):
        assert encoder.lower() in ["convnext", "vit"], "Only convnext and vit encoders are supported"
        super().__init__()
        self.encoder = encoder.lower()

        if encoder == "convnext":
            from .convnext_fpn import DinoConvNeXtFPN
            if feature_info is None:
                raise ValueError("feature_info is required for ConvNeXt encoder (from timm's backbone.feature_info)")
            self.model = DinoConvNeXtFPN(backbone, feature_info)
        elif encoder == "vit":
            from .vit_dpt import DinoViTDPT
            self.model = DinoViTDPT(backbone)

        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        # Combined loss
        self.loss = loss

        # Metrics
        self.train_iou = MeanIoU(num_classes=2, include_background=False)
        self.val_iou = MeanIoU(num_classes=2, include_background=False)
        self.test_iou = MeanIoU(num_classes=2, include_background=False)
        self.test_dice = DiceScore(num_classes=2, include_background=False)

        # Freeze backbone if specified
        for p in self.model.encoder.backbone.parameters():
            p.requires_grad = not self.freeze_backbone

    def forward(self, x):
        return self.model(x)

    def _loss(self, logits, y):
        total, parts = self.loss(logits, y)
        # Log each component once per epoch
        for k, v in parts.items():
            self.log(f"loss/{k}", v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    def _shared_step(self, batch, tta=False):
        x, y = batch
        logits = self(x)
        if tta:  # Test Time Augmentation (horizontal flip) for inference
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
        head_params = list(self.model.decoder.parameters()) + \
                      list(self.model.seg_head.parameters())

        if any(p.requires_grad for p in self.model.encoder.backbone.parameters()):
            optimizer = torch.optim.AdamW([
                {"params": self.model.encoder.backbone.parameters(), "lr": self.lr * 0.1},
                {"params": head_params, "lr": self.lr},
            ], weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(head_params, lr=self.lr, weight_decay=self.weight_decay)

        # prefer explicit max_steps if set; otherwise use Lightning's estimate
        if getattr(self.trainer, "max_steps", None) and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            # Lightning 2.x provides this once dataloaders are connected
            total_steps = int(self.trainer.estimated_stepping_batches)

        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        cosine_steps = max(1, total_steps - warmup_steps)

        # build warmup (linear) then cosine, both stepped per batch
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
                "interval": "step",      # step every optimizer.step()
                "frequency": 1,
                "name": "cosine_warmup",
            },
        }
