from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def get_callbacks(model_name: str, max_epochs: int, mode="train"):
    assert mode in ["train", "prune"]
    if mode == "train":
        ckpt_path = f"checkpoints/{model_name.lower()}"
    elif mode == "prune":
        ckpt_path = f"checkpoints/{model_name.lower()}_pruned"

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="{epoch:02d}-{val_iou:.4f}",
        save_top_k=1,
        monitor="val_iou",
        mode="max",
    )
    lr_cb = LearningRateMonitor(
        logging_interval="epoch"
    )
    return [ckpt_cb, lr_cb]
