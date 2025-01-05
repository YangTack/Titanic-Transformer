from pathlib import Path

import torch
from dataset import TitanicDataset, MU, STD
from pl_model import PLModel
import lightning as L
import lightning.pytorch as pl
import lightning.pytorch.tuner.tuning as pl_tuning
import lightning.pytorch.callbacks as pl_callbacks

def main() -> None:
    L.seed_everything(42)
    dataset = TitanicDataset(path=Path("data"))
    val_dataset = TitanicDataset(path=Path("data"), scope="VAL")
    model = PLModel(lr=1e-5)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=64,
        num_workers=4,
        persistent_workers=True,
    )
    filename = "{epoch}-{valBinaryF1Score:.4f}-{valBinaryRecall:.4f}-{valBinaryPrecision:.4f}-{valBinaryAveragePrecision:.4f}-{valBinaryAUROC:.4f}"
    trainer = L.Trainer(
        max_epochs=1000, 
        callbacks=[
            pl_callbacks.RichModelSummary(),
            pl_callbacks.RichProgressBar(),
            pl_callbacks.LearningRateMonitor(),
            # pl_callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
            pl_callbacks.ModelCheckpoint("./ckpt/best_f1", monitor="valBinaryF1Score", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_recall", monitor="valBinaryRecall", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_precision", monitor="valBinaryPrecision", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_pr", monitor="valBinaryAveragePrecision", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_auroc", monitor="valBinaryAUROC", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_f1_tr", monitor="trainBinaryF1Score", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_recall_tr", monitor="trainBinaryRecall", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_precision_tr", monitor="trainBinaryPrecision", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/last", monitor="epoch", save_top_k=3, mode="max", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_loss", monitor="val_loss", save_top_k=3, mode="min", filename=filename),
            pl_callbacks.ModelCheckpoint("./ckpt/best_loss_tr", monitor="train_loss", save_top_k=3, mode="min", filename=filename),
        ],
        gradient_clip_val=0.1,
        log_every_n_steps=5,
    )
    # ckpt = "ckpt/best_auroc/epoch=189-valBinaryF1Score=0.7791-valBinaryRecall=0.6382-valBinaryPrecision=1.0000-valBinaryAveragePrecision=0.9831-valBinaryAUROC=0.9863.ckpt"
    ckpt = None
    # tuner = pl_tuning.Tuner(trainer)
    # tuner.lr_find(model, train_dataloader, num_training=1000, early_stop_threshold=10)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)

if __name__ == "__main__":
    main()