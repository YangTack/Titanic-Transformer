import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import numpy as np
import torch
import torchmetrics as tm
from model import Classifier

class PLModel(pl.LightningModule):

    def __init__(self, lr=0.01, input_dim=9):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Classifier(input_dim=input_dim)
        self.loss = torch.nn.BCELoss()
        self.train_loss_metric = tm.MeanMetric()
        self.val_loss_metric = tm.MeanMetric()
        self.train_metrics = tm.MetricCollection([
            tm.Accuracy("binary"),
            tm.Recall("binary"),
            tm.Precision("binary"),
            tm.F1Score("binary"),
        ], prefix="train")
        self.val_metrics = tm.MetricCollection([
            tm.Accuracy("binary"),
            tm.Recall("binary"),
            tm.Precision("binary"),
            tm.F1Score("binary"),
            tm.AveragePrecision("binary"),
            tm.AUROC("binary"),
        ], prefix="val")

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> STEP_OUTPUT:
        x, target = batch
        result =  self.model.forward(x)
        loss = self.loss(result, target.reshape(-1, 1).to(result))
        return {
            "loss": loss,
            "result": result,
            "target": target.reshape(-1, 1).to(result)
        }


    def on_train_batch_end(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss, result, target = outputs["loss"], outputs["result"], outputs["target"]
        self.train_loss_metric.update(loss)
        self.train_metrics.update(result, target.reshape(-1, 1))

    def on_train_epoch_end(self) -> None:
        log_dict = {}
        log_dict.update(self.train_metrics.compute())
        log_dict.update({"train_loss": self.train_loss_metric.compute()})
        self.train_metrics.reset()
        self.train_loss_metric.reset()
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        x, target = batch
        result = self.model.forward(x)
        loss = self.loss(result, target.reshape(-1, 1).to(result))
        return {
            "loss": loss,
            "result": result,
            "target": target.reshape(-1, 1).to(result)
        }

    def on_validation_batch_end(self, outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...], batch_idx: int) -> None:
        loss, result, target = outputs["loss"], outputs["result"], outputs["target"]
        self.val_loss_metric.update(loss)
        self.val_metrics.update(result, target.reshape(-1, 1).int())
    
    def on_validation_epoch_end(self) -> None:
        log_dict = {}
        log_dict.update(self.val_metrics.compute())
        log_dict.update({"val_loss": self.val_loss_metric.compute()})
        self.val_metrics.reset()
        self.val_loss_metric.reset()
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)
        return {
            "optimizer": optim,
            "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim,
                T_0=100,
                T_mult=2,
                eta_min=1e-6,
                last_epoch=-1,
            )
        }
