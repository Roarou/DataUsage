import pytorch_lightning as pl
from torch import nn
from Model.pointnet import SpineSegmentationNet
import torch
from Model.get_metrics import calculate_metrics

class SpineSegmentationModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = SpineSegmentationNet()
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        metrics = calculate_metrics(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_metrics', metrics, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
