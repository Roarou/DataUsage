import pytorch_lightning as pl
from torch import nn
from Model.pointnet import SpineSegmentationNet
import torch.optim as optim
from Model.get_metrics import calculate_metrics


class PointCloudModule(pl.LightningModule):
    def __init__(self, data_path, batch_size=1, learning_rate=0.001):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = SpineSegmentationNet()
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        binary_predictions = (outputs >= 0.5).float()
        loss = self.criterion(binary_predictions, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        binary_predictions = (outputs >= 0.5).float()
        loss = self.criterion(binary_predictions, labels)
        metrics = calculate_metrics(binary_predictions, labels)
        self.log_dict({'val_loss': loss, 'val_accuracy': metrics['accuracy']})

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        binary_predictions = (outputs >= 0.5).float()
        loss = self.criterion(binary_predictions, labels)
        metrics = calculate_metrics(binary_predictions, labels)
        self.log_dict({'test_loss': loss, 'test_accuracy': metrics['accuracy']})

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
