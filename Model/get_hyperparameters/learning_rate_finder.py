import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder
import torch
import torch.optim as optim
from torch import nn
from Model.pointnet import SpineSegmentationNet
from Model.pointcloud_normalization import normalize_point_cloud
from get_metrics import calculate_metrics


# Define Dataset
class PointCloudDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        point_cloud, labels = normalize_point_cloud(self.file_list[idx])  # Adjust according to your function
        return point_cloud, labels


# Load your dataset
# Suppose you have a list of point cloud files for training
train_files = ['file1.pcd', 'file2.pcd', 'file3.pcd', ...]
dataset = PointCloudDataset(train_files)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define your model
model = SpineSegmentationModule()

# Use PyTorch Lightning's built-in learning rate finder
trainer = pl.Trainer(gpus=1, max_epochs=1)
lr_finder = trainer.tuner.lr_find(model, dataloader)

# Plot learning rate finder results
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point with steepest gradient descent
new_lr = lr_finder.suggestion()

# Update learning rate of model
model.learning_rate = new_lr

# Train the model
trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(model, dataloader)