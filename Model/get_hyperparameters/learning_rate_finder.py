import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from Model.pointnet import SpineSegmentationNet
from Archive.pointcloud_normalization import normalize_point_cloud
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


# Define Dataset
class PointCloudDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        point_cloud, labels = normalize_point_cloud(self.file_list[idx])  # Adjust according to your function
        return point_cloud, labels

logger = TensorBoardLogger("tb_logs", name="Spine_Segmentation")

# Define Learning Rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# Model Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='my_model/',
    filename='my_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

# Load your dataset
# Suppose you have a list of point cloud files for training
train_files = ['file1.pcd', 'file2.pcd', 'file3.pcd', ...]
dataset = PointCloudDataset(train_files)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define your model
model = SpineSegmentationNet()

# Use PyTorch Lightning's built-in learning rate finder
# PyTorch Lightning trainer with Tensorboard Logger and Callbacks
trainer = pl.Trainer(
    gpus=1,
    max_epochs=10,
    logger=logger,
    callbacks=[lr_monitor, checkpoint_callback],
)
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

# Test the model
trainer.test(model, dataloader)