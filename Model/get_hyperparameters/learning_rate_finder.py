from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from Model.load_dataset import PointcloudDataset  # Import your custom dataset class
from Model.get_hyperparameters.lr_spine_segmentation_net import PointCloudModule

# Define the paths and parameters
data_path = r'G:\SpineDepth'
batch_size = 1

# Create dataset and dataloaders
train_dataset = PointcloudDataset(root_dir=data_path, split='train')
val_dataset = PointcloudDataset(root_dir=data_path, split='val')
test_dataset = PointcloudDataset(root_dir=data_path, split='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Lightning module
model = PointCloudModule(data_path=data_path)

# Initialize TensorBoard logger
logger = TensorBoardLogger(save_dir='logs', name='spine_segmentation_experiment')

# Initialize the Learning Rate Monitor callback
lr_monitor = LearningRateMonitor(logging_interval='step')

# Initialize the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='model-{epoch:02d}',
    save_top_k=-1,  # Save all checkpoints after each epoch
    monitor='val_loss',  # Monitor the validation loss for checkpointing
    mode='min'  # Save the model with the lowest validation loss
)

# Initialize the Trainer for learning rate finder
lr_finder_trainer = pl.Trainer(max_epochs=1, logger=logger, callbacks=[lr_monitor], auto_scale_batch_size='binsearch')

# Find optimal learning rate
lr_finder = lr_finder_trainer.tuner.lr_find(model, train_dataloader)

# Plot learning rate finder results
fig = lr_finder.plot(suggest=True)
fig.show()

# Pick point with steepest gradient descent
new_lr = lr_finder.suggestion()

# Update learning rate of model
model.hparams.learning_rate = new_lr

# Initialize the Trainer for actual training
trainer = pl.Trainer(max_epochs=10, logger=logger, callbacks=[checkpoint_callback])

# Start training
trainer.fit(model, train_dataloader, val_dataloader=val_dataloader)

# Test the model
trainer.test(model, test_dataloaders=test_dataloader)
