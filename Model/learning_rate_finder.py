import pytorch_lightning as pl
from Model.pointnet import SpineSegmentationNet

# Define your model
model = SpineSegmentationNet()

# Define your trainer
trainer = pl.Trainer()

# Run learning rate finder
lr_finder = trainer.tuner.lr_find(model)

# Plot learning rate
fig = lr_finder.plot(suggest=True)
fig.show()

# Get the optimal learning rate
new_lr = lr_finder.suggestion()
print("Suggested learning rate: ", new_lr)