import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from Model.pointnet import SpineSegmentationNet
from Model.pointcloud_normalization import normalize_point_cloud
from get_metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter
# Try different losses, focal loss, cross entropy loss
# Try different optimizers, ADAM, SGD,

# Define the model
model = SpineSegmentationNet()
if torch.cuda.is_available():
    model = model.cuda()

# Create a SummaryWriter
writer = SummaryWriter('runs/spine_segmentation_experiment_1')

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 10

# Load your dataset
# Suppose you have a list of point cloud files for training
train_files = ['file1.pcd', 'file2.pcd', 'file3.pcd', ...]
for epoch in range(epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_files), total=len(train_files))
    for i, file in progress_bar:
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = normalize_point_cloud(file)  # adjust according to your function
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)

        # Apply sigmoid activation to output to get probabilities
        outputs = torch.sigmoid(outputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Calculate metrics
        metrics = calculate_metrics(outputs, labels)

        # Log metrics to TensorBoard
        writer.add_scalar('Training loss', running_loss/(i+1), epoch)
        writer.add_scalar('Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Precision', metrics['precision'], epoch)
        writer.add_scalar('Recall', metrics['recall'], epoch)

        # Print statistics
        running_loss += loss.item()
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(i+1):.4f}, Metrics: {metrics}")

    progress_bar.close()

# After training, close the SummaryWriter
writer.close()

print('Finished Training')

