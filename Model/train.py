import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from Model.pointnet import SpineSegmentationNet
from Model.pointcloud_normalization import normalize_point_cloud

# Define the model
model = SpineSegmentationNet()
if torch.cuda.is_available():
    model = model.cuda()

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

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    progress_bar.close()

print('Finished Training')