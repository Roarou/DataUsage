import torch.optim as optim
import torch
from torch import nn
from tqdm import tqdm
from Model.pointnet import SpineSegmentationNet
from get_metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter
from Model.load_dataset import PointcloudDataset  # Import your custom dataset class
from torch.utils.data import DataLoader

# Define the model
model = SpineSegmentationNet()
if torch.cuda.is_available():
    model = model.cuda()

# Create a SummaryWriter
writer = SummaryWriter('../runs/spine_segmentation_experiment_1')

# Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 10
batch_size = 20
# Define your dataset and dataloader
train_dataset = PointcloudDataset()  # Use appropriate parameters
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Adjust batch_size as neede
# Define validation and test datasets and dataloaders
val_dataset = PointcloudDataset(split='val')  # Use appropriate parameters
test_dataset = PointcloudDataset(split='test')  # Use appropriate parameters
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
for batch_idx, (inputs, labels) in enumerate(train_dataloader):
    print('test')
    print(f"Batch {batch_idx + 1}:")
    print("Inputs:", inputs)
    print("Labels:", labels)
    print("-" * 40)  # Separator between batches
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (inputs, labels) in progress_bar:
        inputs = inputs.float()
        labels = labels.float()
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        binary_predictions = (outputs >= 0.5).float()
        loss = criterion(binary_predictions, labels)
        loss.backward()
        optimizer.step()
        metrics = calculate_metrics(binary_predictions, labels)

        writer.add_scalar('Training loss', running_loss/(i+1), epoch)
        writer.add_scalar('Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Precision', metrics['precision'], epoch)
        writer.add_scalar('Recall', metrics['recall'], epoch)

        running_loss += loss.item()
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(i+1):.4f}, Metrics: {metrics}")

    progress_bar.close()

    # Validation
    model.eval()
    val_loss = 0.0
    val_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.float()
            labels = labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            binary_predictions = (outputs >= 0.5).float()
            loss = criterion(binary_predictions, labels)
            val_loss += loss.item()
            val_metrics_batch = calculate_metrics(binary_predictions, labels)

            for metric in val_metrics:
                val_metrics[metric] += val_metrics_batch[metric]

    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_metrics = {metric: value / len(val_dataloader) for metric, value in val_metrics.items()}

    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', avg_val_metrics['accuracy'], epoch)
    writer.add_scalar('Validation Precision', avg_val_metrics['precision'], epoch)
    writer.add_scalar('Validation Recall', avg_val_metrics['recall'], epoch)

    # Testing
    model.eval()
    test_loss = 0.0
    test_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.float()
            labels = labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            binary_predictions = (outputs >= 0.5).float()
            loss = criterion(binary_predictions, labels)
            test_loss += loss.item()
            test_metrics_batch = calculate_metrics(binary_predictions, labels)

            for metric in test_metrics:
                test_metrics[metric] += test_metrics_batch[metric]

    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_metrics = {metric: value / len(test_dataloader) for metric, value in test_metrics.items()}

    writer.add_scalar('Test Loss', avg_test_loss, epoch)
    writer.add_scalar('Test Accuracy', avg_test_metrics['accuracy'], epoch)
    writer.add_scalar('Test Precision', avg_test_metrics['precision'], epoch)
    writer.add_scalar('Test Recall', avg_test_metrics['recall'], epoch)

    # Save the model after each epoch
    model_save_path = f'model_epoch_{epoch + 1}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved at: {model_save_path}')

# After training, close the SummaryWriter
writer.close()

print('Finished Training')
