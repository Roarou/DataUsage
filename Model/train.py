import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Model.spine_segmentation import SpineSegmentationNet
from Model.load_dataset import PointcloudDataset  # Replace with the proper file name
from Model.get_metrics import calculate_metrics

batch = 24
max_epochs = 50  # You can set the maximum number of epochs as per your requirement
patience = 2
wait = 0
best_val_loss = float('inf')

def train(model, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    criterion = nn.BCELoss()
    progress_bar = tqdm(train_loader, desc='Train Epoch: {}'.format(epoch))
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        binary_predictions = (output >= 0.5).float()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        metrics = calculate_metrics(binary_predictions, target)

        writer.add_scalar('Training loss', total_loss / (batch_idx + 1), epoch)
        writer.add_scalar('F1', metrics['F1'], epoch)
        writer.add_scalar('IoU', metrics['IoU'], epoch)
        writer.add_scalar('Precision', metrics['Precision'], epoch)
        writer.add_scalar('Recall', metrics['Recall'], epoch)
    writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)


def test(model, test_loader, epoch, writer, mode='Test'):
    model.eval()
    total_loss = 0
    criterion = nn.BCELoss()
    progress_bar = tqdm(test_loader, desc=f'{mode} Epoch: {epoch}')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
    loss = total_loss / len(test_loader)
    print(f'{mode} Loss: {loss}')
    writer.add_scalar(f'{mode.lower()}_loss', loss, epoch)
    return loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpineSegmentationNet().to(device)
    print('cuda') if torch.cuda.is_available() else print('cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Prepare DataLoader for train, test, validation
    base_path = r'L:\groundtruth_labeled' # Path to your dataset
    num_points = 20000  # Number of points to sample

    train_dataset = PointcloudDataset(base_path=base_path, split='train', num_points=num_points)
    test_dataset = PointcloudDataset(base_path=base_path, split='test', num_points=num_points)
    validation_dataset = PointcloudDataset(base_path=base_path, split='val', num_points=num_points)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch)
    validation_loader = DataLoader(validation_dataset, batch_size=batch)

    # TensorBoard Writer
    writer = SummaryWriter(log_dir='logs')

    best_test_loss = float('inf')
    for epoch in range(max_epochs):
        epoch = epoch + 1
        train(model, train_loader, optimizer, epoch, writer)
        validation_loss = test(model, validation_loader, epoch, writer, mode='Validation')
        # Save model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_weights = torch.save(model.state_dict(), f'lr_0,0001_all_data\model_dic_epoch_{epoch}.pt')
            print(f'New best loss: {best_val_loss}')
            torch.save(model, f'lr_0,0001_all_data\model_1_epoch_{epoch}.pth')
            wait = 0  # Reset the waiting counter if there is an improvement
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    # reload the best model here before testing if you want to use the best weights
    model.load_state_dict(best_model_weights)
    test_loss = test(model, test_loader, epoch, writer)
    print(f'Test loss: {test_loss}')
    writer.close()
