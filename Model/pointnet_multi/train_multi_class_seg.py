import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Model.pointnet_multi.spine_multi_class_segmentation import SpineSegmentationNet
from Model.pointnet_multi.load_dataset_multi import PointcloudDataset  # Replace with the proper file name
from Model.get_metrics import calculate_metrics
import time

batch = 24
max_epochs = 25  # You can set the maximum number of epochs as per your requirement
patience = 2
wait = 0
best_val_loss = float('inf')


def train(model, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    weights = torch.tensor([1.11, 50, 50, 50, 50, 50]).to(device)
    criterion = nn.NLLLoss(weight=weights)
    progress_bar = tqdm(train_loader, desc='Train Epoch: {}'.format(epoch))

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output, _ = model(data)
        print(output.size())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        predictions = torch.argmax(output, dim=1)
        metrics = calculate_metrics(predictions, target)
        writer.add_scalar('Training loss', total_loss / (batch_idx + 1), epoch)
        writer.add_scalar('F1', metrics['F1'], epoch)
        writer.add_scalar('IoU', metrics['IoU'], epoch)
        writer.add_scalar('Precision', metrics['Precision'], epoch)
        writer.add_scalar('Recall', metrics['Recall'], epoch)
    writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)


def test(model, test_loader, epoch, writer, mode='Test'):
    model.eval()
    total_loss = 0
    weights = torch.tensor([1.11,50, 50, 50, 50, 50]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    progress_bar = tqdm(test_loader, desc=f'{mode} Epoch: {epoch}')
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            metrics = calculate_metrics(predictions, target)
            writer.add_scalar('Training loss', total_loss / (batch_idx + 1), epoch)
            writer.add_scalar('F1', metrics['F1'], epoch)
            writer.add_scalar('IoU', metrics['IoU'], epoch)
            writer.add_scalar('Precision', metrics['Precision'], epoch)
            writer.add_scalar('Recall', metrics['Recall'], epoch)
    loss = total_loss / len(test_loader)
    print(f'{mode} Loss: {loss}')
    writer.add_scalar(f'{mode.lower()}_loss', loss, epoch)
    return loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpineSegmentationNet().to(device)
    print('cuda') if torch.cuda.is_available() else print('cpu')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare DataLoader for train, test, validation
    base_path = r'L:\groundtruth_labeled'  # Path to your dataset
    num_points = 20000  # Number of points to sample
    tt = time.time()
    train_dataset = PointcloudDataset(base_path=base_path, split='train', num_points=num_points)
    test_dataset = PointcloudDataset(base_path=base_path, split='test', num_points=num_points)
    validation_dataset = PointcloudDataset(base_path=base_path, split='val', num_points=num_points)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch)
    validation_loader = DataLoader(validation_dataset, batch_size=batch)
    print(f'loading {time.time() - tt}')
    # TensorBoard Writer
    writer = SummaryWriter(log_dir='../logs_segmentation')
    try:
        checkpoint = torch.load('../segmentation_multi/model_1_epoch.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    best_test_loss = float('inf')
    for epoch in range(start_epoch, max_epochs):
        epoch = epoch + 1
        train(model, train_loader, optimizer, epoch, writer)
        validation_loss = test(model, validation_loader, epoch, writer, mode='Validation')
        # Save model
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_model_weights = torch.save(model.state_dict(), f'segmentation_multi\model_dic_epoch_{epoch}.pt')
            print(f'New best loss: {best_val_loss}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss,
            }, f'../segmentation_multi/model_1_epoch.pth')
            wait = 0  # Reset the waiting counter if there is an improvement
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    # reload the best model here before testing if you want to use the best weights
    model.load_state_dict('segmentation_multi\model_1_epoch.pth')
    test_loss = test(model, test_loader, epoch, writer)
    print(f'Test loss: {test_loss}')
    writer.close()
