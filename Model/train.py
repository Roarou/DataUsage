import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Model.spine_segmentation import SpineSegmentationNet
from Model.load_dataset import PointcloudDataset  # Replace with the proper file name
from Model.get_metrics import calculate_metrics


def train(model, train_loader, optimizer, epoch, writer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Train Epoch: {}'.format(epoch))
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        binary_predictions = (output >= 0.5).float()
        progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        metrics = calculate_metrics(binary_predictions, output)

        writer.add_scalar('Training loss', total_loss / (batch_idx + 1), epoch)
        writer.add_scalar('Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Precision', metrics['precision'], epoch)
        writer.add_scalar('Recall', metrics['recall'], epoch)
    writer.add_scalar('train_loss', total_loss / len(train_loader), epoch)


def test(model, test_loader, epoch, writer, mode='Test'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = F.binary_cross_entropy(output, target)
            total_loss += loss.item()
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
    base_path = r'G:\SpineDepth\groundtruth_labeled'  # Path to your dataset
    num_points = 340000  # Number of points to sample

    train_dataset = PointcloudDataset(base_path=base_path, split='train', num_points=num_points)
    test_dataset = PointcloudDataset(base_path=base_path, split='test', num_points=num_points)
    validation_dataset = PointcloudDataset(base_path=base_path, split='val', num_points=num_points)

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=3)
    validation_loader = DataLoader(validation_dataset, batch_size=3)

    # TensorBoard Writer
    writer = SummaryWriter()

    best_test_loss = float('inf')
    for epoch in range(1, 10):
        train(model, train_loader, optimizer, epoch, writer)
        test_loss = test(model, test_loader, epoch, writer)
        validation_loss = test(model, validation_loader, epoch, writer, mode='Validation')

        # Save model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print(f'New best loss: {best_test_loss}')
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')

    writer.close()

