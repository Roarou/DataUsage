import os
import wandb
import random
import numpy as np
from tqdm.auto import tqdm
from graph_cnn import DGCNN
import torch
import torch.nn.functional as F

from torch_scatter import scatter
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv
wandb_project = "pyg-point-cloud" #@param {"type": "string"}
wandb_run_name = "train-dgcnn" #@param {"type": "string"}

wandb.init(project=wandb_project, name=wandb_run_name, job_type="train")

config = wandb.config

config.seed = 42
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(config.seed)
torch.manual_seed(config.seed)
device = torch.device(config.device)

config.category = 'Airplane' #@param ["Bag", "Cap", "Car", "Chair", "Earphone", "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug", "Pistol", "Rocket", "Skateboard", "Table"] {type:"raw"}
config.random_jitter_translation = 1e-2
config.random_rotation_interval_x = 15
config.random_rotation_interval_y = 15
config.random_rotation_interval_z = 15
config.validation_split = 0.2
config.batch_size = 16
config.num_workers = 6

config.num_nearest_neighbours = 30
config.aggregation_operator = "max"
config.dropout = 0.5
config.initial_lr = 1e-3
config.lr_scheduler_step_size = 5
config.gamma = 0.8

config.epochs = 1
segmentation_class_frequency = {}
for idx in tqdm(range(len(train_val_dataset))):
    pc_viz = train_val_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_val_dataset[idx].y.numpy().tolist()
    for label in set(segmentation_label):
        segmentation_class_frequency[label] = segmentation_label.count(label)
class_offset = min(list(segmentation_class_frequency.keys()))
print("Class Offset:", class_offset)

for idx in range(len(train_val_dataset)):
    train_val_dataset[idx].y -= class_offset

num_train_examples = int((1 - config.validation_split) * len(train_val_dataset))
train_dataset = train_val_dataset[:num_train_examples]
val_dataset = train_val_dataset[num_train_examples:]

config.num_classes = train_dataset.num_classes

model = DGCNN(
    out_channels=train_dataset.num_classes,
    k=config.num_nearest_neighbours,
    aggr=config.aggregation_operator
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config.lr_scheduler_step_size, gamma=config.gamma
)


def train_step(epoch):
    model.train()

    ious, categories = [], []
    total_loss = correct_nodes = total_nodes = 0
    y_map = torch.empty(
        train_loader.dataset.num_classes, device=device
    ).long()
    num_train_examples = len(train_loader)

    progress_bar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}/{config.epochs}"
    )

    for data in progress_bar:
        data = data.to(device)

        optimizer.zero_grad()
        outs = model(data)
        loss = F.nll_loss(outs, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = jaccard_index(
                out[:, part].argmax(dim=-1), y_map[y],
                task="multiclass", num_classes=part.size(0)
            )
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)
    mean_iou = float(scatter(iou, category, reduce='mean').mean())

    return {
        "Train/Loss": total_loss / num_train_examples,
        "Train/Accuracy": correct_nodes / total_nodes,
        "Train/IoU": mean_iou
    }


@torch.no_grad()
def val_step(epoch):
    model.eval()

    ious, categories = [], []
    total_loss = correct_nodes = total_nodes = 0
    y_map = torch.empty(
        val_loader.dataset.num_classes, device=device
    ).long()
    num_val_examples = len(val_loader)

    progress_bar = tqdm(
        val_loader, desc=f"Validating Epoch {epoch}/{config.epochs}"
    )

    for data in progress_bar:
        data = data.to(device)
        outs = model(data)

        loss = F.nll_loss(outs, data.y)
        total_loss += loss.item()

        correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = jaccard_index(
                out[:, part].argmax(dim=-1), y_map[y],
                task="multiclass", num_classes=part.size(0)
            )
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)
    mean_iou = float(scatter(iou, category, reduce='mean').mean())

    return {
        "Validation/Loss": total_loss / num_val_examples,
        "Validation/Accuracy": correct_nodes / total_nodes,
        "Validation/IoU": mean_iou
    }


@torch.no_grad()
def visualization_step(epoch, table):
    model.eval()
    for data in tqdm(visualization_loader):
        data = data.to(device)
        outs = model(data)

        predicted_labels = outs.argmax(dim=1)
        accuracy = predicted_labels.eq(data.y).sum().item() / data.num_nodes

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        ious, categories = [], []
        y_map = torch.empty(
            visualization_loader.dataset.num_classes, device=device
        ).long()
        for out, y, category in zip(
                outs.split(sizes), data.y.split(sizes), data.category.tolist()
        ):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)
            y_map[part] = torch.arange(part.size(0), device=device)
            iou = jaccard_index(
                out[:, part].argmax(dim=-1), y_map[y],
                task="multiclass", num_classes=part.size(0)
            )
            ious.append(iou)
        categories.append(data.category)
        iou = torch.tensor(ious, device=device)
        category = torch.cat(categories, dim=0)
        mean_iou = float(scatter(iou, category, reduce='mean').mean())

        gt_pc_viz = data.pos.cpu().numpy().tolist()
        segmentation_label = data.y.cpu().numpy().tolist()
        frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
        for label in set(segmentation_label):
            frequency_dict[label] = segmentation_label.count(label)
        for j in range(len(gt_pc_viz)):
            # gt_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
            gt_pc_viz[j] += [segmentation_label[j] + 1]

        predicted_pc_viz = data.pos.cpu().numpy().tolist()
        segmentation_label = data.y.cpu().numpy().tolist()
        frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
        for label in set(segmentation_label):
            frequency_dict[label] = segmentation_label.count(label)
        for j in range(len(predicted_pc_viz)):
            # predicted_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
            predicted_pc_viz[j] += [segmentation_label[j] + 1]

        table.add_data(
            epoch, wandb.Object3D(np.array(gt_pc_viz)),
            wandb.Object3D(np.array(predicted_pc_viz)),
            accuracy, mean_iou
        )

    return table


def save_checkpoint(epoch):
    """Save model checkpoints as Weights & Biases artifacts"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "checkpoint.pt")

    artifact_name = wandb.util.make_artifact_name_safe(
        f"{wandb.run.name}-{wandb.run.id}-checkpoint"
    )

    checkpoint_artifact = wandb.Artifact(artifact_name, type="checkpoint")
    checkpoint_artifact.add_file("checkpoint.pt")
    wandb.log_artifact(
        checkpoint_artifact, aliases=["latest", f"epoch-{epoch}"]
    )


table = wandb.Table(columns=["Epoch", "Ground-Truth", "Prediction", "Accuracy", "IoU"])

for epoch in range(1, config.epochs + 1):
    train_metrics = train_step(epoch)
    val_metrics = val_step(epoch)

    metrics = {**train_metrics, **val_metrics}
    metrics["learning_rate"] = scheduler.get_last_lr()[-1]
    wandb.log(metrics)

    table = visualization_step(epoch, table)

    scheduler.step()
    save_checkpoint(epoch)

wandb.log({"Evaluation": table})

wandb.finish()