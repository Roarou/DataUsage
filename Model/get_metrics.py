import torch
import numpy as np


# Define custom metrics functions
def calculate_iou(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    iou = intersection / (union + 1e-7)  # Adding a small epsilon to avoid division by zero
    return iou


def calculate_f1(pred, target):
    # Calculate F1 score
    tp = torch.sum(pred * target)
    fp = torch.sum(pred) - tp
    fn = torch.sum(target) - tp
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1


def calculate_precision(pred, target):
    tp = torch.sum(pred * target)
    fp = torch.sum(pred) - tp
    precision = tp / (tp + fp + 1e-7)
    return precision


def calculate_recall(pred, target):
    tp = torch.sum(pred * target)
    fn = torch.sum(target) - tp
    recall = tp / (tp + fn + 1e-7)
    return recall


def calculate_metrics(pred, target):
    """
    Calculate evaluation metrics for segmentation task.

    Parameters:
    pred (torch.Tensor): Predictions tensor.
    target (torch.Tensor): Ground truth tensor.

    Returns:
    dict: A dictionary containing all the calculated metrics.
    """
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    # Apply threshold to get discrete class predictions
    pred_classes = (pred >= 0.5).float()

    # Calculate metrics
    iou = calculate_iou(pred_classes, target)
    f1 = calculate_f1(pred_classes, target)
    precision = calculate_precision(pred_classes, target)
    recall = calculate_recall(pred_classes, target)

    # Put metrics in dictionary
    metrics = {
        'IoU': iou.item(),
        'F1': f1.item(),
        'Precision': precision.item(),
        'Recall': recall.item()
    }

    return metrics


if __name__ == "__main__":
    pred = torch.tensor([0.3, 0.6, 0.8])
    target = torch.tensor([0, 1, 1])
    metrics = calculate_metrics(pred, target)
    print(metrics)
