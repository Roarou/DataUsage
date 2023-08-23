import torch
import numpy as np


# Define custom metrics functions
def calculate_iou(predicted, target):
    """
    Compute Intersection over Union (IoU) for binary segmentation.

    IoU is the ratio of the intersection of predicted and actual positives to their union.
    It measures the overlap between the predicted and actual positive regions.
    """
    intersection = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 1)).item()
    union = torch.sum(torch.eq(predicted, 1) | torch.eq(target, 1)).item()

    iou = intersection / union if union != 0 else 0.0

    return iou


def calculate_f1_score(predicted, target):
    """
    Compute F1 score for binary segmentation.

    F1 score is the harmonic mean of precision and recall.
    It balances precision and recall, giving a single metric for evaluation.
    """
    true_positive = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 1)).item()
    false_positive = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 0)).item()
    false_negative = torch.sum(torch.eq(predicted, 0) & torch.eq(target, 1)).item()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return f1_score


def calculate_precision(predicted, target):
    """
    Compute precision for binary segmentation.

    Precision measures how many of the positively predicted cases were true positives.
    It's the ratio of true positives to the total predicted positives.
    """
    true_positive = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 1)).item()
    false_positive = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 0)).item()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0

    return precision


def calculate_recall(predicted, target):
    """
    Compute recall for binary segmentation.

    Recall measures how many of the actual positives were correctly predicted.
    It's the ratio of true positives to the total actual positives.
    """
    true_positive = torch.sum(torch.eq(predicted, 1) & torch.eq(target, 1)).item()
    false_negative = torch.sum(torch.eq(predicted, 0) & torch.eq(target, 1)).item()

    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

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
    f1 = calculate_f1_score(pred_classes, target)
    precision = calculate_precision(pred_classes, target)
    recall = calculate_recall(pred_classes, target)

    # Put metrics in dictionary
    metrics = {
        'IoU': iou,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

    return metrics


if __name__ == "__main__":
    pred = torch.tensor([0.3, 0.6, 0.8])
    target = torch.tensor([0, 1, 1])
    metrics = calculate_metrics(pred, target)
    print(metrics)
