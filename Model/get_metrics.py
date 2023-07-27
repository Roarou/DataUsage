import torchmetrics

# Define metrics outside of the function
iou_metric = torchmetrics.IoU(num_classes=2)
f1_metric = torchmetrics.F1(num_classes=2, average='weighted')
precision_metric = torchmetrics.Precision(num_classes=2, average='weighted')
recall_metric = torchmetrics.Recall(num_classes=2, average='weighted')


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
    iou = iou_metric(pred_classes, target)
    f1 = f1_metric(pred_classes, target)
    precision = precision_metric(pred_classes, target)
    recall = recall_metric(pred_classes, target)
    # print(f"iou:  {iou}, precision: {precision}, recall: {recall}, f1: {f1}")
    # Put metrics in dictionary
    metrics = {
        'IoU': iou,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

    return metrics


if __name__ == "__main__":
    pred = 0
    target = 0
    calculate_metrics(pred,target)