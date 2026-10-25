import torch

from src.evaluation.metrics import compute_metrics


def evaluate(model, loader, device, num_classes=30):
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if not all_labels:
        raise ValueError("Cannot evaluate an empty loader.")

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        num_classes=num_classes,
    )

    return {
        "metrics": metrics,
        "labels": all_labels,
        "predictions": all_predictions,
    }