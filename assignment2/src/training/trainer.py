import torch

from src.evaluation.metrics import compute_metrics


def run_epoch(
    *,
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    num_classes=30,
):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.detach().argmax(dim=1)

            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty loader.")

    metrics = compute_metrics(
        all_labels,
        all_predictions,
        num_classes=num_classes,
    )
    metrics["loss"] = total_loss / total_samples

    return metrics