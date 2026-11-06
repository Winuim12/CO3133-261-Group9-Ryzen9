from time import perf_counter

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
    scaler=None,
    channels_last=False,
):
    is_training = optimizer is not None
    model.train(is_training)

    device = torch.device(device)
    total_loss = torch.zeros((), device=device)
    total_samples = 0
    all_labels = []
    all_predictions = []
    data_wait_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = last_batch_end = perf_counter()

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            data_wait_seconds += perf_counter() - last_batch_end
            images = images.to(device, non_blocking=True, memory_format=(
                torch.channels_last if channels_last else torch.preserve_format
            ))
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            # Validation stays FP32 so checkpoint verification uses the same policy.
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=is_training and scaler is not None and scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            predictions = logits.detach().argmax(dim=1)

            # Only retain O(samples) integers, and transfer once per epoch.
            all_labels.append(labels.detach())
            all_predictions.append(predictions)
            last_batch_end = perf_counter()

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty loader.")

    metrics = compute_metrics(
        torch.cat(all_labels).cpu().tolist(),
        torch.cat(all_predictions).cpu().tolist(),
        num_classes=num_classes,
    )
    metrics["loss"] = total_loss.item() / total_samples
    seconds = perf_counter() - start
    metrics["timing"] = {
        "seconds": seconds,
        "samples_per_second": total_samples / seconds,
        "data_wait_seconds": data_wait_seconds,
        "data_wait_scope": "Host wait for loader; can overlap GPU work, not pure IO time",
    }

    return metrics
