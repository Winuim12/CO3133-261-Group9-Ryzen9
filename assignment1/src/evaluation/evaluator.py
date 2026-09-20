# evaluator.py

import torch

from src.evaluation.metrics import calculate_macro_f1, count_trainable_parameters
from src.evaluation.confusion_matrix import calculate_confusion_matrix

from time import perf_counter

def evaluate_model(model, dataloader, loss_function, device):
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_inference_time = 0.0

    all_predictions = []
    all_targets = []

    #Warm up the model
    warmup_batch = next(iter(dataloader), None)

    if warmup_batch is not None:
        warmup_images, _ = warmup_batch
        warmup_images = warmup_images.to(device)

        with torch.no_grad():
            model(warmup_images)

        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    with torch.no_grad():
        for image, targets in dataloader:
            images = image.to(device)
            targets = targets.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

            inference_start = perf_counter()
            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

            total_inference_time += perf_counter() - inference_start

            num_classes = logits.size(1)
            loss = loss_function(logits, targets)

            predictions = logits.argmax(dim=1)
            batch_size = images.size(0)
            
            total_loss += loss.item() * batch_size
            total_correct += (predictions == targets).sum().item()
            total_samples += batch_size

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    macro_f1 = calculate_macro_f1(predictions, targets)
    trainable_parameters = count_trainable_parameters(model)
    confusion_matrix = calculate_confusion_matrix(predictions, targets, num_classes)
    inference_time_per_sample = total_inference_time / total_samples

    return {
        "loss": total_loss / total_samples, 
        "accuracy": total_correct / total_samples,
        "predictions": predictions,
        "targets": targets,
        "macro_f1": macro_f1,
        "trainable_parameters": trainable_parameters,
        "confusion_matrix": confusion_matrix,
        "inference_time_seconds": total_inference_time,
        "inference_time_per_sample_seconds": inference_time_per_sample,
    }