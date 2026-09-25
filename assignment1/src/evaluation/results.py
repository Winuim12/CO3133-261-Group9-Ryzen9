#results.py

import json
from pathlib import Path

def save_evaluation_results(path, model_name, checkpoint_path, test_results):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "test_samples": int(test_results["targets"].numel()),
        "loss": float(test_results["loss"]),
        "accuracy": float(test_results["accuracy"]),
        "macro_f1": float(test_results["macro_f1"]),
        "trainable_parameters": int(test_results["trainable_parameters"]),
        "inference_time_seconds": float(test_results["inference_time_seconds"]),
        "inference_time_per_sample_seconds": float(test_results["inference_time_per_sample_seconds"]),
        "confusion_matrix": test_results["confusion_matrix"].tolist(),
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def save_training_results(path, model_name, shared_config, model_config, checkpoint_path, history, training_time_seconds):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_losses = history["validation_loss"]
    requested_epochs = int(shared_config["training"]["epochs"])
    completed_epochs = len(validation_losses)
    best_validation_loss = min(validation_losses)
    best_epoch = validation_losses.index(best_validation_loss) + 1

    summary = {
        "model_name": model_name,
        "seed": int(shared_config["seed"]),
        "data_config": shared_config["data"],
        "training_config": shared_config["training"],
        "evaluation_config": shared_config["evaluation"],
        "model_parameters": model_config["model"]["parameters"],
        "checkpoint_path": str(checkpoint_path),
        "training_time_seconds": float(training_time_seconds),
        "history": history,
        "requested_epochs": requested_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stopped_early": completed_epochs < requested_epochs,
    }

    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")