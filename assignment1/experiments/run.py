#run.py

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

from torch import nn

from src.models.factory import create_model
from src.training.optimizer import create_optimizer
from src.training.trainer import train_model
from src.data.dataloader import create_dataloaders
from src.data.pipeline import create_normalized_datasets
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.evaluation.evaluator import evaluate_model
from src.training.checkpoint import load_checkpoint
from src.evaluation.results import save_evaluation_results, save_training_results
from src.evaluation.confusion_matrix import save_confusion_matrix_plot
from src.evaluation.curves import save_training_curves
from src.evaluation.error_analysis import save_prediction_examples
from src.tracking.run_paths import create_run_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_experiment(
    model_name,
    model_parameters,
    training_config,
    train_loader,
    validation_loader,
    checkpoint_path,
    device,
):
    model = create_model(
        model_name=model_name,
        model_parameters=model_parameters,
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=training_config["optimizer"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    training_start = perf_counter()

    history = train_model(
        model=model, 
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epochs=training_config["epochs"],
        checkpoint_path=checkpoint_path
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)

    training_time_seconds = (perf_counter() - training_start)

    return model, history, training_time_seconds

def evaluate_best_checkpoint(
    model,
    test_loader,
    checkpoint_path,
    device,
):
    load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    return evaluate_model(
        model=model,
        dataloader=test_loader,
        loss_function=nn.CrossEntropyLoss(),
        device=device,
    )

def evaluate_test_if_requested(
    evaluate_test,
    model,
    test_loader,
    checkpoint_path,
    device,
):
    if not evaluate_test:
        return None

    return evaluate_best_checkpoint(
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

def run_configured_experiment(
    shared_config,
    model_config,
    train_dataset,
    validation_dataset,
    test_dataset,
    checkpoint_path,
    device,
):

    data_config = shared_config["data"]
    training_config = shared_config["training"]
    model_settings = model_config["model"]

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    model, history, training_time_seconds = run_experiment(
        model_name=model_settings["name"],
        model_parameters=model_settings["parameters"],
        training_config=training_config,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model, history, test_loader, training_time_seconds

def run_model_from_config(
    model_name, 
    epochs_override=None, 
    evaluate_test=False, 
    run_name=None
):
    normalized_name = model_name.lower()

    shared_config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_config = load_config(PROJECT_ROOT / "config" / f"{normalized_name}.yaml")

    if epochs_override is not None: 
        shared_config["training"]["epochs"] = epochs_override

    seed = shared_config["seed"]
    set_seed(seed)

    data_config = shared_config["data"]
    data_directory = (PROJECT_ROOT / "data" / "raw" / "fashion_mnist")

    print("Preparing Fashion-MNIST datasets...", flush=True)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        mean,
        std,
    ) = create_normalized_datasets(
        data_dir = data_directory,
        validation_size = data_config["validation_size"],
        seed = seed, 
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint_path = (PROJECT_ROOT / "results" / "checkpoints" / normalized_name / f"{normalized_name}_best.pt")
    # confusion_matrix_path = PROJECT_ROOT / "results" / "figures" / normalized_name / "confusion_matrix.png"
    # training_curves_path = PROJECT_ROOT / "results" / "figures" / normalized_name / "training_curves.png"
    # prediction_examples_path = PROJECT_ROOT / "results" / "figures" / normalized_name / "prediction_examples.png"
    # metrics_directory = PROJECT_ROOT / "results" / "metrics" / normalized_name

    # training_results_path = metrics_directory / "training.json"
    # evaluation_results_path = metrics_directory / "evaluation.json"

    run_paths = create_run_paths(
        results_root = PROJECT_ROOT / "results" / "runs",
        model_name = normalized_name,
        run_name = run_name,
    )

    checkpoint_path = run_paths["checkpoint_path"]
    training_results_path = run_paths["training_results_path"]
    evaluation_results_path = run_paths["evaluation_results_path"]
    training_curves_path = run_paths["training_curves_path"]
    confusion_matrix_path = run_paths["confusion_matrix_path"]
    prediction_examples_path = run_paths["prediction_examples_path"]

    print(
        f"Training {model_config['model']['name']} on {device} "
        f"for {shared_config['training']['epochs']} epochs...",
        flush=True,
    )

    model, history, test_loader, training_time_seconds = run_configured_experiment(
        shared_config=shared_config,
        model_config=model_config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    save_training_results(
        path=training_results_path,
        model_name=normalized_name,
        shared_config=shared_config,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        history=history,
        training_time_seconds=training_time_seconds,
    )

    save_training_curves(
        history=history,
        path=training_curves_path,
        title=f"{model.__class__.__name__} Training Curves",
    )

    test_results = evaluate_test_if_requested(
        evaluate_test=evaluate_test,
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Training mean: {mean: .6f}")
    print(f"Training standard deviation: {std: .6f}")
    print(f"Epochs: {shared_config['training']['epochs']}")
    print(f"Training time: {training_time_seconds:.2f} seconds")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print("Final training accuracy: "f"{history['train_accuracy'][-1]:.2%}")
    print("Final validation loss: "f"{history['validation_loss'][-1]:.4f}")
    print("Final validation accuracy: "f"{history['validation_accuracy'][-1]:.2%}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training results: {training_results_path}")
    print(f"Training curves: {training_curves_path}")
    print(f"Run ID: {run_paths['run_id']}")
    print(f"Run directory: {run_paths['run_directory']}")

    if test_results is not None:
        save_evaluation_results(
            path=evaluation_results_path,
            model_name=normalized_name,
            checkpoint_path=checkpoint_path,
            test_results=test_results
        )

        save_confusion_matrix_plot(
            matrix=test_results["confusion_matrix"],
            class_names=test_dataset.classes,
            path=confusion_matrix_path,
            title=(f"{model.__class__.__name__} Confusion Matrix"),
        )

        save_prediction_examples(
            dataset=test_dataset,
            targets=test_results["targets"],
            predictions=test_results["predictions"],
            class_names=test_dataset.classes,
            mean=mean,
            std=std,
            path=prediction_examples_path,
            examples_per_group=5,
            title=(f"{model.__class__.__name__} Prediction Examples"),
        )

        print(f"Test samples: {len(test_dataset):,}")
        print(f"Test loss: {test_results['loss']:.4f}")
        print(f"Test accuracy: {test_results['accuracy']:.2%}")
        print(f"Test Macro-F1: {test_results['macro_f1']:.4f}")
        print("Trainable parameters: "f"{test_results['trainable_parameters']:,}")
        print("Inference time: "f"{test_results['inference_time_seconds']:.4f} seconds")
        print("Inference time per sample: "f"{test_results['inference_time_per_sample_seconds'] * 1000:.4f} ms")
        print(f"Evaluation results: {evaluation_results_path}")
        print(f"Confusion matrix: {confusion_matrix_path}")
        print(f"Prediction examples: {prediction_examples_path}")


    return {
        "model": model,
        "history": history,
        "test_results": test_results,
        "training_time_seconds": training_time_seconds,
        "run_id": run_paths["run_id"],
        "run_directory": run_paths["run_directory"],
    }

def run_evaluation_from_config(model_name, epochs_override=None, evaluate_test=False):
    normalized_name = model_name.lower()

    shared_config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_config = load_config(PROJECT_ROOT / "config" / f"{normalized_name}.yaml")

    if epochs_override is not None: 
        shared_config["training"]["epochs"] = epochs_override

    seed = shared_config["seed"]
    set_seed(seed)

    data_config = shared_config["data"]
    data_directory = (PROJECT_ROOT / "data" / "raw" / "fashion_mnist")

    print("Preparing Fashion-MNIST datasets...", flush=True)

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        mean,
        std,
    ) = create_normalized_datasets(
        data_dir = data_directory,
        validation_size = data_config["validation_size"],
        seed = seed, 
    )

    _, _, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_settings = model_config["model"]
    model = create_model(model_name=model_settings["name"], model_parameters=model_settings["parameters"])
    checkpoint_path = (PROJECT_ROOT / "results" / "checkpoints" / normalized_name / f"{normalized_name}_best.pt")
    evaluation_results_path = PROJECT_ROOT / "results" / "metrics" / normalized_name / "evaluation.json"
    confusion_matrix_path = PROJECT_ROOT / "results" / "figures" / normalized_name / "confusion_matrix.png"
    prediction_examples_path = PROJECT_ROOT / "results" / "figures" / normalized_name / "prediction_examples.png"

    print(
        f"Evaluating {model_settings['name']} on {device}...",
        flush=True,
    )

    test_results = evaluate_best_checkpoint(
        model=model,
        test_loader=test_loader,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    save_evaluation_results(
        path=evaluation_results_path,
        model_name=normalized_name,
        checkpoint_path=checkpoint_path,
        test_results=test_results,
    )

    save_confusion_matrix_plot(
        matrix=test_results["confusion_matrix"],
        class_names=test_dataset.classes,
        path=confusion_matrix_path,
        title=(f"{model.__class__.__name__} Confusion Matrix"),
    )

    save_prediction_examples(
        dataset=test_dataset,
        targets=test_results["targets"],
        predictions=test_results["predictions"],
        class_names=test_dataset.classes,
        mean=mean,
        std=std,
        path=prediction_examples_path,
        examples_per_group=5,
        title=(f"{model.__class__.__name__} Prediction Examples"),
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Test samples: {len(test_dataset):,}")
    print(f"Training mean: {mean: .6f}")
    print(f"Training standard deviation: {std: .6f}")
    print(f"Best checkpoint: {checkpoint_path}")

    print(f"Test loss: {test_results['loss']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.2%}")
    print(f"Test Macro-F1: {test_results['macro_f1']:.4f}")
    print("Trainable parameters: "f"{test_results['trainable_parameters']:,}")
    print("Inference time: "f"{test_results['inference_time_seconds']:.4f} seconds")
    print("Inference time per sample: "f"{test_results['inference_time_per_sample_seconds'] * 1000:.4f} ms")
    print(f"Evaluation results: {evaluation_results_path}")
    print(f"Confusion matrix: {confusion_matrix_path}")
    print(f"Prediction examples: {prediction_examples_path}")

    return model, test_results

def parse_arguments(arguments=None):
    parser = ArgumentParser(description="Train a Fashion-MNIST model.")

    parser.add_argument(
        "--model",
        required=True,
        help="Model configuration name, such as linear, mlp, ...",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Overide the epoch count from config/config.yaml"
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=("Optional human-readable name for a versioned training run.")
    )

    evaluation_group = parser.add_mutually_exclusive_group()

    evaluation_group.add_argument(
        "--evaluate-test",
        action="store_true",
        help=("Train the model, then evaluate the best checkpoint on the test set.")
    )

    evaluation_group.add_argument(
        "--evaluate-only",
        action="store_true",
        help=("Skip training and evaluate the best checkpoint.")
    )

    return parser.parse_args(arguments)

def main():
    arguments = parse_arguments()

    if arguments.evaluate_only:
        run_evaluation_from_config(
            model_name=arguments.model,
        )
        return

    run_model_from_config(
        model_name=arguments.model,
        epochs_override=arguments.epochs,
        evaluate_test=arguments.evaluate_test,
        run_name=arguments.run_name,
    )

if __name__ == "__main__":
    main()