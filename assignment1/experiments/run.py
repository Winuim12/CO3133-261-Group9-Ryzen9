#run.py

from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

from torch import device, nn

from experiments.context import ExperimentContext, prepare_experiment_context

from src.models.factory import create_model
from src.training.optimizer import create_optimizer
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.training.checkpoint import load_checkpoint
from src.evaluation.results import save_evaluation_results, save_training_results
from src.evaluation.confusion_matrix import save_confusion_matrix_plot
from src.evaluation.curves import save_training_curves
from src.evaluation.error_analysis import save_prediction_examples
from src.tracking.run_paths import create_run_paths, load_run_paths

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

def run_training(context: ExperimentContext, run_name=None):
    shared_config = context.shared_config
    model_config = context.model_config
    model_settings = model_config["model"]

    run_paths = create_run_paths(
        results_root = PROJECT_ROOT / "results" / "runs",
        model_name = context.model_name,
        run_name = run_name,
    )

    checkpoint_path = run_paths["checkpoint_path"]

    print(
        f"Training {model_settings['name']} on {context.device} for {shared_config['training']['epochs']} epochs...", 
        flush = True
    )

    model, history, training_time_seconds = run_experiment(
        model_name=model_settings["name"],
        model_parameters=model_settings["parameters"],
        training_config=shared_config["training"],
        train_loader=context.train_loader,
        validation_loader=context.validation_loader,
        checkpoint_path=checkpoint_path,
        device=context.device,
    )

    save_training_results(
        path=run_paths["training_results_path"],
        model_name=context.model_name,
        shared_config=shared_config,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        history=history,
        training_time_seconds=training_time_seconds,
    )

    save_training_curves(
        history=history,
        path=run_paths["training_curves_path"],
        title=f"{model.__class__.__name__} Training Curves",
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {context.device}")
    print(f"Training samples: {len(context.train_dataset):,}")
    print(f"Validation samples: {len(context.validation_dataset):,}")
    print(f"Training mean: {context.mean: .6f}")
    print(f"Training standard deviation: {context.std: .6f}")
    print(f"Epochs: {shared_config['training']['epochs']}")
    print(f"Training time: {training_time_seconds:.2f} seconds")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print("Final training accuracy: "f"{history['train_accuracy'][-1]:.2%}")
    print("Final validation loss: "f"{history['validation_loss'][-1]:.4f}")
    print("Final validation accuracy: "f"{history['validation_accuracy'][-1]:.2%}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training results: {run_paths['training_results_path']}")
    print(f"Training curves: {run_paths['training_curves_path']}")
    print(f"Run ID: {run_paths['run_id']}")
    print(f"Run directory: {run_paths['run_directory']}")


    return {
        "model": model,
        "history": history,
        "training_time_seconds": training_time_seconds,
        "run_id": run_paths["run_id"],
        "run_directory": run_paths["run_directory"],
    }

def run_evaluation(context: ExperimentContext, run_id):
    model_settings = context.model_config["model"]

    model = create_model(
        model_name=model_settings["name"],
        model_parameters=model_settings["parameters"],
    )

    run_paths = load_run_paths(
        results_root = PROJECT_ROOT / "results" / "runs",
        model_name = context.model_name,
        run_id = run_id,
    )

    checkpoint_path = run_paths["checkpoint_path"]

    print(
        f"Evaluating {model_settings['name']} on {context.device}...", 
        flush = True
    )

    test_results = evaluate_best_checkpoint(
        model=model,
        test_loader=context.test_loader,
        checkpoint_path=checkpoint_path,
        device=context.device,
    )

    save_evaluation_results(
        path=run_paths["evaluation_results_path"],
        model_name=context.model_name,
        checkpoint_path=checkpoint_path,
        test_results=test_results,
    )

    save_confusion_matrix_plot(
        matrix=test_results["confusion_matrix"],
        class_names=context.test_dataset.classes,
        path=run_paths["confusion_matrix_path"],
        title=(f"{model.__class__.__name__} Confusion Matrix"),
    )

    save_prediction_examples(
        dataset=context.test_dataset,
        targets=test_results["targets"],
        predictions=test_results["predictions"],
        class_names=context.test_dataset.classes,
        mean=context.mean,
        std=context.std,
        path=run_paths["prediction_examples_path"],
        examples_per_group=5,
        title=(f"{model.__class__.__name__} Prediction Examples"),
    )

    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {context.device}")
    print(f"Test samples: {len(context.test_dataset):,}")
    print(f"Training mean: {context.mean: .6f}")
    print(f"Training standard deviation: {context.std: .6f}")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Test loss: {test_results['loss']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.2%}")
    print(f"Test Macro-F1: {test_results['macro_f1']:.4f}")
    print("Trainable parameters: "f"{test_results['trainable_parameters']:,}")
    print("Inference time: "f"{test_results['inference_time_seconds']:.4f} seconds")
    print("Inference time per sample: "f"{test_results['inference_time_per_sample_seconds'] * 1000:.4f} ms")
    print(f"Evaluation results: {run_paths['evaluation_results_path']}")
    print(f"Confusion matrix: {run_paths['confusion_matrix_path']}")
    print(f"Prediction examples: {run_paths['prediction_examples_path']}")
    print(f"Run ID: {run_paths['run_id']}")
    print(f"Run directory: {run_paths['run_directory']}")
    
    return {
        "model": model,
        "test_results": test_results,
        "run_id": run_paths["run_id"],
        "run_directory": run_paths["run_directory"],
    }



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

    parser.add_argument(
        "--run-id",
        default=None,
        help=("Identifier of an existing versioned used by evaluation-only mode. ")
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

    parsed_arguments = parser.parse_args(arguments)

    if parsed_arguments.evaluate_only and parsed_arguments.run_id is None:
        parser.error("--evaluate-only requires --run-id")

    if parsed_arguments.run_id is not None and not parsed_arguments.evaluate_only:
        parser.error("--run-id can only be used with --evaluate-only")

    return parsed_arguments

def main():
    arguments = parse_arguments()
    epochs_override = (None if arguments.evaluate_only else arguments.epochs)
    context = prepare_experiment_context(
        model_name=arguments.model,
        epochs_override=epochs_override,
    )
    

    if arguments.evaluate_only:
        run_evaluation(
            context=context,
            run_id=arguments.run_id
        )

        return
    
    training_results = run_training(
        context=context,
        run_name=arguments.run_name,
    )

    if arguments.evaluate_test:
        run_evaluation(
            context=context,
            run_id=training_results["run_id"],
        )

if __name__ == "__main__":
    main()