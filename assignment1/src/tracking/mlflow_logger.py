#mlflow_logger.py

import json
from pathlib import Path


def require_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        if exc.name != "mlflow":
            raise
        raise RuntimeError(
            "MLflow is not installed; install it with 'pip instal mlflow' "
            "to use --use-mlflow"
        ) from exc

    return mlflow

def log_epoch_metrics(mlflow_client, history):
    for index in range(len(history["train_loss"])):
        mlflow_client.log_metrics(
            {
                "train_loss": history["train_loss"][index],
                "train_accuracy": history["train_accuracy"][index],
                "validation_loss": history["validation_loss"][index],
                "validation_accuracy": history["validation_accuracy"][index],
            },
            step=index + 1,
        )

def log_run_parameters(mlflow_client, run_id, summary):
    parameters = {
        "local_run_id": run_id,
        "model_name": summary["model_name"],
        "seed": summary["seed"],
    }

    for section, prefix in (
        ("data_config", "data"),
        ("training_config", "training"),
        ("model_parameters", "model"),
    ):
        parameters.update({
            f"{prefix}.{name}": value
            for name, value in summary[section].items()
        }
        )

    mlflow_client.log_params(parameters)

def log_evaluation_metrics(mlflow_client, results):
    mlflow_client.log_metrics({
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
        "test_macro_f1": results["macro_f1"],
        "inference_time_seconds": results["inference_time_seconds"],
    })

def log_training_summary(mlflow_client, summary):
    mlflow_client.log_metrics({
        "best_validation_loss": summary["best_validation_loss"],
        "best_epoch": summary["best_epoch"],
        "completed_epochs": summary["completed_epochs"],
        "training_time_seconds": summary["training_time_seconds"],
    })

def log_saved_run(mlflow_client, run_directory):
    run_directory = Path(run_directory)
    summary = json.loads(
        (run_directory / "training.json").read_text(encoding="utf-8")
    )

    with mlflow_client.start_run(run_name=run_directory.name):
        log_run_parameters(mlflow_client, run_directory.name, summary)
        log_epoch_metrics(mlflow_client, summary["history"])
        log_training_summary(mlflow_client, summary)
        evaluation_path = run_directory / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            log_evaluation_metrics(mlflow_client, evaluation)
        mlflow_client.log_artifacts(str(run_directory))