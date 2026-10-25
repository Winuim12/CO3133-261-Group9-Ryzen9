#test_context.py

from experiments.context import prepare_experiment_context


def test_prepare_context_loads_project_relative_configs(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "config.yaml").write_text(
        "seed: 42\ndata:\n  num_classes: 30\n",
        encoding="utf-8",
    )
    (config_dir / "baseline.yaml").write_text(
        "model:\n  name: baseline_cnn\n",
        encoding="utf-8",
    )

    context = prepare_experiment_context(
        "config/baseline.yaml",
        project_root=tmp_path,
    )

    assert context.project_root == tmp_path.resolve()
    assert context.config["seed"] == 42
    assert context.config["data"]["num_classes"] == 30
    assert context.config["model"]["name"] == "baseline_cnn"
    assert context.device.type in {"cpu", "cuda"}