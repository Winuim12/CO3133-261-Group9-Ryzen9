#config.py

from pathlib import Path
from copy import deepcopy

import yaml

def load_config(path):
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")

    return config

def merge_configs(base, overrides):
    merged = deepcopy(base)

    for key, value in overrides.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged

def load_experiment_config(shared_path, model_path, experiment_path=None):
    config = merge_configs(
        load_config(shared_path),
        load_config(model_path),
    )

    if experiment_path is not None:
        config = merge_configs(
            config,
            load_config(experiment_path),
        )

    return config