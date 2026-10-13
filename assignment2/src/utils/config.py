#config.py

from pathlib import Path

import yaml

def load_config(path):
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")

    return config