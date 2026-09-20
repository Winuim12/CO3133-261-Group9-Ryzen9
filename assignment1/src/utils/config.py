from pathlib import Path

import yaml

def load_config(path):
    config_path = Path(path)

    with config_path.open(mode="r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)