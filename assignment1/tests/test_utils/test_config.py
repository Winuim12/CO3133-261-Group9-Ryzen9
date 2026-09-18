#test_config.py
"""
RUN: python -m pytest tests/utils/test_config.py -v
"""

from src.utils.config import load_config

def test_load_config_returns_yaml_values(tmp_path):
    config_path = tmp_path / "test_config.yaml"

    config_path.write_text(
        """
        seed: 42
        training: 
            epochs: 20
            learning_rate: 0.001
            optimizer: adamw
        """,
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["seed"] == 42
    assert config["training"]["epochs"] == 20
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["optimizer"] == "adamw"
