#test_config.py

from src.utils.config import load_config

def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "seed: 42\ndata:\n num_classes: 30\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config == {
        "seed": 42,
        "data": {"num_classes": 30},
    }