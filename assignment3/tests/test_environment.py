"""Exercise the same CPU smoke command used by local setup, Docker, and CI."""

import json

import pytest

from experiments.check_environment import main


def test_cpu_tensor_and_torchvision_operator(capsys):
    assert main(["--device", "cpu", "--smoke-test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["selected_device"] == "cpu"
    assert report["smoke_test"] == "passed"


def test_missing_config_fails_before_running_smoke_check():
    with pytest.raises(FileNotFoundError):
        main(["--config", "config/not-present.yaml"])
