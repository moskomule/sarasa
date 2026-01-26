import dataclasses
import json
import pathlib
import sys

import pytest

from sarasa import Config


def test_config_custom_model_type():
    @dataclasses.dataclass
    class CustomModelConfig:
        param: int = 42

    cfg = Config.from_cli(model_type=CustomModelConfig)
    assert cfg.model.param == 42


def test_config_complex_custom_model_type(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "model:llama3", "--model.param", "100"])

    @dataclasses.dataclass
    class Llama3:
        param: int = 42

    @dataclasses.dataclass
    class Qwen3:
        param: str = "hello"

    cfg = Config.from_cli(model_type=Llama3 | Qwen3)
    assert isinstance(cfg.model, Llama3)
    assert cfg.model.param == 100


@pytest.fixture
def config_json() -> str:
    file = pathlib.Path("config.json")
    with open(file, "w") as f:
        json.dump({"checkpoint": {"save_freq": 10}}, f)
    yield str(file)
    file.unlink()


def test_config_overriding(config_json, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", config_json])
    cfg = Config.from_cli()
    assert cfg.checkpoint.save_freq == 10


def test_config_overriding_cli(config_json, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", config_json, "--checkpoint.save_freq", "20"])
    cfg = Config.from_cli()
    assert cfg.checkpoint.save_freq == 20
