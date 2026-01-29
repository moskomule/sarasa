import dataclasses
import sys

import pytest

from sarasa import Config


def test_config_custom_model_type(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program"])  # avoid checking actual sys.argv

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
def config_py(tmp_path, num_configs) -> str:
    file = tmp_path / "config.py"
    lines = ["from sarasa.config import *"]
    for i in range(num_configs):
        lines.append(
            f"config{i} = Config(Model(), AdamW(), LRScheduler(), Data(), checkpoint=Checkpoint(save_freq=10))"
        )
    with open(file, "w") as f:
        f.write("\n".join(lines))
    yield str(file)


@pytest.mark.parametrize("num_configs", [1])
def test_config_loading(config_py, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", config_py])
    cfg = Config.from_cli()
    assert cfg.checkpoint.save_freq == 10


@pytest.mark.parametrize("num_configs", [1])
def test_config_loading_overriding(config_py, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", config_py, "--checkpoint.save_freq", "100"])
    cfg = Config.from_cli()
    assert cfg.checkpoint.save_freq == 100


@pytest.mark.parametrize("num_configs", [0, 2])
def test_config_loading_content_error(config_py, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", config_py])
    with pytest.raises(ValueError):
        Config.from_cli()


def test_config_loading_filetype_error(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["program", "--config_file", str(config_file)])
    with pytest.raises(ValueError):
        Config.from_cli()
