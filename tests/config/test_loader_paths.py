from __future__ import annotations

from contsg.config.loader import ConfigLoader


def test_loader_prefers_generators_model_config_path(tmp_path):
    config_dir = tmp_path / "configs"
    generators_dir = config_dir / "generators"
    generators_dir.mkdir(parents=True)

    (generators_dir / "my_model.yaml").write_text(
        "model:\n"
        "  channels: 123\n"
        "  layers: 5\n",
        encoding="utf-8",
    )

    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.from_args(dataset="debug", model="my_model")

    assert cfg.model.name == "my_model"
    assert cfg.model.channels == 123
    assert cfg.model.layers == 5


def test_loader_falls_back_to_legacy_models_path(tmp_path):
    config_dir = tmp_path / "configs"
    legacy_models_dir = config_dir / "models"
    legacy_models_dir.mkdir(parents=True)

    (legacy_models_dir / "legacy_model.yaml").write_text(
        "model:\n"
        "  channels: 77\n",
        encoding="utf-8",
    )

    loader = ConfigLoader(config_dir=config_dir)
    cfg = loader.from_args(dataset="debug", model="legacy_model")

    assert cfg.model.name == "legacy_model"
    assert cfg.model.channels == 77
