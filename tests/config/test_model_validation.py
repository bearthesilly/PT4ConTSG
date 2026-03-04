from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field

from contsg.config.model_validation import resolve_strict_schema, validate_model_config
from contsg.config.schema import ExperimentConfig, ModelConfig
from contsg.registry import Registry


@pytest.fixture(autouse=True)
def preserve_registry_state():
    original_models = Registry._models.copy()
    original_datasets = Registry._datasets.copy()
    original_metrics = Registry._metrics.copy()
    original_model_configs = Registry._model_configs.copy()
    original_model_aliases = Registry._model_aliases.copy()
    original_dataset_aliases = Registry._dataset_aliases.copy()

    yield

    Registry._models.clear()
    Registry._models.update(original_models)
    Registry._datasets.clear()
    Registry._datasets.update(original_datasets)
    Registry._metrics.clear()
    Registry._metrics.update(original_metrics)
    Registry._model_configs.clear()
    Registry._model_configs.update(original_model_configs)
    Registry._model_aliases.clear()
    Registry._model_aliases.update(original_model_aliases)
    Registry._dataset_aliases.clear()
    Registry._dataset_aliases.update(original_dataset_aliases)


def _make_config(tmp_path, model_dict: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "data": {
                "name": "debug",
                "data_folder": str(tmp_path),
                "n_var": 2,
                "seq_length": 32,
            },
            "model": model_dict,
        }
    )


def test_relaxed_mode_allows_external_model_without_schema(tmp_path):
    cfg = _make_config(
        tmp_path,
        {
            "name": "external_model",
            "some_custom_param": "value",
        },
    )

    result = validate_model_config(cfg, strict_schema=False)

    assert result.strict_schema is False
    assert result.schema_class is None
    assert result.config.model.name == "external_model"
    assert getattr(result.config.model, "some_custom_param") == "value"


def test_strict_mode_rejects_external_model_without_schema(tmp_path):
    cfg = _make_config(
        tmp_path,
        {
            "name": "external_model",
            "some_custom_param": "value",
        },
    )

    with pytest.raises(ValueError, match="Strict schema validation is enabled"):
        validate_model_config(cfg, strict_schema=True)


def test_registry_config_class_is_used_for_external_model(tmp_path):
    class ExternalModelConfig(ModelConfig):
        name: Literal["external_schema_model"] = "external_schema_model"
        hidden_dim: int = Field(..., ge=1)

    @Registry.register_model("external_schema_model", config_class=ExternalModelConfig)
    class ExternalSchemaModel:
        pass

    cfg = _make_config(
        tmp_path,
        {
            "name": "external_schema_model",
            "hidden_dim": 32,
        },
    )

    result = validate_model_config(cfg, strict_schema=True)

    assert result.schema_class is ExternalModelConfig
    assert result.schema_source == "registry"
    assert result.config.model.hidden_dim == 32


def test_builtin_schema_validation_still_applies_in_relaxed_mode(tmp_path):
    cfg = _make_config(
        tmp_path,
        {
            "name": "verbalts",
            "condition_type": "not_supported",
        },
    )

    with pytest.raises(ValueError, match="Invalid model configuration for 'verbalts'"):
        validate_model_config(cfg, strict_schema=False)


def test_resolve_strict_schema_with_env_and_cli_precedence(monkeypatch):
    monkeypatch.setenv("CONTSG_STRICT_SCHEMA", "1")
    assert resolve_strict_schema(None) is True
    assert resolve_strict_schema(False) is False

    monkeypatch.setenv("CONTSG_STRICT_SCHEMA", "off")
    assert resolve_strict_schema(None) is False
