"""
Model configuration validation utilities.

This module implements a hybrid validation strategy:
- Built-in models: validated with model-specific schemas when available
- External models: minimally validated by base ModelConfig in relaxed mode
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Type

from pydantic import BaseModel, ValidationError

from contsg.config.schema import (
    BridgeModelConfig,
    CTTPModelConfig,
    DiffuSETSModelConfig,
    ExperimentConfig,
    ModelConfig,
    RetrievalModelConfig,
    T2SModelConfig,
    TEditModelConfig,
    TTSCGANModelConfig,
    Text2MotionModelConfig,
    TimeVQVAEModelConfig,
    TimeWeaverModelConfig,
    VerbalTSModelConfig,
    WaveStitchModelConfig,
)
from contsg.registry import Registry


BUILTIN_MODEL_CONFIG_CLASSES: dict[str, Type[BaseModel]] = {
    "verbalts": VerbalTSModelConfig,
    "bridge": BridgeModelConfig,
    "t2s": T2SModelConfig,
    "timevqvae": TimeVQVAEModelConfig,
    "timeweaver": TimeWeaverModelConfig,
    "wavestitch": WaveStitchModelConfig,
    "retrieval": RetrievalModelConfig,
    "tedit": TEditModelConfig,
    "diffusets": DiffuSETSModelConfig,
    "cttp": CTTPModelConfig,
    "text2motion": Text2MotionModelConfig,
    "ttscgan": TTSCGANModelConfig,
}

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class ModelValidationResult:
    """Structured result for model configuration validation."""

    config: ExperimentConfig
    strict_schema: bool
    resolved_model_name: str
    schema_class: Type[BaseModel] | None
    schema_source: str | None


def _parse_env_flag(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return None


def resolve_strict_schema(cli_flag: bool | None) -> bool:
    """
    Resolve strict schema mode from CLI flag and environment.

    Precedence:
    1. CLI flag (if specified)
    2. CONTSG_STRICT_SCHEMA environment variable
    3. Default False
    """
    if cli_flag is not None:
        return cli_flag

    env_flag = _parse_env_flag("CONTSG_STRICT_SCHEMA")
    return bool(env_flag)


def _get_model_schema(model_name: str) -> tuple[Type[BaseModel] | None, str | None]:
    """Resolve model-specific schema from registry first, then built-in fallback map."""
    config_cls = Registry.get_model_config_class(model_name)
    if config_cls is not None:
        return config_cls, "registry"

    builtin_cls = BUILTIN_MODEL_CONFIG_CLASSES.get(model_name)
    if builtin_cls is not None:
        return builtin_cls, "builtin"

    return None, None


def validate_model_config(
    config: ExperimentConfig,
    strict_schema: bool | None = None,
) -> ModelValidationResult:
    """
    Validate `config.model` with optional model-specific schema.

    In relaxed mode, models without schema use base ModelConfig validation.
    In strict mode, models without schema are rejected.
    """
    strict = resolve_strict_schema(strict_schema)

    resolved_name = Registry.resolve_model_name(config.model.name)
    model_payload = config.model.model_dump(mode="python")
    model_payload["name"] = resolved_name

    schema_class, schema_source = _get_model_schema(resolved_name)
    if schema_class is None:
        if strict:
            raise ValueError(
                "Strict schema validation is enabled, but no model schema is registered for "
                f"'{resolved_name}'. Register one via "
                "@Registry.register_model(..., config_class=YourModelConfig), "
                "or disable strict mode."
            )
        validated_model = ModelConfig.model_validate(model_payload)
    else:
        try:
            validated_model = schema_class.model_validate(model_payload)
        except AttributeError as exc:
            raise ValueError(
                f"Registered schema for '{resolved_name}' does not provide "
                "Pydantic model_validate()."
            ) from exc
        except ValidationError as exc:
            raise ValueError(
                f"Invalid model configuration for '{resolved_name}' "
                f"(schema source: {schema_source}).\n{exc}"
            ) from exc

    if not hasattr(validated_model, "name"):
        raise ValueError(f"Model schema for '{resolved_name}' must define a 'name' field.")

    object.__setattr__(config, "model", validated_model)
    return ModelValidationResult(
        config=config,
        strict_schema=strict,
        resolved_model_name=resolved_name,
        schema_class=schema_class,
        schema_source=schema_source,
    )
