"""
Registry pattern implementation for ConTSG.

This module provides decorators for registering models, datasets, and metrics,
enabling easy extension without modifying multiple configuration files.

Usage:
    @Registry.register_model("my_model")
    class MyModelModule(BaseGeneratorModule):
        pass

    @Registry.register_dataset("my_dataset")
    class MyDataset(BaseDataModule):
        pass

    @Registry.register_metric("my_metric")
    class MyMetric(BaseMetric):
        pass
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


class RegistryError(Exception):
    """Exception raised for registry-related errors."""

    pass


class Registry:
    """
    Unified registry for models, datasets, and metrics.

    This class implements a centralized registry pattern that allows
    components to be registered via decorators and retrieved by name.
    """

    _models: Dict[str, Type] = {}
    _datasets: Dict[str, Type] = {}
    _metrics: Dict[str, Type] = {}
    _model_configs: Dict[str, Type] = {}  # Optional: model-specific config classes

    # Aliases for common variations
    _model_aliases: Dict[str, str] = {
        "tw": "timeweaver",
        "vts": "verbalts",
    }
    _dataset_aliases: Dict[str, str] = {}

    # ==========================================================================
    # Model Registration
    # ==========================================================================

    @classmethod
    def register_model(
        cls,
        name: str,
        aliases: Optional[List[str]] = None,
        config_class: Optional[Type] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a model class.

        Args:
            name: Primary name for the model
            aliases: Optional list of alternative names
            config_class: Optional Pydantic config class for this model

        Returns:
            Decorator function

        Example:
            @Registry.register_model("verbalts", aliases=["vts"])
            class VerbalTSModule(BaseGeneratorModule):
                pass
        """

        def decorator(model_cls: Type[T]) -> Type[T]:
            if name in cls._models:
                raise RegistryError(
                    f"Model '{name}' is already registered by {cls._models[name].__name__}"
                )

            cls._models[name] = model_cls

            # Register aliases
            if aliases:
                for alias in aliases:
                    cls._model_aliases[alias] = name

            # Register config class if provided
            if config_class:
                cls._model_configs[name] = config_class

            # Store metadata on the class itself
            model_cls._registry_name = name  # type: ignore
            model_cls._registry_aliases = aliases or []  # type: ignore

            return model_cls

        return decorator

    @classmethod
    def resolve_model_name(cls, name: str) -> str:
        """Resolve a model alias to its canonical name."""
        return cls._model_aliases.get(name, name)

    @classmethod
    def get_model_config_class(cls, name: str) -> Optional[Type]:
        """Get the registered config class for a model name or alias."""
        resolved_name = cls.resolve_model_name(name)
        return cls._model_configs.get(resolved_name)

    @classmethod
    def get_model(cls, name: str) -> Type:
        """
        Get a registered model class by name.

        Args:
            name: Model name or alias

        Returns:
            Model class

        Raises:
            RegistryError: If model is not found
        """
        # Resolve alias
        resolved_name = cls.resolve_model_name(name)

        if resolved_name not in cls._models:
            available = cls.list_models()
            raise RegistryError(
                f"Unknown model: '{name}'. Available models: {available}"
            )

        return cls._models[resolved_name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return sorted(cls._models.keys())

    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered model.

        Returns:
            Dict with 'class', 'name', 'aliases', 'docstring', 'config_class'
        """
        model_cls = cls.get_model(name)
        resolved_name = cls.resolve_model_name(name)

        return {
            "class": model_cls,
            "name": resolved_name,
            "aliases": getattr(model_cls, "_registry_aliases", []),
            "docstring": model_cls.__doc__ or "",
            "config_class": cls.get_model_config_class(resolved_name),
        }

    # ==========================================================================
    # Dataset Registration
    # ==========================================================================

    @classmethod
    def register_dataset(
        cls,
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a dataset class.

        Args:
            name: Primary name for the dataset
            aliases: Optional list of alternative names

        Returns:
            Decorator function

        Example:
            @Registry.register_dataset("synth-m")
            class SynthMDataModule(BaseDataModule):
                pass
        """

        def decorator(dataset_cls: Type[T]) -> Type[T]:
            if name in cls._datasets:
                raise RegistryError(
                    f"Dataset '{name}' is already registered by {cls._datasets[name].__name__}"
                )

            cls._datasets[name] = dataset_cls

            if aliases:
                for alias in aliases:
                    cls._dataset_aliases[alias] = name

            dataset_cls._registry_name = name  # type: ignore
            dataset_cls._registry_aliases = aliases or []  # type: ignore

            return dataset_cls

        return decorator

    @classmethod
    def get_dataset(cls, name: str) -> Type:
        """
        Get a registered dataset class by name.

        Args:
            name: Dataset name or alias

        Returns:
            Dataset class

        Raises:
            RegistryError: If dataset is not found
        """
        resolved_name = cls._dataset_aliases.get(name, name)

        if resolved_name not in cls._datasets:
            available = cls.list_datasets()
            raise RegistryError(
                f"Unknown dataset: '{name}'. Available datasets: {available}"
            )

        return cls._datasets[resolved_name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return sorted(cls._datasets.keys())

    # ==========================================================================
    # Metric Registration
    # ==========================================================================

    @classmethod
    def register_metric(
        cls,
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a metric class.

        Args:
            name: Primary name for the metric
            aliases: Optional list of alternative names

        Returns:
            Decorator function

        Example:
            @Registry.register_metric("dtw")
            class DTWMetric(BaseMetric):
                pass
        """

        def decorator(metric_cls: Type[T]) -> Type[T]:
            if name in cls._metrics:
                raise RegistryError(
                    f"Metric '{name}' is already registered by {cls._metrics[name].__name__}"
                )

            cls._metrics[name] = metric_cls
            metric_cls._registry_name = name  # type: ignore

            return metric_cls

        return decorator

    @classmethod
    def get_metric(cls, name: str) -> Type:
        """
        Get a registered metric class by name.

        Args:
            name: Metric name

        Returns:
            Metric class

        Raises:
            RegistryError: If metric is not found
        """
        if name not in cls._metrics:
            available = cls.list_metrics()
            raise RegistryError(
                f"Unknown metric: '{name}'. Available metrics: {available}"
            )

        return cls._metrics[name]

    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all registered metric names."""
        return sorted(cls._metrics.keys())

    # ==========================================================================
    # Auto-discovery
    # ==========================================================================

    @classmethod
    def auto_discover(cls, package_name: str = "contsg") -> None:
        """
        Auto-discover and import all modules to trigger registration.

        This method imports all submodules in the specified package,
        which triggers the @register decorators.

        Args:
            package_name: Package name to discover (default: "contsg")
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return

        # Import models
        try:
            models_pkg = importlib.import_module(f"{package_name}.models")
            cls._import_submodules(models_pkg)
        except ImportError:
            pass

        # Import datasets
        try:
            datasets_pkg = importlib.import_module(f"{package_name}.data.datasets")
            cls._import_submodules(datasets_pkg)
        except ImportError:
            pass

        # Import metrics
        try:
            metrics_pkg = importlib.import_module(f"{package_name}.eval.metrics")
            cls._import_submodules(metrics_pkg)
        except ImportError:
            pass

    @classmethod
    def _import_submodules(cls, package: Any) -> None:
        """Import all submodules of a package."""
        if not hasattr(package, "__path__"):
            return

        for _, name, _ in pkgutil.iter_modules(package.__path__):
            try:
                importlib.import_module(f"{package.__name__}.{name}")
            except Exception as e:
                # Log but don't fail on optional dependency errors
                print(f"Warning: Failed to import {package.__name__}.{name}: {e}")

    # ==========================================================================
    # Utilities
    # ==========================================================================

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._models.clear()
        cls._datasets.clear()
        cls._metrics.clear()
        cls._model_configs.clear()
        cls._model_aliases.clear()
        cls._dataset_aliases.clear()

    @classmethod
    def summary(cls) -> str:
        """Get a summary of all registered components."""
        lines = [
            "Registry Summary",
            "=" * 40,
            f"Models ({len(cls._models)}): {', '.join(cls.list_models())}",
            f"Datasets ({len(cls._datasets)}): {', '.join(cls.list_datasets())}",
            f"Metrics ({len(cls._metrics)}): {', '.join(cls.list_metrics())}",
        ]
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

# These are shortcuts to Registry methods for cleaner import statements
register_model = Registry.register_model
register_dataset = Registry.register_dataset
register_metric = Registry.register_metric

get_model = Registry.get_model
get_dataset = Registry.get_dataset
get_metric = Registry.get_metric

list_models = Registry.list_models
list_datasets = Registry.list_datasets
list_metrics = Registry.list_metrics
