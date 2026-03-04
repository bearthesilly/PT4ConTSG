"""
ConTSG Leaderboard utilities.

This subpackage provides tools for generating, validating, and serving
the ConTSG-Bench leaderboard data:

- dataset_meta / model_meta: Metadata registries for datasets and models
- metric_catalog: Canonical metric definitions (name, group, direction, weight)
- schema: Pydantic models for snapshot file validation
- aggregate: Pipeline to convert eval_results.json → parquet snapshots
- validate: Schema and value validation for generated snapshots
- mock: Mock data generator for development and testing
"""
