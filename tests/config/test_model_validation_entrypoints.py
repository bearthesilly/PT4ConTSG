from __future__ import annotations

import pytest
import yaml

from contsg.eval.evaluator import Evaluator
from contsg.tracker import ExperimentTracker


def _write_invalid_verbalts_config(config_path, data_folder: str) -> None:
    payload = {
        "data": {
            "name": "debug",
            "data_folder": data_folder,
            "n_var": 2,
            "seq_length": 32,
        },
        "model": {
            "name": "verbalts",
            "condition_type": "not_supported",
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)


def test_tracker_from_experiment_dir_validates_model_schema(tmp_path):
    experiment_dir = tmp_path / "exp_tracker"
    experiment_dir.mkdir()
    _write_invalid_verbalts_config(experiment_dir / "config.yaml", data_folder=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid model configuration for 'verbalts'"):
        ExperimentTracker.from_experiment_dir(experiment_dir)


def test_evaluator_from_experiment_validates_model_schema(tmp_path):
    experiment_dir = tmp_path / "exp_eval"
    experiment_dir.mkdir()
    _write_invalid_verbalts_config(experiment_dir / "config.yaml", data_folder=str(tmp_path))

    with pytest.raises(ValueError, match="Invalid model configuration for 'verbalts'"):
        Evaluator.from_experiment(experiment_dir, cache_only=True)
