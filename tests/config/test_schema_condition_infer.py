from __future__ import annotations

import json

import numpy as np

from contsg.config.schema import ExperimentConfig


def _make_base_payload(data_folder: str, model_name: str = "external_model") -> dict:
    return {
        "data": {
            "name": "debug",
            "data_folder": data_folder,
            "n_var": 2,
            "seq_length": 32,
        },
        "model": {"name": model_name},
    }


def test_attribute_inference_for_custom_model_when_attribute_enabled(tmp_path):
    dataset_dir = tmp_path / "dataset_attr"
    dataset_dir.mkdir(parents=True)

    with open(dataset_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"attr_n_ops": [2, 2]}, f)

    # Second attribute contains unknown value -1, so inferred classes become 2 + 1 = 3.
    np.save(dataset_dir / "train_attrs_idx.npy", np.array([[0, 1], [1, -1]], dtype=np.int64))

    payload = _make_base_payload(str(dataset_dir), model_name="my_plugin_model")
    payload["condition"] = {"attribute": {"enabled": True}}

    cfg = ExperimentConfig.model_validate(payload)

    assert cfg.condition.attribute.enabled is True
    assert cfg.condition.attribute.discrete_configs == [{"num_classes": 2}, {"num_classes": 3}]


def test_label_inference_for_custom_model_when_label_enabled(tmp_path):
    dataset_dir = tmp_path / "dataset_label"
    dataset_dir.mkdir(parents=True)
    np.save(dataset_dir / "train_labels.npy", np.array([0, 2, 1, 2], dtype=np.int64))

    payload = _make_base_payload(str(dataset_dir), model_name="my_plugin_model")
    payload["condition"] = {"label": {"enabled": True}}

    cfg = ExperimentConfig.model_validate(payload)

    assert cfg.condition.label.enabled is True
    assert cfg.condition.label.num_classes == 3


def test_label_inference_prefers_labels_over_attr_fallback(tmp_path):
    dataset_dir = tmp_path / "dataset_label_priority"
    dataset_dir.mkdir(parents=True)

    with open(dataset_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"attr_n_ops": [2, 2]}, f)

    # Deliberately malformed attrs shape (3 columns) to ensure attr fallback is not touched.
    np.save(dataset_dir / "train_attrs_idx.npy", np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64))
    np.save(dataset_dir / "train_labels.npy", np.array([0, 1], dtype=np.int64))

    payload = _make_base_payload(str(dataset_dir), model_name="my_plugin_model")
    payload["condition"] = {
        "attribute": {"enabled": False},
        "label": {"enabled": True},
    }

    cfg = ExperimentConfig.model_validate(payload)

    assert cfg.condition.label.enabled is True
    assert cfg.condition.label.num_classes == 2
    assert cfg.condition.attribute.enabled is False
    assert cfg.condition.attribute.discrete_configs == []
