from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CTTP_DIR = REPO_ROOT / "configs" / "cttp"
GENERATOR_DIR = REPO_ROOT / "configs" / "generators"


def test_required_cttp_configs_exist_and_legacy_names_absent():
    required = {
        "cttp_ettm1.yaml",
        "cttp_istanbul_traffic.yaml",
        "cttp_airquality_beijing.yaml",
        "cttp_synth-m.yaml",
        "cttp_synth-u.yaml",
        "cttp_weather_concept.yaml",
        "cttp_weather_morphology.yaml",
        "cttp_ptbxl_concept.yaml",
        "cttp_ptbxl_morphology.yaml",
        "cttp_telecomts_instance.yaml",
        "cttp_telecomts_segment.yaml",
    }

    existing = {path.name for path in CTTP_DIR.glob("cttp_*.yaml")}

    assert required.issubset(existing)
    assert "cttp_ettm1_instance.yaml" not in existing
    assert "cttp_istanbul_traffic_instance.yaml" not in existing


def test_cttp_loss_type_policy_by_dataset():
    for path in CTTP_DIR.glob("cttp_*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        loss_type = cfg["model"]["loss_type"]

        if path.name.startswith("cttp_ptbxl_"):
            assert loss_type == "supcon", f"{path.name} must use supcon"
        else:
            assert loss_type == "ce", f"{path.name} must use ce"


def test_generator_configs_do_not_reference_legacy_cttp_names():
    legacy_refs = {
        "cttp_ettm1_instance.yaml",
        "cttp_istanbul_traffic_instance.yaml",
    }

    for path in GENERATOR_DIR.glob("*.yaml"):
        content = path.read_text(encoding="utf-8")
        for legacy in legacy_refs:
            assert legacy not in content, f"{path.name} references {legacy}"
