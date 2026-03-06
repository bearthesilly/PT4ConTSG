from __future__ import annotations

from pathlib import Path


def _read_hf_space_app_source() -> str:
    app_path = Path(__file__).resolve().parents[1] / "hf_space" / "app.py"
    return app_path.read_text(encoding="utf-8")


def test_header_links_markdown_exposes_primary_project_links():
    source = _read_hf_space_app_source()

    assert "HEADER_LINKS_MARKDOWN" in source
    assert "https://github.com/seqml/ConTSG-Bench" in source
    assert "https://huggingface.co/spaces/mldi-lab/ConTSG-Bench-Leaderboard" in source
    assert "https://arxiv.org/abs/2603.04767" in source
