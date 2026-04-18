#!/usr/bin/env python3
"""
Run `contsg evaluate` on PT-FG (or any) experiment folders with multiple seeds,
then aggregate metrics into a Markdown report (mean ± sample std over seeds).

Requires: `contsg evaluate` with `--seed` support (see contsg/cli.py).

Example:
  ./scripts/eval_ptfg_multiseed.py --config scripts/ptfg_multiseed_eval.yaml
  ./scripts/eval_ptfg_multiseed.py --experiment experiments/my_run --seeds 42,142,242
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class ExpEntry:
    rel_path: str
    label: str


def _load_yaml_config(path: Path) -> Tuple[List[ExpEntry], List[int]]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config (pip install pyyaml)")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in raw.get("seeds", [42, 142, 242])]
    entries: List[ExpEntry] = []
    for item in raw.get("experiments", []):
        if isinstance(item, str):
            entries.append(ExpEntry(rel_path=item, label=Path(item).name))
        else:
            entries.append(
                ExpEntry(
                    rel_path=item["rel_path"],
                    label=item.get("label", Path(item["rel_path"]).name),
                )
            )
    return entries, seeds


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run_eval(
    repo_root: Path,
    contsg: str,
    exp_dir: Path,
    seed: int,
    checkpoint: str,
    use_cache: bool,
    refresh_cache: bool,
    extra_args: List[str],
) -> int:
    out_name = f"eval_seed{seed}.json"
    cmd = [
        contsg,
        "evaluate",
        str(exp_dir.relative_to(repo_root)),
        "--checkpoint",
        checkpoint,
        "--seed",
        str(seed),
        "--output",
        out_name,
    ]
    if use_cache:
        cmd.append("--use-cache")
    if refresh_cache:
        cmd.append("--refresh-cache")
    cmd.extend(extra_args)
    print(f"[run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=repo_root)


def _load_metrics(repo_root: Path, exp_rel: str, seed: int) -> Dict[str, float]:
    p = repo_root / exp_rel / "results" / f"eval_seed{seed}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing results file: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    metrics = data.get("metrics") or {}
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out[k] = float(v)
    return out


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    xs = list(vals)
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def _fmt(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.6f}"


def _write_md(
    path: Path,
    repo_root: Path,
    entries: List[ExpEntry],
    seeds: List[int],
    per_run: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    lines: List[str] = []
    lines.append("# PT-FG multi-seed evaluation summary\n\n")
    lines.append(
        f"- Repo root: `{repo_root}`\n"
        f"- Seeds: {seeds}\n"
        f"- Per-seed JSON: `results/eval_seed<seed>.json`\n\n"
    )

    all_keys: List[str] = []
    for label_data in per_run.values():
        for sd in seeds:
            row = label_data.get(sd, {})
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
    all_keys.sort()

    for ent in entries:
        label = ent.label
        lines.append(f"## {label}\n\n")
        lines.append(f"*Experiment:* `{ent.rel_path}`\n\n")

        by_seed = per_run.get(label, {})
        if not by_seed:
            lines.append("*No results loaded.*\n\n")
            continue

        hdr = "| Metric | " + " | ".join(f"seed {s}" for s in seeds) + " | mean ± std |\n"
        sep = "| --- | " + " | ".join("---" for _ in seeds) + " | --- |\n"
        lines.append(hdr)
        lines.append(sep)

        for key in all_keys:
            vals = []
            cells = []
            for s in seeds:
                v = by_seed.get(s, {}).get(key)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(_fmt(v))
                    vals.append(v)
            if not vals:
                continue
            m, sdv = _mean_std(vals)
            if len(vals) == 1:
                agg = _fmt(m)
            else:
                agg = f"{_fmt(m)} ± {_fmt(sdv)}"
            lines.append(f"| {key} | " + " | ".join(cells) + f" | {agg} |\n")
        lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"[done] Wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-seed contsg evaluate + MD summary")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with experiments[] and seeds[] (see scripts/ptfg_multiseed_eval.yaml)",
    )
    ap.add_argument(
        "--experiment",
        type=str,
        action="append",
        default=None,
        help="Experiment dir relative to repo (repeatable). Ignored if --config is set.",
    )
    ap.add_argument(
        "--label",
        type=str,
        action="append",
        default=None,
        help="Display label for each --experiment (same order). Optional.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="42,142,242",
        help="Comma-separated seeds (default: 42,142,242)",
    )
    ap.add_argument("--checkpoint", default="best", help="contsg evaluate --checkpoint")
    ap.add_argument(
        "--contsg",
        default="contsg",
        help="contsg executable (default: contsg on PATH)",
    )
    ap.add_argument(
        "--use-cache",
        action="store_true",
        help="Pass --use-cache to contsg evaluate",
    )
    ap.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Pass --refresh-cache (recommended with --use-cache when varying seed)",
    )
    ap.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate existing eval_seed*.json files",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown output path (default: paper/ptfg_multiseed_eval_summary.md)",
    )
    ap.add_argument(
        "extra",
        nargs="*",
        help="Extra args forwarded to contsg evaluate (e.g. --metrics dtw,crps,acd)",
    )
    args = ap.parse_args()
    repo_root = args.repo_root.resolve()

    if args.config:
        entries, seeds = _load_yaml_config(args.config.resolve())
    else:
        if not args.experiment:
            ap.error("Provide --config or at least one --experiment")
        seeds = _parse_seeds(args.seeds)
        labels = args.label or []
        entries = []
        for i, rel in enumerate(args.experiment):
            lab = labels[i] if i < len(labels) else Path(rel).name
            entries.append(ExpEntry(rel_path=rel, label=lab))

    out_md = args.output_md or (repo_root / "paper" / "ptfg_multiseed_eval_summary.md")

    per_run: Dict[str, Dict[int, Dict[str, float]]] = {e.label: {} for e in entries}

    for ent in entries:
        exp_path = repo_root / ent.rel_path
        if not exp_path.is_dir():
            print(f"[warn] Missing directory, skip: {exp_path}", file=sys.stderr)
            continue
        for seed in seeds:
            if not args.skip_run:
                code = _run_eval(
                    repo_root,
                    args.contsg,
                    exp_path,
                    seed,
                    args.checkpoint,
                    args.use_cache,
                    args.refresh_cache,
                    list(args.extra),
                )
                if code != 0:
                    print(f"[error] evaluate failed for {ent.label} seed={seed}", file=sys.stderr)
                    return code
            try:
                per_run[ent.label][seed] = _load_metrics(repo_root, ent.rel_path, seed)
            except FileNotFoundError as e:
                print(f"[warn] {e}", file=sys.stderr)

    _write_md(out_md.resolve(), repo_root, entries, seeds, per_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
