#!/usr/bin/env python3
"""Parse grid search logs for PTFG V2 on synth-m and rank combos."""
import json
import re
import sys
from pathlib import Path

DATASET = "synth-m"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _PROJECT_ROOT / "log" / "grid_search"

# Rich/Lightning metric tables: Unicode │ (U+2502) or ASCII |
_CELL = r"[\|\u2502]"
ROW_METRIC_RE = re.compile(
    rf"{_CELL}\s+([\w.]+)\s+{_CELL}\s+([\d.eE+-]+)\s+{_CELL}"
)

RANK_METRICS = {
    "dtw": "lower",
    "crps": "lower",
    "acd": "lower",
    "sd": "lower",
    "kd": "lower",
    "mdd": "lower",
    "fid": "lower",
    "jftsd": "lower",
    "prdc_f1.f1": "higher",
    "joint_prdc_f1.f1": "higher",
}


def parse_log(log_path):
    metrics = {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.split("\n"):
        line = line.strip()
        m = ROW_METRIC_RE.search(line)
        if m:
            try:
                metrics[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return metrics


def main():
    logs = sorted(LOG_DIR.glob("ptfg_synth-m_*.log"))
    if not logs:
        print("No logs found in {} for {}".format(LOG_DIR, DATASET))
        sys.exit(1)

    results = []
    for log_path in logs:
        label = log_path.stem.replace("ptfg_{}_".format(DATASET), "")
        metrics = parse_log(log_path)
        if metrics:
            results.append({"label": label, "metrics": metrics})
        else:
            print("  WARNING: no metrics from {}".format(log_path.name))

    if not results:
        print("No results parsed!")
        sys.exit(1)

    # Compute per-metric ranks
    for metric_name, direction in RANK_METRICS.items():
        default = float("inf") if direction == "lower" else 0.0
        vals = [(i, r["metrics"].get(metric_name, default)) for i, r in enumerate(results)]
        vals.sort(key=lambda x: x[1], reverse=(direction == "higher"))
        for rank, (idx, _) in enumerate(vals):
            results[idx].setdefault("ranks", {})[metric_name] = rank + 1

    for r in results:
        ranks = r.get("ranks", {})
        r["avg_rank"] = sum(ranks.values()) / max(len(ranks), 1)

    results.sort(key=lambda r: r["avg_rank"])

    # Print table
    print()
    print("=" * 120)
    print("Grid Search Results: PTFG V2 on {}  ({} combos)".format(DATASET, len(results)))
    print("=" * 120)
    header = "{:<5} {:<30} {:<7} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9}".format(
        "Rank", "Label", "AvgRk", "DTW", "CRPS", "ACD", "FID", "KD", "PRDC", "J-PRDC")
    print(header)
    print("-" * 120)
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        print("{:<5} {:<30} {:<7.1f} {:<9.3f} {:<9.4f} {:<9.4f} {:<9.2f} {:<9.4f} {:<9.4f} {:<9.4f}".format(
            i, r["label"], r["avg_rank"],
            m.get("dtw", -1), m.get("crps", -1), m.get("acd", -1),
            m.get("fid", -1), m.get("kd", -1),
            m.get("prdc_f1.f1", -1), m.get("joint_prdc_f1.f1", -1)))

    print()
    print("Best: {}".format(results[0]["label"]))
    print("Metrics: {}".format(json.dumps(results[0]["metrics"], indent=2)))


if __name__ == "__main__":
    main()
