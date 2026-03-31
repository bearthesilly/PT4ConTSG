#!/usr/bin/env python3
"""
Grid search for PTFactorGenerator V2 on ConTSG-Bench.

Search space (matching ConTSG paper + self_cond ablation):
  lr:         [1e-3, 1e-4]
  batch_size: [32, 64, 128, 256]
  self_cond:  [true, false]
  Total: 2 x 4 x 2 = 16 combinations per dataset

Usage:
    python scripts/grid_search_ptfg.py --dataset synth-u
    python scripts/grid_search_ptfg.py --dataset synth-m
    python scripts/grid_search_ptfg.py --dataset synth-u --dataset synth-m

Output:
    configs/grid_search/ptfg_<dataset>_<label>.yaml
    scripts/run_grid_<dataset>.sh
    scripts/summarize_grid_<dataset>.py
"""
import argparse
import itertools
import yaml
from pathlib import Path
from copy import deepcopy


# ============================================================
# Grid search space (ConTSG paper + self_cond)
# ============================================================

GRID = {
    "lr":         [1e-3, 1e-4],
    "batch_size": [32, 64, 128, 256],
    "self_cond":  [True, False],
}

SEED = 42

BASE_CONFIGS = {
    "synth-u": "configs/generators/ptfg_synth-u.yaml",
    "synth-m": "configs/generators/ptfg_synth-m.yaml",
}


def load_base_config(dataset):
    with open(BASE_CONFIGS[dataset], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def combo_label(combo):
    lr_s = "lr1e-3" if combo["lr"] == 1e-3 else "lr1e-4"
    return "{}_bs{}_{}" .format(lr_s, combo["batch_size"],
                                "scT" if combo["self_cond"] else "scF")


def apply_combo(base, combo):
    cfg = deepcopy(base)
    cfg["train"]["lr"] = combo["lr"]
    cfg["train"]["batch_size"] = combo["batch_size"]
    for stage in cfg["train"].get("stages", []):
        stage["lr"] = combo["lr"]
    cfg["model"]["self_cond"] = combo["self_cond"]
    cfg["seed"] = SEED
    return cfg


def generate_configs(dataset, output_dir):
    base = load_base_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    results = []
    for combo in combos:
        label = combo_label(combo)
        filepath = output_dir / "ptfg_{}_{}.yaml".format(dataset, label)
        cfg = apply_combo(base, combo)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        results.append((str(filepath), label, combo))

    return results


def generate_shell_script(dataset, configs, script_dir):
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "run_grid_{}.sh".format(dataset)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Grid search for PTFG V2 on {}".format(dataset),
        "# Search space: lr=[1e-3,1e-4] x batch_size=[32,64,128,256] x self_cond=[T,F]",
        "# Total: {} combinations".format(len(configs)),
        "",
        'LOGDIR="log/grid_search"',
        "mkdir -p $LOGDIR",
        "",
    ]

    for i, (config_path, label, combo) in enumerate(configs, 1):
        log_file = "$LOGDIR/ptfg_{}_{}.log".format(dataset, label)
        lines.append("# [{}/{}] {}".format(i, len(configs), label))
        lines.append('echo "[{}/{}] Running: {}"'.format(i, len(configs), label))
        lines.append("contsg train --config {} 2>&1 | tee {}".format(config_path, log_file))
        lines.append("")

    lines.append('echo "Grid search complete! {} experiments finished."'.format(len(configs)))
    lines.append('echo "Logs in: $LOGDIR"')

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print("  Shell script: {}".format(script_path))


def generate_summary_script(dataset, configs, script_dir):
    script_path = script_dir / "summarize_grid_{}.py".format(dataset)

    # Write the summary script as a standalone file
    content = [
        '#!/usr/bin/env python3',
        '"""Parse grid search logs for PTFG V2 on {} and rank combos."""'.format(dataset),
        'import json',
        'import re',
        'import sys',
        'from pathlib import Path',
        '',
        'DATASET = "{}"'.format(dataset),
        'LOG_DIR = Path("log/grid_search")',
        '',
        'RANK_METRICS = {',
        '    "dtw": "lower",',
        '    "crps": "lower",',
        '    "acd": "lower",',
        '    "sd": "lower",',
        '    "kd": "lower",',
        '    "mdd": "lower",',
        '    "fid": "lower",',
        '    "jftsd": "lower",',
        '    "prdc_f1.f1": "higher",',
        '    "joint_prdc_f1.f1": "higher",',
        '}',
        '',
        '',
        'def parse_log(log_path):',
        '    metrics = {}',
        '    text = log_path.read_text(encoding="utf-8", errors="ignore")',
        '    for line in text.split("\\n"):',
        '        line = line.strip()',
        '        m = re.match(r".*\\|\\s+([\\w.]+)\\s+\\|\\s+([\\d.]+)\\s+\\|", line)',
        '        if m:',
        '            try:',
        '                metrics[m.group(1)] = float(m.group(2))',
        '            except ValueError:',
        '                pass',
        '    return metrics',
        '',
        '',
        'def main():',
        '    logs = sorted(LOG_DIR.glob("ptfg_{}_*.log"))'.format(dataset),
        '    if not logs:',
        '        print("No logs found in {} for {}".format(LOG_DIR, DATASET))',
        '        sys.exit(1)',
        '',
        '    results = []',
        '    for log_path in logs:',
        '        label = log_path.stem.replace("ptfg_{}_".format(DATASET), "")',
        '        metrics = parse_log(log_path)',
        '        if metrics:',
        '            results.append({"label": label, "metrics": metrics})',
        '        else:',
        '            print("  WARNING: no metrics from {}".format(log_path.name))',
        '',
        '    if not results:',
        '        print("No results parsed!")',
        '        sys.exit(1)',
        '',
        '    # Compute per-metric ranks',
        '    for metric_name, direction in RANK_METRICS.items():',
        '        default = float("inf") if direction == "lower" else 0.0',
        '        vals = [(i, r["metrics"].get(metric_name, default)) for i, r in enumerate(results)]',
        '        vals.sort(key=lambda x: x[1], reverse=(direction == "higher"))',
        '        for rank, (idx, _) in enumerate(vals):',
        '            results[idx].setdefault("ranks", {})[metric_name] = rank + 1',
        '',
        '    for r in results:',
        '        ranks = r.get("ranks", {})',
        '        r["avg_rank"] = sum(ranks.values()) / max(len(ranks), 1)',
        '',
        '    results.sort(key=lambda r: r["avg_rank"])',
        '',
        '    # Print table',
        '    print()',
        '    print("=" * 120)',
        '    print("Grid Search Results: PTFG V2 on {}  ({} combos)".format(DATASET, len(results)))',
        '    print("=" * 120)',
        '    header = "{:<5} {:<30} {:<7} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9} {:<9}".format(',
        '        "Rank", "Label", "AvgRk", "DTW", "CRPS", "ACD", "FID", "KD", "PRDC", "J-PRDC")',
        '    print(header)',
        '    print("-" * 120)',
        '    for i, r in enumerate(results, 1):',
        '        m = r["metrics"]',
        '        print("{:<5} {:<30} {:<7.1f} {:<9.3f} {:<9.4f} {:<9.4f} {:<9.2f} {:<9.4f} {:<9.4f} {:<9.4f}".format(',
        '            i, r["label"], r["avg_rank"],',
        '            m.get("dtw", -1), m.get("crps", -1), m.get("acd", -1),',
        '            m.get("fid", -1), m.get("kd", -1),',
        '            m.get("prdc_f1.f1", -1), m.get("joint_prdc_f1.f1", -1)))',
        '',
        '    print()',
        '    print("Best: {}".format(results[0]["label"]))',
        '    print("Metrics: {}".format(json.dumps(results[0]["metrics"], indent=2)))',
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
    ]

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(content) + "\n")

    print("  Summary script: {}".format(script_path))


def main():
    parser = argparse.ArgumentParser(description="Grid search config generator for PTFG V2")
    parser.add_argument("--dataset", "-d", action="append",
                        choices=list(BASE_CONFIGS.keys()), required=True)
    args = parser.parse_args()

    total = 1
    for v in GRID.values():
        total *= len(v)

    print("Search space: {} combinations per dataset".format(total))
    for k, v in GRID.items():
        print("  {}: {}".format(k, v))
    print()

    config_dir = Path("configs/grid_search")
    script_dir = Path("scripts")

    for dataset in args.dataset:
        print("[{}]".format(dataset))
        configs = generate_configs(dataset, config_dir)
        print("  {} configs -> {}/".format(len(configs), config_dir))
        generate_shell_script(dataset, configs, script_dir)
        generate_summary_script(dataset, configs, script_dir)
        print()

    print("Next steps:")
    print("  1. bash scripts/run_grid_<dataset>.sh")
    print("  2. python scripts/summarize_grid_<dataset>.py")


if __name__ == "__main__":
    main()
