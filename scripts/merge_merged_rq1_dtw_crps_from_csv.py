#!/usr/bin/env python3
"""Fill baseline DTW/CRPS in paper/merged_rq1_statistical_and_ptfg.md from full_results.csv."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

SECTION_TO_DS = {
    "Synth-U": "synth-u",
    "Synth-M": "synth-m",
    "AirQuality Beijing": "airquality_beijing_intrinsic",
    "ETTm1": "ettm1_llm",
    "Istanbul Traffic": "istanbul_traffic_llm",
    "TelecomTS": "telecomts_segment",
    "Weather Conceptual": "weather_extrinsic",
    "Weather Morphological": "weather_intrinsic",
    "PTB-XL Conceptual": "ptb_extrinsic",
    "PTB-XL Morphological": "ptb_intrinsic",
}

MODEL_TO_CSV = {
    "Bridge": "bridge",
    "DiffuSETS": "diffusets",
    "T2S": "t2s",
    "TEdit": "tedit",
    "Text2Motion": "text2motion",
    "TimeVQVAE": "timevqvae",
    "TimeWeaver": "timeweaver",
    "TTSCGAN": "ttscgan",
    "VerbalTS": "verbalts",
    "WaveStitch": "wavestitch",
}


def ffloat(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def mu_sd_from_row(row: dict) -> tuple:
    mu, sd = ffloat(row.get("mean")), ffloat(row.get("std"))
    if mu is not None:
        return mu, sd
    vals = [ffloat(row.get(f"seed{i}")) for i in range(3)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, None
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def load_lookup(csv_path: Path) -> dict:
    d = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model"].strip(), row["dataset"].strip(), row["metric"].strip())
            d[key] = mu_sd_from_row(row)
    return d


def fmt_cell(mu, sd) -> str:
    if mu is None:
        return ""
    if sd is None or sd == 0.0:
        return f"{mu:.6f}"
    return f"{mu:.6f} ± {sd:.6f}"


def merge_md(md_path: Path, csv_path: Path) -> str:
    lookup = load_lookup(csv_path)
    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    current_section = None
    for line in lines:
        msec = re.match(r"^## (.+)\s*$", line)
        if msec:
            current_section = msec.group(1).strip()
            out.append(line)
            continue
        if current_section and line.startswith("|"):
            raw = line.strip()
            if raw.startswith("|") and raw.endswith("|"):
                cells = [c.strip() for c in raw[1:-1].split("|")]
                if len(cells) >= 6 and cells[0] not in ("Model", "---") and "---" not in cells[0]:
                    model = cells[0]
                    if not model.startswith("PT-FG"):
                        csv_model = MODEL_TO_CSV.get(model)
                        ds = SECTION_TO_DS.get(current_section)
                        if csv_model and ds:
                            mu_d, sd_d = lookup.get((csv_model, ds, "dtw"), (None, None))
                            mu_c, sd_c = lookup.get((csv_model, ds, "crps"), (None, None))
                            new_dtw = fmt_cell(mu_d, sd_d) if not cells[1] else cells[1]
                            new_crps = fmt_cell(mu_c, sd_c) if not cells[2] else cells[2]
                            line = "| " + " | ".join([model, new_dtw, new_crps] + cells[3:]) + " |\n"
        out.append(line)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    root = args.repo_root
    csv_path = args.csv or (root / "full_results.csv")
    md_path = args.md or (root / "paper" / "merged_rq1_statistical_and_ptfg.md")
    text = merge_md(md_path, csv_path)
    md_path.write_text(text, encoding="utf-8")
    print(f"Updated {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
