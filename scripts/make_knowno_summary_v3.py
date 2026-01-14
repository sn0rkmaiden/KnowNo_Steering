#!/usr/bin/env python3
"""
scripts/make_knowno_summary.py

Create a compact LaTeX table from a mirrored `metrics/` directory.

Output columns (always):
Model | Vocab | Setting | Base | Str1 | Str3 | Str5 | Str10

Key behaviors (based on your metrics folder structure):
- NO separate baseline row.
- Baseline is selected per (model_size, setting) so default baselines do NOT mix with
  repeated-prompt baselines.
- Rows are created from *steering* metric files only, grouped by (model_size, vocab, setting).
  For each row:
    Base = baseline value for (model_size, setting) if present, else '--'
    StrK = average metric over all feature files matching that row and steering strength K

Settings mapping (to match your filenames):
- Steering files: "...._prrepeat2.json" -> Setting = "repeat2_both" (canonical)
- Baseline files:
    results_baseline.json           -> Setting = "default"
    repeat2_both_baseline.json      -> Setting = "repeat2_both"
    repeat2_clarify_baseline.json   -> Setting = "repeat2_clarify"
    repeat3_both_baseline.json      -> Setting = "repeat3_both"
  (You can extend patterns in parse_baseline_setting if you add more.)

Flags:
- --include_repeats : include non-default settings (repeat*) rows.
  Without this flag, we ignore non-default steering files and non-default baselines.

Examples:
  python scripts/make_knowno_summary.py --metrics_dir metrics --out_tex table.tex

  python scripts/make_knowno_summary.py --metrics_dir metrics --out_tex table.tex --include_repeats
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def latex_escape(s: str) -> str:
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in s)


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_metric_value(data: dict, metric_name: str) -> Optional[float]:
    if metric_name in data:
        return safe_float(data[metric_name])
    m = data.get("metrics")
    if isinstance(m, dict) and metric_name in m:
        return safe_float(m[metric_name])
    return None


def is_baseline_file(fname: str) -> bool:
    f = fname.lower()
    return ("baseline" in f) or ("results_baseline" in f) or f.startswith("baseline")


def parse_strength(fname: str) -> Optional[int]:
    m = re.search(r"steering[_-]str([0-9]+)", fname.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_-])str(?:ength)?\s*([0-9]+)", fname.lower())
    if m:
        return int(m.group(1))
    return None


def infer_model_size_from_folders(folders: List[str]) -> Optional[str]:
    joined = " ".join(folders).lower()
    if "2b" in joined:
        return "2B"
    if "9b" in joined:
        return "9B"
    return None


def infer_vocab_from_folders(folders: List[str]) -> Optional[str]:
    low = [f.lower() for f in folders]
    if any("clar_vocab" in f for f in low):
        return "C"
    if any("question_vocab" in f for f in low) or any("quest_vocab" in f for f in low):
        return "Q"
    return None


def parse_steering_setting(fname: str) -> str:
    f = fname.lower()
    if "prrepeat2" in f:
        return "repeat2_both"  # canonical setting name to match baseline file
    # If you later add prrepeat3 etc, extend here.

    # For filenames like "..._repeat2_both.json" or "..._repeat2_clarify.json"
    m = re.search(r"(prrepeat[^.]+|repeat[^.]+)", f)
    if m:
        # strip trailing extension just in case
        return m.group(1).replace(".json", "").rstrip("_-")
    return "default"


def parse_baseline_setting(fname: str) -> str:
    f = fname.lower()
    if "results_baseline" in f or f == "results_baseline.json":
        return "default"
    # repeat2_both_baseline.json -> repeat2_both
    m = re.search(r"(repeat\d+_(?:both|clarify))_baseline", f)
    if m:
        return m.group(1)
    # repeat3_both_baseline.json -> repeat3_both
    m = re.search(r"(repeat\d+_both)_baseline", f)
    if m:
        return m.group(1)
    # fallback: treat as default if nothing matches
    return "default"


@dataclass(frozen=True)
class RowKey:
    model: str    # 2B / 9B
    vocab: str    # C / Q
    setting: str  # default / repeat2_both / ...


def fmt(x: Optional[float], decimals: int = 3, pct: bool = False) -> str:
    if x is None:
        return "--"
    if pct:
        x = x * 100.0
    return f"{x:.{decimals}f}"


def aggregate(
    metrics_dir: Path,
    metric_name: str,
    strengths: List[int],
    include_repeats: bool,
) -> Tuple[Dict[Tuple[str, str], float], Dict[RowKey, Dict[int, Optional[float]]]]:
    """
    Returns:
      baseline_by_key: {(model_size, setting) -> baseline_value}
      scores: RowKey -> {strength -> avg(metric)}
    """
    files = [p for p in metrics_dir.rglob("*.json") if p.is_file()]

    baseline_vals: Dict[Tuple[str, str], List[float]] = {}
    strength_vals: Dict[RowKey, Dict[int, List[float]]] = {}

    for p in files:
        rel = p.relative_to(metrics_dir)
        folders = list(rel.parts[:-1])
        fname = rel.name

        model = infer_model_size_from_folders(folders)
        if model is None:
            continue

        data = load_json(p)
        if data is None:
            continue

        val = get_metric_value(data, metric_name)
        if val is None:
            continue

        # Baselines: key by (model, baseline_setting)
        if is_baseline_file(fname):
            b_setting = parse_baseline_setting(fname)
            if (not include_repeats) and b_setting != "default":
                continue
            baseline_vals.setdefault((model, b_setting), []).append(val)
            continue

        # Steering runs: key by (model, vocab, steering_setting)
        vocab = infer_vocab_from_folders(folders)
        if vocab is None:
            continue

        setting = parse_steering_setting(fname)
        if (not include_repeats) and setting != "default":
            continue

        strength = parse_strength(fname)
        if strength is None or strength not in strengths:
            continue

        key = RowKey(model=model, vocab=vocab, setting=setting)
        if key not in strength_vals:
            strength_vals[key] = {s: [] for s in strengths}
        strength_vals[key][strength].append(val)

    baseline_by_key: Dict[Tuple[str, str], float] = {}
    for k, vals in baseline_vals.items():
        if vals:
            baseline_by_key[k] = sum(vals) / len(vals)

    scores: Dict[RowKey, Dict[int, Optional[float]]] = {}
    for key, per_strength in strength_vals.items():
        scores[key] = {}
        for s in strengths:
            vals = per_strength.get(s, [])
            scores[key][s] = (sum(vals) / len(vals)) if vals else None

    return baseline_by_key, scores


def build_table(
    baseline_by_key: Dict[Tuple[str, str], float],
    scores: Dict[RowKey, Dict[int, Optional[float]]],
    metric_name: str,
    strengths: List[int],
    decimals: int,
    pct: bool,
    caption: str,
    label: str,
) -> str:
    cols = ["Model", "Vocab", "Setting", "Base"] + [f"Str{s}" for s in strengths]
    colspec = "l l l " + " ".join(["r"] * (1 + len(strengths)))

    def sort_key(k: RowKey):
        model_order = {"2B": 0, "9B": 1}.get(k.model, 99)
        vocab_order = {"C": 0, "Q": 1}.get(k.vocab, 99)
        setting_order = 0 if k.setting == "default" else 1
        return (model_order, k.model, vocab_order, k.vocab, setting_order, k.setting)

    keys_sorted = sorted(scores.keys(), key=sort_key)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\hline")

    for k in keys_sorted:
        base = baseline_by_key.get((k.model, k.setting))
        row = [k.model, k.vocab, k.setting, fmt(base, decimals=decimals, pct=pct)]
        for s in strengths:
            row.append(fmt(scores[k].get(s), decimals=decimals, pct=pct))
        lines.append(" & ".join(latex_escape(str(x)) for x in row) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=str, required=True)
    ap.add_argument("--out_tex", type=str, required=True)
    ap.add_argument("--metrics", type=str, default="ask_f1", help="Comma-separated metric names")
    ap.add_argument("--strengths", type=str, default="1,3,5,10", help="Comma-separated strengths")
    ap.add_argument("--include_repeats", action="store_true")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--pct", action="store_true", help="Format values as percent (x100)")
    ap.add_argument("--caption_prefix", type=str, default="KnowNo results")
    ap.add_argument("--label_prefix", type=str, default="tab:knowno")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_tex = Path(args.out_tex)

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    strengths = [int(x.strip()) for x in args.strengths.split(",") if x.strip()]

    tables: List[str] = []
    for metric_name in metric_names:
        baseline_by_key, scores = aggregate(
            metrics_dir=metrics_dir,
            metric_name=metric_name,
            strengths=strengths,
            include_repeats=args.include_repeats,
        )
        caption = f"{args.caption_prefix}: {metric_name}"
        label = f"{args.label_prefix}:{metric_name}"
        tables.append(
            build_table(
                baseline_by_key=baseline_by_key,
                scores=scores,
                metric_name=metric_name,
                strengths=strengths,
                decimals=args.decimals,
                pct=args.pct,
                caption=caption,
                label=label,
            )
        )

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(tables), encoding="utf-8")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
