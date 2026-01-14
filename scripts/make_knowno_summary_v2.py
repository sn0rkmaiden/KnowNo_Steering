#!/usr/bin/env python3
"""
scripts/make_knowno_summary.py

Create a compact LaTeX table from a mirrored `metrics/` directory.

Columns: Model | Vocab | Setting | Base | Str1 | Str3 | Str5 | Str10

- Compute ONE baseline value per model size (2B, 9B) from baseline metric JSON files
  and place it under "Base" for every row with that model size (regardless of vocab/setting).
- Steering values are averaged across all feature runs found for each
  (model, vocab, setting, strength).
- Setting = "default" or a short repeat label extracted from filename (repeat*, prrepeat*).
- --include_repeats controls whether repeat settings are included.

Assumptions about folder naming:
- model size inferred from path folders containing "2b" or "9b"
- vocab inferred from path folders containing "clar_vocab" (C) or "question_vocab" (Q)

Assumptions about filenames:
- steering strength encoded like: steering_str10_feat375.json
- baseline files contain "baseline" or "results_baseline" in filename
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
    # steering_str10_...
    m = re.search(r"steering[_-]str([0-9]+)", fname.lower())
    if m:
        return int(m.group(1))
    # str10 / strength10
    m = re.search(r"(?:^|[_-])str(?:ength)?\s*([0-9]+)", fname.lower())
    if m:
        return int(m.group(1))
    return None


def parse_setting(fname: str) -> str:
    f = fname.lower()
    if ("repeat" in f) or ("prrepeat" in f):
        m = re.search(r"(prrepeat[^_.]*|repeat[^_.]*)", f)
        return m.group(1) if m else "repeat"
    return "default"


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


@dataclass(frozen=True)
class RowKey:
    model: str    # 2B / 9B
    vocab: str    # C / Q
    setting: str  # default / repeat...


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
) -> Tuple[Dict[str, float], Dict[RowKey, Dict[int, Optional[float]]]]:
    """
    Returns:
      baseline_by_model: {"2B": value, "9B": value}
      scores: RowKey -> {strength -> avg(metric)}
    """
    files = [p for p in metrics_dir.rglob("*.json") if p.is_file()]

    baseline_vals: Dict[str, List[float]] = {}  # model -> list of baseline metric values
    strength_vals: Dict[RowKey, Dict[int, List[float]]] = {}  # row -> strength -> vals

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

        # baseline accumulation (per model only)
        if is_baseline_file(fname):
            baseline_vals.setdefault(model, []).append(val)
            continue

        # steering runs
        vocab = infer_vocab_from_folders(folders)
        if vocab is None:
            continue

        setting = parse_setting(fname)
        if (not include_repeats) and setting != "default":
            continue

        strength = parse_strength(fname)
        if strength is None or strength not in strengths:
            continue

        key = RowKey(model=model, vocab=vocab, setting=setting)
        if key not in strength_vals:
            strength_vals[key] = {s: [] for s in strengths}
        strength_vals[key][strength].append(val)

    baseline_by_model: Dict[str, float] = {}
    for model, vals in baseline_vals.items():
        if vals:
            baseline_by_model[model] = sum(vals) / len(vals)

    scores: Dict[RowKey, Dict[int, Optional[float]]] = {}
    for key, per_strength in strength_vals.items():
        scores[key] = {}
        for s in strengths:
            vals = per_strength.get(s, [])
            scores[key][s] = (sum(vals) / len(vals)) if vals else None

    return baseline_by_model, scores


def build_table(
    baseline_by_model: Dict[str, float],
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
        base = baseline_by_model.get(k.model)
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
        baseline_by_model, scores = aggregate(
            metrics_dir=metrics_dir,
            metric_name=metric_name,
            strengths=strengths,
            include_repeats=args.include_repeats,
        )
        caption = f"{args.caption_prefix}: {metric_name}"
        label = f"{args.label_prefix}:{metric_name}"
        tables.append(
            build_table(
                baseline_by_model=baseline_by_model,
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
