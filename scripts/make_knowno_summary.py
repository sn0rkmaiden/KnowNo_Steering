#!/usr/bin/env python3
"""Generate LaTeX summary tables from a KnowNo `metrics/` directory.

Scans all `*.json` metric files under `metrics_dir`, infers condition metadata
from their paths, aggregates metrics across features (mean), and produces LaTeX
that compiles with plain LaTeX (no extra packages).

Table layout (per metric):
  Model | Vocab | Variant | Base | Str1 | Str3 | Str5 | Str10
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


def latex_escape(text: str) -> str:
    """Escape LaTeX special chars so the output compiles."""
    # Backslash first
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("_", r"\_")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    return text


@dataclass(frozen=True)
class Key:
    model: str        # "2B" or "9B" (or other)
    vocab: str        # "C" or "Q" (or other)
    variant: str      # "default" or repetition variant


STRENGTHS_DEFAULT = [1, 3, 5, 10]


def infer_model_size(model_dir: str) -> str:
    m = model_dir.lower()
    if "2b" in m:
        return "2B"
    if "9b" in m:
        return "9B"
    # fallback: keep as-is
    return model_dir


def infer_vocab(vocab_dir: str) -> str:
    v = vocab_dir.lower()
    if v.startswith("clar"):
        return "C"
    if v.startswith("question"):
        return "Q"
    return vocab_dir


def parse_file_metadata(rel_path: Path) -> Tuple[str, str, str, Optional[float]]:
    """Return (model_size, vocab, variant, strength). strength=None => baseline."""
    parts = rel_path.parts
    model_size = infer_model_size(parts[0]) if len(parts) >= 1 else "?"
    vocab = infer_vocab(parts[1]) if len(parts) >= 2 else "?"
    name = rel_path.name

    # steering files
    m = re.match(r"^steering_str(?P<strength>-?\d+(?:\.\d+)?)_feat\d+(?P<suffix>.*)\.json$", name)
    if m:
        strength = float(m.group("strength"))
        suffix = m.group("suffix").lstrip("_")
        variant = suffix if suffix else "default"
        return model_size, vocab, variant, strength

    # baseline files
    if name == "results_baseline.json":
        return model_size, vocab, "default", None
    m2 = re.match(r"^(?P<var>.+)_baseline\.json$", name)
    if m2:
        return model_size, vocab, m2.group("var"), None

    # other JSONs: treat as variant=stem, baseline
    return model_size, vocab, rel_path.stem, None


def is_repetition_variant(variant: str) -> bool:
    v = variant.lower()
    return ("repeat" in v) or ("prrepeat" in v)


def format_val(x: Optional[float], *, as_percent: bool = False, decimals: int = 3) -> str:
    if x is None:
        return "--"
    if as_percent:
        return f"{100.0 * x:.1f}"
    return f"{x:.{decimals}f}"


def gather_metrics(metrics_dir: Path, metric_keys: List[str], include_repetition: bool) -> Tuple[Dict[Key, Dict[str, List[float]]], Dict[Key, Dict[float, Dict[str, List[float]]]]]:
    """Return (baseline_map, steered_map).

    baseline_map[key][metric] -> list of values
    steered_map[key][strength][metric] -> list of values
    """
    baseline_map: Dict[Key, Dict[str, List[float]]] = {}
    steered_map: Dict[Key, Dict[float, Dict[str, List[float]]]] = {}

    for path in metrics_dir.rglob("*.json"):
        rel = path.relative_to(metrics_dir)
        model_size, vocab, variant, strength = parse_file_metadata(rel)
        if (not include_repetition) and is_repetition_variant(variant):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        # verify this looks like a metrics file
        if not all(k in data for k in metric_keys):
            # allow missing breakdown etc; but require at least one metric
            if not any(k in data for k in metric_keys):
                continue

        key = Key(model=model_size, vocab=vocab, variant=variant)

        if strength is None:
            base = baseline_map.setdefault(key, {k: [] for k in metric_keys})
            for mk in metric_keys:
                if mk in data and isinstance(data[mk], (int, float)):
                    base[mk].append(float(data[mk]))
        else:
            s = float(strength)
            slot = steered_map.setdefault(key, {})
            met = slot.setdefault(s, {k: [] for k in metric_keys})
            for mk in metric_keys:
                if mk in data and isinstance(data[mk], (int, float)):
                    met[mk].append(float(data[mk]))

    return baseline_map, steered_map


def mean_or_none(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    if not xs:
        return None
    return mean(xs)


def make_table_for_metric(
    metric: str,
    baseline_map: Dict[Key, Dict[str, List[float]]],
    steered_map: Dict[Key, Dict[float, Dict[str, List[float]]]],
    strengths: List[float],
    percent_metrics: List[str],
    include_repetition: bool,
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    as_percent = metric in percent_metrics

    # keys to print: union of baseline and steered
    all_keys = set(baseline_map.keys()) | set(steered_map.keys())
    # sort by model, vocab, then variant (default first)
    def sort_key(k: Key):
        var_rank = 0 if k.variant == "default" else 1
        return (k.model, k.vocab, var_rank, k.variant)
    keys_sorted = sorted(all_keys, key=sort_key)

    header_cols = ["Model", "Vocab", "Variant", "Base"] + [f"Str{int(s) if float(s).is_integer() else s}" for s in strengths]

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    colspec = "lll" + "r" * (1 + len(strengths))
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(" & ".join([latex_escape(c) for c in header_cols]) + r" \\")
    lines.append(r"\hline")

    for k in keys_sorted:
        if (not include_repetition) and is_repetition_variant(k.variant):
            continue
        base_val = None
        if k in baseline_map:
            base_val = mean_or_none(baseline_map[k].get(metric, []))

        row = [latex_escape(k.model), latex_escape(k.vocab), latex_escape(k.variant)]
        row.append(format_val(base_val, as_percent=as_percent))

        for s in strengths:
            sval = None
            if k in steered_map and float(s) in steered_map[k]:
                sval = mean_or_none(steered_map[k][float(s)].get(metric, []))
            row.append(format_val(sval, as_percent=as_percent))

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=str, required=True, help="Path to metrics folder")
    ap.add_argument("--out_tex", type=str, required=True, help="Output .tex path")
    ap.add_argument("--metrics", type=str, default="ask_f1", help="Comma-separated metrics to include (one table per metric)")
    ap.add_argument("--strengths", type=str, default="1,3,5,10", help="Comma-separated steering strengths to pivot into columns")
    ap.add_argument("--include_repetition", action="store_true", help="Include rows for prompt-repetition variants")
    ap.add_argument("--percent_metrics", type=str, default="plan_object_accuracy,plan_location_accuracy", help="Metrics to print as percentages")
    ap.add_argument("--caption_prefix", type=str, default="", help="Optional caption prefix")
    ap.add_argument("--label_prefix", type=str, default="tab:knowno_", help="Optional label prefix")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip() != ""]
    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip() != ""]
    percent_metrics = [m.strip() for m in args.percent_metrics.split(",") if m.strip() != ""]

    baseline_map, steered_map = gather_metrics(metrics_dir, metric_list, args.include_repetition)

    out_parts: List[str] = []
    out_parts.append("% Auto-generated by make_knowno_latex_summary.py")
    out_parts.append("% Uses only standard LaTeX (tabular + hline).")
    out_parts.append("")

    for metric in metric_list:
        cap = None
        if args.caption_prefix:
            cap = f"{args.caption_prefix}{metric}"
        lab = None
        if args.label_prefix:
            lab = f"{args.label_prefix}{metric}"
        out_parts.append(make_table_for_metric(
            metric=metric,
            baseline_map=baseline_map,
            steered_map=steered_map,
            strengths=strengths,
            percent_metrics=percent_metrics,
            include_repetition=args.include_repetition,
            caption=cap,
            label=lab,
        ))

    Path(args.out_tex).write_text("\n".join(out_parts))
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
