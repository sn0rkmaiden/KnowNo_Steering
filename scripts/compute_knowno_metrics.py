#!/usr/bin/env python
"""Compute KnowNo metrics.

Two modes:

1) Single-file mode (backwards compatible)
   python scripts/compute_knowno_metrics.py --eval_json path/to/results.json --out metrics.json

2) Directory sweep mode
   python scripts/compute_knowno_metrics.py --results_dir path/to/results

Directory mode mirrors the directory structure under a sibling (or provided) metrics
root and writes one metrics JSON per results JSON.

Example:
  results/2bmodel/question_vocab/231_ch0/steering_str5_feat231.json
->metrics/2bmodel/question_vocab/231_ch0/steering_str5_feat231.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from tqdm import tqdm

from knowno_eval.metrics import compute_knowno_metrics, save_metrics, DEFAULT_ASK_CATEGORIES


def _parse_ask_categories(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def _iter_json_files(root: Path) -> Iterable[Path]:
    # include only .json files (exclude obvious logs)
    for p in root.rglob("*.json"):
        if p.name.endswith(".stderr.log"):
            continue
        yield p


def _default_metrics_root(results_dir: Path) -> Path:
    # sibling "metrics" next to "results"
    parent = results_dir.parent
    return parent / "metrics"


def _safe_load_has_results(path: Path) -> bool:
    """Cheap validation to skip non-eval JSONs."""
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return isinstance(obj, dict) and isinstance(obj.get("results"), list)
    except Exception:
        return False


def _compute_one(in_path: Path, out_path: Path, ask_cats: Optional[Set[str]]) -> Tuple[bool, Optional[str]]:
    """Compute metrics for one eval JSON. Returns (ok, error_message)."""
    try:
        m = compute_knowno_metrics(str(in_path), ask_categories=ask_cats)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics(m, str(out_path))
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser()

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval_json", help="Single eval JSON output from run_knowno_eval.py")
    mode.add_argument("--results_dir", help="Directory containing many eval JSONs (will recurse)")

    # single-file output
    p.add_argument("--out", default=None, help="Output path for single-file mode")

    # directory mode output root
    p.add_argument(
        "--metrics_dir",
        default=None,
        help=(
            "Root directory to write metrics in directory mode. "
            "Default: sibling folder named 'metrics' next to results_dir."
        ),
    )

    p.add_argument(
        "--ask_categories",
        type=str,
        default=None,
        help=(
            "Comma-separated ambiguity_type values that should trigger clarification. "
            f"Default: {','.join(sorted(DEFAULT_ASK_CATEGORIES))}"
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metrics files in directory mode.",
    )

    p.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Skip JSON files that don't look like run_knowno_eval outputs (no 'results' list).",
    )

    args = p.parse_args()

    ask_cats = _parse_ask_categories(args.ask_categories)

    # ---------- single-file mode ----------
    if args.eval_json:
        m = compute_knowno_metrics(args.eval_json, ask_categories=ask_cats)

        print("n =", m.n)
        print("ask_precision =", round(m.ask_precision, 4))
        print("ask_recall    =", round(m.ask_recall, 4))
        print("ask_f1        =", round(m.ask_f1, 4))
        print("avg_question_similarity =", round(m.avg_question_similarity, 4))
        print("plan_object_accuracy    =", round(m.plan_object_accuracy, 4))
        print("plan_location_accuracy  =", round(m.plan_location_accuracy, 4))

        if args.out:
            save_metrics(m, args.out)
            print("saved metrics to", args.out)
        return

    # ---------- directory mode ----------
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"--results_dir does not exist or is not a directory: {results_dir}")

    metrics_dir = Path(args.metrics_dir).expanduser().resolve() if args.metrics_dir else _default_metrics_root(results_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the directory structure up-front so "metrics" matches "results"
    # (even for folders that contain no JSON files).
    for d in results_dir.rglob("*"):
        if d.is_dir():
            (metrics_dir / d.relative_to(results_dir)).mkdir(parents=True, exist_ok=True)

    # gather files
    files = list(_iter_json_files(results_dir))
    if args.skip_invalid:
        files = [p for p in files if _safe_load_has_results(p)]

    if not files:
        print("No JSON files found under", results_dir)
        return

    ok = 0
    skipped = 0
    failed = 0

    for in_path in tqdm(files, desc="Computing metrics", unit="file"):
        rel = in_path.relative_to(results_dir)
        out_path = metrics_dir / rel

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        success, err = _compute_one(in_path, out_path, ask_cats)
        if success:
            ok += 1
        else:
            failed += 1
            print(f"[WARN] failed: {in_path} -> {out_path}: {err}", file=sys.stderr)

    print("results_dir =", str(results_dir))
    print("metrics_dir =", str(metrics_dir))
    print("computed    =", ok)
    print("skipped     =", skipped)
    print("failed      =", failed)


if __name__ == "__main__":
    main()
