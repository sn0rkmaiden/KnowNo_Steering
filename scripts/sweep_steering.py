#!/usr/bin/env python3
"""
Sweep SAE steering over multiple (feature, strength) combinations by repeatedly calling scripts/run_knowno_eval.py.

Outputs are saved as:
  steering_str{strength}_feat{feature}.json

Example:
  python scripts/sweep_steering.py \
    --csv data/knowno_for_eval.csv \
    --model_name gemma-2-9b-it \
    --sae_release <SAE_RELEASE> \
    --sae_id <SAE_ID> \
    --features 123,456,789 \
    --strengths 0,2,4,6 \
    --out_dir results/sweep \
    --num_examples 200 \
    --seed 0 \
    --compute_max_per_prompt
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_list_arg(value: str, kind: str) -> List[str]:
    """
    Accept:
      - comma-separated: "1,2,3"
      - whitespace-separated: "1 2 3"
      - single value: "1"
    Returns list of strings as provided (trimmed).
    """
    if value is None:
        return []
    v = value.strip()
    if not v:
        return []
    # allow commas and/or whitespace
    parts = re.split(r"[,\s]+", v)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _read_lines(path: Path) -> List[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _safe_strength_for_filename(s: str) -> str:
    """
    Keep filenames OS-friendly while still human-readable.
    - replace '/' and spaces
    - leave '.' intact (it's safe), but normalize very long floats.
    """
    s = s.strip()
    s = s.replace("/", "_").replace(" ", "")
    return s


def build_cmd(
    python_exe: str,
    run_script: Path,
    csv_path: Path,
    model_name: str,
    sae_release: str,
    sae_id: str,
    feature: str,
    strength: str,
    out_path: Path,
    passthrough: List[str],
) -> List[str]:
    cmd = [
        python_exe,
        str(run_script),
        "--csv",
        str(csv_path),
        "--out",
        str(out_path),
        "--model_name",
        model_name,
        "--use_steering",
        "--sae_release",
        sae_release,
        "--sae_id",
        sae_id,
        "--steering_feature",
        str(feature),
        "--steering_strength",
        str(strength),
    ]
    cmd.extend(passthrough)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep steering features and strengths for KnowNo eval.")
    parser.add_argument("--csv", type=str, required=True, help="Path to KnowNo CSV (e.g., data/knowno_for_eval.csv).")
    parser.add_argument("--model_name", type=str, required=True, help="HF model name or local identifier used by run_knowno_eval.py.")
    parser.add_argument("--sae_release", type=str, required=True, help="SAE release identifier (sae-lens).")
    parser.add_argument("--sae_id", type=str, required=True, help="SAE id within the release (sae-lens).")

    group_f = parser.add_mutually_exclusive_group(required=True)
    group_f.add_argument("--features", type=str, help="Comma/space-separated feature ids, e.g. '123,456,789'.")
    group_f.add_argument("--features_file", type=str, help="Text file with one feature id per line.")

    group_s = parser.add_mutually_exclusive_group(required=True)
    group_s.add_argument("--strengths", type=str, help="Comma/space-separated strengths, e.g. '0,2,4,6'.")
    group_s.add_argument("--strengths_file", type=str, help="Text file with one strength per line.")

    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write JSON outputs into.")
    parser.add_argument("--run_script", type=str, default="scripts/run_knowno_eval.py", help="Path to run_knowno_eval.py.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONs.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")

    # Everything else is passed through to run_knowno_eval.py (e.g., --num_examples, --seed, --device, --max_new_tokens, --compute_max_per_prompt, ...)
    args, passthrough = parser.parse_known_args()

    csv_path = Path(args.csv)
    run_script = Path(args.run_script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.features_file:
        feats = _read_lines(Path(args.features_file))
    else:
        feats = _parse_list_arg(args.features, "features")
    if args.strengths_file:
        strengths = _read_lines(Path(args.strengths_file))
    else:
        strengths = _parse_list_arg(args.strengths, "strengths")

    if not feats:
        raise SystemExit("No features provided.")
    if not strengths:
        raise SystemExit("No strengths provided.")

    python_exe = sys.executable

    # A couple of safe defaults to reduce noisy/hanging behavior in some setups
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")

    total = len(feats) * len(strengths)
    i = 0

    for feat in feats:
        for strength in strengths:
            i += 1
            strength_str = _safe_strength_for_filename(str(strength))
            out_path = out_dir / f"steering_str{strength_str}_feat{feat}.json"

            if out_path.exists() and not args.overwrite:
                print(f"[{i}/{total}] SKIP exists: {out_path}")
                continue

            cmd = build_cmd(
                python_exe=python_exe,
                run_script=run_script,
                csv_path=csv_path,
                model_name=args.model_name,
                sae_release=args.sae_release,
                sae_id=args.sae_id,
                feature=feat,
                strength=strength,
                out_path=out_path,
                passthrough=passthrough,
            )

            print(f"[{i}/{total}] RUN feat={feat} strength={strength} -> {out_path}")
            print("  " + " ".join(shlex.quote(c) for c in cmd))

            if args.dry_run:
                continue

            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                raise SystemExit(f"Run failed (exit={proc.returncode}) for feat={feat}, strength={strength}")

    print("Done.")


if __name__ == "__main__":
    main()
