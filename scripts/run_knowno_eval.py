#!/usr/bin/env python3
"""
Quiet KnowNo evaluation runner.

What it does:
- Sets HF/Transformers env vars to reduce download/progress spam.
- Optionally redirects STDERR (where TensorFlow/XLA cuDNN/cuBLAS "already registered" lines are printed).
- Keeps a single progress bar from the evaluation loop.
"""

from __future__ import annotations

import os
import sys
import argparse


def _set_quiet_env() -> None:
    # reduce logging
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


def _redirect_stderr(path: str):
    """
    Redirect both sys.stderr and the underlying OS fd=2.
    Needed because the XLA/cuDNN/cuBLAS warnings are C++ logs written to STDERR.
    """
    f = open(path, "w", buffering=1)
    sys.stderr = f  # type: ignore
    os.dup2(f.fileno(), 2)
    return f


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to knowno_for_eval.csv")
    p.add_argument("--out", required=True, help="Where to write evaluation JSON")
    p.add_argument("--model_name", default="google/gemma-2b-it", help="TransformerLens model name")
    p.add_argument("--num_examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # steering
    p.add_argument("--use_steering", action="store_true")
    p.add_argument("--sae_release", type=str, default=None)
    p.add_argument("--sae_id", type=str, default=None)
    p.add_argument("--steering_feature", type=int, default=None)
    p.add_argument("--steering_strength", type=float, default=0.0)
    p.add_argument("--max_act", type=float, default=None)
    p.add_argument("--compute_max_per_prompt", action="store_true")

    # prompt repetition experiment
    p.add_argument(
        "--prompt_repeat",
        default="none",
        choices=["none", "repeat2", "repeat2_verbose", "repeat3", "padding"],
        help="Apply a repetition/padding transform to the prompt.",
    )
    p.add_argument(
        "--repeat_stage",
        default="both",
        choices=["clarify", "plan", "both"],
        help="Which stage(s) to apply --prompt_repeat to.",
    )

    # output hygiene
    p.add_argument(
        "--stderr_log",
        default=None,
        help="Redirect STDERR to this file (silences XLA/cuDNN spam). "
             "Default: '<out>.stderr.log'. Set to '-' to keep STDERR.",
    )

    args = p.parse_args()

    _set_quiet_env()

    # Redirect stderr BEFORE importing any ML libraries
    stderr_file = None
    stderr_path = args.stderr_log
    if stderr_path is None:
        stderr_path = args.out + ".stderr.log"
    if stderr_path != "-":
        stderr_file = _redirect_stderr(stderr_path)

    try:
        from knowno_eval.eval import run_knowno_eval
        from knowno_eval.models import SteeringConfig

        steering = SteeringConfig(
            enabled=bool(args.use_steering),
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            feature=args.steering_feature,
            strength=float(args.steering_strength),
            max_act=args.max_act,
            compute_max_per_prompt=bool(args.compute_max_per_prompt),
        )

        run_knowno_eval(
            csv_path=args.csv,
            out_json=args.out,
            model_name=args.model_name,
            num_examples=args.num_examples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            steering=steering,
            prompt_repeat=args.prompt_repeat,
            repeat_stage=args.repeat_stage,
        )
    finally:
        if stderr_file is not None:
            stderr_file.flush()
            stderr_file.close()


if __name__ == "__main__":
    main()
