#!/usr/bin/env python
"""
KnowNo evaluation entrypoint with clean console output.

Goals:
- Keep a single progress bar for examples.
- Suppress Hugging Face download progress bars + verbose warnings.
- Avoid importing TensorFlow/Flax/JAX backends via Transformers where possible.

Usage:
  python scripts/run_knowno_eval.py --csv data/knowno_for_eval.csv --out results.json
"""
from __future__ import annotations

import os

# ---------------------------
# Must be set BEFORE importing transformers / knowno_eval
# ---------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # best-effort if TF is present

import argparse
import logging
import warnings

# Reduce noisy warnings from optional deps (pydantic etc.)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Reduce library logging noise
logging.basicConfig(level=logging.ERROR)
for name in [
    "transformers",
    "huggingface_hub",
    "torch",
    "transformer_lens",
    "sae_lens",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Transformers also has its own logging control
try:
    from transformers.utils import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
    hf_logging.disable_default_handler()
    hf_logging.enable_propagation()
except Exception:
    pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to knowno_for_eval.csv")
    p.add_argument("--out", required=True, help="Where to write evaluation JSON")
    p.add_argument("--model_name", default="google/gemma-2b-it", help="HF/TransformerLens model name")
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

    args = p.parse_args()

    # Delayed import so env/logging settings apply before heavy libs load
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
    )


if __name__ == "__main__":
    main()
