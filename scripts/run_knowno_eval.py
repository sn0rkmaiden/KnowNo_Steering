#!/usr/bin/env python
import argparse
from knowno_eval.eval import run_knowno_eval
from knowno_eval.models import SteeringConfig

def main():
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

    args = p.parse_args()

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
