#!/usr/bin/env python
import argparse
from knowno_eval.metrics import compute_knowno_metrics, save_metrics, DEFAULT_ASK_CATEGORIES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", required=True, help="Output JSON from run_knowno_eval.py")
    p.add_argument("--out", required=False, default=None, help="Optional path to save metrics JSON")
    p.add_argument("--ask_categories", type=str, default=None,
                   help="Comma-separated ambiguity_type values that should trigger clarification. "
                        f"Default: {','.join(sorted(DEFAULT_ASK_CATEGORIES))}")
    args = p.parse_args()

    ask_cats = None
    if args.ask_categories:
        ask_cats = {x.strip() for x in args.ask_categories.split(",") if x.strip()}

    m = compute_knowno_metrics(args.eval_json, ask_categories=ask_cats)

    # print a compact summary
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

if __name__ == "__main__":
    main()
