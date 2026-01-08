# KnowNo evaluation

## What it does

For each row in `data/knowno_for_eval.csv` it runs **two steps**:

1) **Clarification step** (JSON-only)
- model returns:
  `{ "ambiguous": true/false, "question": [...] }`

2) **Final plan step** (text-only)
- if the model asked clarifying questions, we simulate a user reply by deterministically choosing the **first** option in `user_intent` (split by `|`) and providing the recorded `intent_location`.
- model returns a short plan.

Outputs are saved to a single JSON file.

## Setup

Create an env and install deps:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run evaluation (baseline)

```bash
python scripts/run_knowno_eval.py \
  --csv data/knowno_data_processed.csv \
  --out results/results_baseline.json \
  --model_name google/gemma-2b-it \
  --num_examples 200
```

## Run evaluation (with SAE steering)

```bash
python scripts/run_knowno_eval.py \
  --csv data/knowno_data_processed.csv \
  --out results/results_steered.json \
  --model_name gemma-2-9b-it \
  --use_steering \
  --sae_release <SAE_RELEASE> \
  --sae_id <SAE_ID> \
  --steering_feature 1234 \
  --steering_strength 6.0 \
  --compute_max_per_prompt
```

## Compute metrics

```bash
python scripts/compute_knowno_metrics.py \
  --eval_json results_baseline.json \
  --out results/metrics_baseline.json
```

Override which categories "should ask" (clarification-necessary):

```bash
python scripts/compute_knowno_metrics.py \
  --eval_json results_baseline.json \
  --ask_categories spatial_ambiguous_task,multilabel_task,creative_multilabel_task
```

## Metrics

- **ask_precision/ask_recall/ask_f1**: did the model ask a clarifying question when it *should* (based on category list)
- **avg_question_similarity**: best lexical similarity between model questions and the (templated) gold question when applicable
- **plan_object_accuracy**: whether the final plan mentions one of the acceptable objects (from `user_intent` or `intent_object`)
- **plan_location_accuracy**: whether the final plan mentions `intent_location`
- **breakdown_by_category** in the saved metrics JSON

