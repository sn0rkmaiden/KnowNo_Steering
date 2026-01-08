from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .models import GemmaHookedModel, SteeringConfig
from .prompts import build_clarify_prompt, build_final_plan_prompt
from .parsing import parse_clarify_json, coerce_clarify_schema

def _pick_provider_choice(user_intent: str, intent_object: str) -> str:
    """
    For KnowNo rows that represent multiple valid objects, user_intent often encodes options using '|'.
    We pick the first option deterministically for reproducible evaluation.
    """
    s = (user_intent or "").strip()
    if "|" in s:
        return s.split("|")[0].strip()
    return (intent_object or "").strip()

def run_knowno_eval(
    csv_path: str,
    out_json: str,
    model_name: str,
    num_examples: Optional[int] = None,
    seed: int = 0,
    max_new_tokens: int = 256,
    steering: Optional[SteeringConfig] = None,
) -> str:
    df = pd.read_csv(csv_path)
    if num_examples is not None:
        df = df.sample(n=min(int(num_examples), len(df)), random_state=int(seed)).reset_index(drop=True)

    model = GemmaHookedModel(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        steering=steering or SteeringConfig(enabled=False),
    )

    results: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        env = str(row.get("environment_full", "") or "")
        task = str(row.get("ambiguous_task", "") or "")
        ambiguity_type = str(row.get("ambiguity_type", "") or "")
        gold_question = row.get("question", None)
        gold_question = "" if pd.isna(gold_question) else str(gold_question)
        intent_object = str(row.get("intent_object", "") or "")
        intent_location = str(row.get("intent_location", "") or "")
        user_intent = str(row.get("user_intent", "") or "")

        clarify_prompt = build_clarify_prompt(env, task)
        raw, _ = model.request(clarify_prompt, json_format=True)
        parsed = coerce_clarify_schema(parse_clarify_json(raw))
        questions = parsed["question"]
        predicted_ambiguous = bool(parsed["ambiguous"])

        provider_object = _pick_provider_choice(user_intent, intent_object)
        provider_reply = None
        if predicted_ambiguous and questions:
            provider_reply = f"I meant: {provider_object}. Location/target: {intent_location}."

        final_prompt = build_final_plan_prompt(env, task, questions, provider_reply)
        final_plan, _ = model.request(final_prompt, json_format=False)

        results.append({
            "id": int(row.get("id", -1)) if "id" in row else -1,
            "ambiguity_type": ambiguity_type,
            "environment_full": env,
            "ambiguous_task": task,
            "gold_question": gold_question,
            "intent_object": intent_object,
            "intent_location": intent_location,
            "user_intent": user_intent,
            "model_clarify_raw": raw,
            "model_ambiguous": predicted_ambiguous,
            "model_questions": questions,
            "provider_object_choice": provider_object,
            "provider_reply": provider_reply,
            "model_final_plan": final_plan,
        })

    payload = {
        "run_info": {
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "num_examples": len(results),
            "seed": seed,
            "steering": asdict(steering or SteeringConfig(enabled=False)),
        },
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_json
