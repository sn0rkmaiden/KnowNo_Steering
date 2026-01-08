from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .text_similarity import similarity, normalize_text

DEFAULT_ASK_CATEGORIES = {
    "spatial_ambiguous_task",
    "multilabel_task",
    "creative_multilabel_task",
}

def _as_set_from_csv(s: str) -> Set[str]:
    # options separated by '|' in our adapted CSV
    s = (s or "").strip()
    if not s:
        return set()
    if "|" in s:
        return {x.strip() for x in s.split("|") if x.strip()}
    return {s}

def plan_mentions_object(plan: str, acceptable_objects: Set[str]) -> bool:
    plan_n = normalize_text(plan)
    for obj in acceptable_objects:
        if normalize_text(obj) and normalize_text(obj) in plan_n:
            return True
    return False

def plan_mentions_location(plan: str, location: str) -> bool:
    loc = normalize_text(location)
    return bool(loc) and (loc in normalize_text(plan))

@dataclass
class KnowNoMetrics:
    n: int
    ask_precision: float
    ask_recall: float
    ask_f1: float
    avg_question_similarity: float
    plan_object_accuracy: float
    plan_location_accuracy: float
    breakdown_by_category: Dict[str, Dict[str, float]]

def compute_knowno_metrics(
    eval_json_path: str,
    ask_categories: Optional[Set[str]] = None,
) -> KnowNoMetrics:
    ask_categories = ask_categories or set(DEFAULT_ASK_CATEGORIES)

    payload = json.load(open(eval_json_path, "r", encoding="utf-8"))
    rows: List[Dict[str, Any]] = payload["results"]

    # ask-necessity classification
    tp=fp=fn=tn=0
    q_sims: List[float] = []

    # plan correctness
    obj_ok=0
    loc_ok=0

    # per category breakdown
    cat_stats: Dict[str, Dict[str, float]] = {}

    for r in rows:
        cat = str(r.get("ambiguity_type","") or "")
        need_ask = cat in ask_categories
        asked = bool(r.get("model_ambiguous")) and len(r.get("model_questions") or [])>0

        if asked and need_ask: tp += 1
        elif asked and not need_ask: fp += 1
        elif (not asked) and need_ask: fn += 1
        else: tn += 1

        # question similarity only when a gold question exists
        gold_q = str(r.get("gold_question","") or "")
        if need_ask and gold_q.strip() and asked:
            best = 0.0
            for q in (r.get("model_questions") or []):
                best = max(best, similarity(str(q), gold_q))
            q_sims.append(best)

        # plan object correctness: accept provider choice (single) OR any of the options in user_intent
        plan = str(r.get("model_final_plan","") or "")
        acceptable = _as_set_from_csv(str(r.get("user_intent","") or "")) or _as_set_from_csv(str(r.get("intent_object","") or ""))
        if plan_mentions_object(plan, acceptable):
            obj_ok += 1
        if plan_mentions_location(plan, str(r.get("intent_location","") or "")):
            loc_ok += 1

        # category breakdown
        st = cat_stats.setdefault(cat, {"n":0, "need_ask":0, "asked":0, "ask_acc":0})
        st["n"] += 1
        st["need_ask"] += int(need_ask)
        st["asked"] += int(asked)

    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall = tp / (tp+fn) if (tp+fn) else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0

    avg_q = sum(q_sims)/len(q_sims) if q_sims else 0.0
    plan_obj_acc = obj_ok / len(rows) if rows else 0.0
    plan_loc_acc = loc_ok / len(rows) if rows else 0.0

    # compute ask accuracy per category vs need_ask
    for cat, st in cat_stats.items():
        n = st["n"]
        need = st["need_ask"]
        asked = st["asked"]
        # "ask_rate" and "need_rate" helpful
        st["ask_rate"] = asked / n if n else 0.0
        st["need_rate"] = need / n if n else 0.0
        # no exact accuracy without labeled per-row? We have need_ask definition.
        # We'll compute rate of correct ask decisions in this category:
        # correct = asked when need_ask else not asked
        correct = 0
        for r in rows:
            if str(r.get("ambiguity_type","") or "") != cat:
                continue
            need_ask_r = (cat in ask_categories)
            asked_r = bool(r.get("model_ambiguous")) and len(r.get("model_questions") or [])>0
            if asked_r == need_ask_r:
                correct += 1
        st["ask_decision_accuracy"] = correct / n if n else 0.0

    return KnowNoMetrics(
        n=len(rows),
        ask_precision=precision,
        ask_recall=recall,
        ask_f1=f1,
        avg_question_similarity=avg_q,
        plan_object_accuracy=plan_obj_acc,
        plan_location_accuracy=plan_loc_acc,
        breakdown_by_category=cat_stats,
    )

def save_metrics(metrics: KnowNoMetrics, out_path: str) -> str:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)
    return out_path
