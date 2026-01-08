from __future__ import annotations
from typing import List, Optional

CLARIFY_TEMPLATE = """You are a helpful robot in a small indoor environment.

Environment:
<DESCRIPTION>

User instruction:
<TASK>

Decide whether the instruction is ambiguous *given the environment*.
If it is ambiguous, generate clarifying question(s) that would help you perform it correctly.

Return your final answer **only** as a JSON object in the following format:

{
  "ambiguous": true or false,
  "question": ["question 1", "question 2", ...]
}

Formatting rules:
- Always output exactly one JSON object.
- If the task is **not ambiguous**, use \"ambiguous\": false and \"question\": [].
- If the task **is ambiguous**, use \"ambiguous\": true and provide at least one question.
- Never output explanations or text outside the JSON.
"""

def build_clarify_prompt(environment: str, task: str) -> str:
    return (CLARIFY_TEMPLATE
            .replace("<DESCRIPTION>", (environment or "").strip())
            .replace("<TASK>", (task or "").strip()))

def build_final_plan_prompt(environment: str, task: str, questions: List[str], provider_reply: Optional[str]) -> str:
    parts = []
    parts.append("You are a helpful robot. Produce a short, concrete plan of actions.")
    parts.append(f"Environment: {environment.strip()}")
    parts.append(f"Original instruction: {task.strip()}")

    if questions:
        qs = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions) if q.strip()])
        parts.append("You asked clarifying question(s):\n" + qs)

    if provider_reply:
        parts.append("The user answered: " + provider_reply.strip())

    parts.append("Now write your plan. Return only the plan text (no JSON).")
    return "\n\n".join(parts)
