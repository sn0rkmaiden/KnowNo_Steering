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

def apply_prompt_repetition(prompt: str, mode: str = "none") -> str:
    """Apply prompt repetition / padding transforms.

    Modes:
      - none:     <PROMPT>
      - repeat2:  <PROMPT>\n\n<PROMPT>
      - repeat2_verbose: <PROMPT>\n\nLet me repeat that:\n<PROMPT>
      - repeat3:  <PROMPT>\n\nLet me repeat that:\n<PROMPT>\n\nLet me repeat that one more time:\n<PROMPT>
      - padding:  <PROMPT> + filler to roughly match repeat2 length (char-based)

    Note: padding is only a control condition; it is not intended to be meaningful text.
    """
    mode = (mode or "none").strip().lower()
    if mode in {"none", ""}:
        return prompt

    if mode in {"repeat2", "repeat"}:
        return prompt + "\n\n" + prompt

    if mode in {"repeat2_verbose", "repeat_verbose", "verbose"}:
        return prompt + "\n\nLet me repeat that:\n" + prompt

    if mode in {"repeat3", "triple"}:
        return (
            prompt
            + "\n\nLet me repeat that:\n"
            + prompt
            + "\n\nLet me repeat that one more time:\n"
            + prompt
        )

    if mode in {"padding", "pad"}:
        # Pad to ~2x the prompt length (roughly matching repeat2), using a low-semantic filler.
        target_len = len(prompt) * 2
        if len(prompt) >= target_len:
            return prompt
        needed = target_len - len(prompt)
        # Use '. ' (2 chars) as the primitive; add a leading blank line for separation.
        n = max(0, (needed - 2) // 2)
        filler = "\n\n" + (". " * n)
        return prompt + filler

    raise ValueError(f"Unknown repetition mode: {mode}")
