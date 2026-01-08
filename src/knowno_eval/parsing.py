import json
import re
import ast
from typing import Any, Dict, Optional

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_EOS_RE = re.compile(r"</s>|<eos>|<\|endoftext\|>", re.IGNORECASE)

def _strip_fences_and_eos(s: str) -> str:
    s = (s or "").strip()
    # if response is in a code fence, keep inner
    m = _CODE_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    s = _EOS_RE.sub("", s)
    return s.strip()

def _first_curly_to_end(s: str) -> str:
    i = s.find("{")
    return s[i:].strip() if i != -1 else s.strip()

def _repair_common_json_issues(s: str) -> str:
    # Remove trailing commas before ] or }
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    # Replace Python literals with JSON literals
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # Quote keys if model omitted quotes: {ambiguous: true, question: [...]}
    s = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', s)
    return s

def parse_clarify_json(raw_output: str) -> Optional[Dict[str, Any]]:
    """
    Robust parse for:
      {
        "ambiguous": true|false,
        "question": [ "...", "...", ... ]
      }
    Accepts loose JSON, python-ish dicts, code fences, and minor truncation.
    """
    body = _first_curly_to_end(_strip_fences_and_eos(raw_output))

    # 1) strict JSON
    try:
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) repaired JSON
    repaired = _repair_common_json_issues(body)
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 3) literal eval fallback (after mapping json literals)
    pyish = (repaired
             .replace("null", "None")
             .replace("true", "True")
             .replace("false", "False"))
    try:
        obj = ast.literal_eval(pyish)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def coerce_clarify_schema(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"ambiguous": False, "question": []}

    ambiguous = bool(obj.get("ambiguous", False))

    q = obj.get("question", obj.get("questions", []))
    if isinstance(q, str):
        questions = [q]
    elif isinstance(q, list):
        questions = [str(x) for x in q if str(x).strip()]
    else:
        questions = []

    return {"ambiguous": ambiguous, "question": questions}
