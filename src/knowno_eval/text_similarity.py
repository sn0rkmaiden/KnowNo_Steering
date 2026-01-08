import re
from typing import Set

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def token_set(s: str) -> Set[str]:
    return set(_WORD_RE.findall(normalize_text(s)))

def jaccard(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def contains_score(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0
    if a_n in b_n or b_n in a_n:
        return 0.95
    return 0.0

def similarity(a: str, b: str) -> float:
    # fast exact/contains
    c = contains_score(a, b)
    if c > 0:
        return c
    # fallback lexical overlap
    return jaccard(a, b)
