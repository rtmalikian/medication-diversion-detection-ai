"""Verified generation: an AI flag scores ONLY if its quote is real.

verify_quote returns True when the (normalised) quote appears in the source
text exactly, or with a token-overlap fuzzy match >= 0.9 against a sliding
window of the source. Anything else is dropped and counted — hallucinated
quotes never score. This is the same discipline as Raphael's other
clinical products (Discharge Decoded, Prior-Auth Co-Pilot).
"""
from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", (s or "").lower())).strip()


def _tokens(s: str) -> list[str]:
    return _norm(s).split()


def verify_quote(quote: str, source: str, fuzzy_threshold: float = 0.9) -> bool:
    if not quote or not source:
        return False
    nq, ns = _norm(quote), _norm(source)
    if nq and nq in ns:
        return True
    qt = _tokens(quote)
    st = _tokens(source)
    if not qt or len(st) < len(qt):
        return len(qt) > 0 and " ".join(qt) in " ".join(st)
    qset = set(qt)
    n = len(qt)
    for i in range(0, len(st) - n + 1):
        window = st[i:i + n]
        overlap = sum(1 for t in window if t in qset) / n
        if overlap >= fuzzy_threshold:
            return True
    return False
