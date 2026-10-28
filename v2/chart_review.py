"""Layer B — AI chart review with verified generation.

The gateway LLM is asked to extract red-flag OBSERVATIONS from the chart
narrative, each with a verbatim quote. A deterministic verifier then checks
every quote against the source text; unverified flags are dropped (and
counted in the result so the hallucination rate is visible, not hidden).

The prompt forbids conclusions about the person: it extracts observations
only, in pattern language. No flag may describe intent or character.
"""
from __future__ import annotations

import json

from . import gateway
from .verify import verify_quote

PROMPT_TEMPLATE = """You are an extraction tool for a clinician-review queue.
Extract ONLY concrete, checkable observations from the chart narrative that
match these signal types: lost_medication_report, early_refill_request_narrative,
outside_prescriber_mention, uds_inconsistency_documented,
dose_escalation_request_documented, other_source_mention.

Rules:
- For every observation return the EXACT verbatim quote from the chart.
- Never infer intent, honesty, or character. Observations only.
- If nothing matches, return an empty list.
Return JSON: {{"flags": [{{"signal": "...", "verbatim_quote": "...", "chart_location": "narrative"}}]}}

<chart>
{chart}
</chart>"""


async def review_chart(narrative: str, chat_fn=None) -> dict:
    """Returns {flags: [verified flags], proposed: n, dropped: n}."""
    if not narrative or not narrative.strip():
        return {"flags": [], "proposed": 0, "dropped": 0}
    if chat_fn is None:
        resp = await gateway.chat_completion(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(chart=narrative)}])
        content = resp["choices"][0]["message"]["content"]
    else:
        content = await chat_fn(narrative)
    try:
        parsed = json.loads(content)
        raw_flags = parsed.get("flags", [])
    except (json.JSONDecodeError, AttributeError):
        return {"flags": [], "proposed": 0, "dropped": 0, "parse_error": True}
    verified, dropped = [], 0
    for f in raw_flags:
        quote = (f or {}).get("verbatim_quote", "")
        if verify_quote(quote, narrative):
            verified.append({"signal": f.get("signal", "unknown"),
                             "verbatim_quote": quote,
                             "chart_location": f.get("chart_location", "narrative"),
                             "layer": "B", "verified": True})
        else:
            dropped += 1
    return {"flags": verified, "proposed": len(raw_flags), "dropped": dropped}
