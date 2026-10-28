"""The contradiction mechanism — Materiality x Repetition x Directionality.

Raphael's rule (2026-10-28): a history that contradicts objective chart data
is a real signal, but it must NOT carry the same weight as objective factors,
because patients have the right to make mistakes and to be poor historians.

Design (named in SCORING.md and README as the MRD Escalator):

  MATERIALITY — only contradictions about facts relevant to
    controlled-substance risk can ever score. A wrong appointment date or a
    misremembered pharmacy name is minor: informational forever, 0 points.
  REPETITION — a single material contradiction is an informational flag
    only: 0 points, and it can never by itself move a case up a zone.
    Points require the same kind of material contradiction repeated across
    >= 2 SEPARATE encounters.
  DIRECTIONALITY — honest mistakes point both ways. If repeated material
    contradictions consistently favour obtaining more controlled substance,
    each qualifying repetition beyond the first earns points. If they point
    randomly (favouring and non-favouring mixed, or mostly non-favouring),
    they earn a token amount — far less — because that pattern is what
    ordinary poor recall looks like.

Confirmation + chart-error handling:
  - Nothing scores unconfirmed: a clinician must confirm the contradiction
    (both sides of it) before any points.
  - The flag targets the DISCREPANCY, not the patient. If the clinician
    marks the chart as possibly wrong, the item becomes a chart-correction
    request and scores 0 — record error is one of the innocent explanations
    and sometimes the record, not the history, is what is wrong.
  - Every output presents innocent explanations alongside the flag: poor
    recall, interpretation/language effects, timing differences, and record
    or chart error.

Caps: contradiction points are capped at 12 — below the single objective
Layer A opioid+benzodiazepine overlap item (15) — and they live inside the
Layer C cap (15), so contradictions plus all other Tier 4 items together
still cannot reach the high zone without objective Layer A/B support.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INNOCENT_EXPLANATIONS = [
    "poor recall — patients have the right to make mistakes and to be poor historians",
    "interpretation or language effects, especially in interpreted encounters",
    "timing differences (the statement and the record describing different days)",
    "record or chart error — the chart may be wrong; treat this as a discrepancy to resolve, not a fault to assign",
]

DIRECTIONS = ("favours_more_controlled", "favours_less_controlled", "neutral")

POINTS_PER_QUALIFYING_REPETITION = 4     # material + consistently favouring
POINTS_PER_RANDOM_REPETITION = 1         # material + random/mixed direction
CONTRADICTION_CAP = 12


@dataclass
class Contradiction:
    statement_quote: str                    # what the patient said, verbatim
    chart_fact: str                         # the objective record it conflicts with
    chart_source: str = ""                  # e.g. "MedicationDispense 2026-09-14, Pharmacy A"
    materiality: str = "minor"              # material | minor
    direction: str = "neutral"              # see DIRECTIONS
    encounter_id: str = ""                  # contradictions must span encounters to score
    confirmed: bool = False                 # clinician-confirmed (both sides reviewed)
    chart_possibly_wrong: bool = False      # clinician marks record error as plausible
    interpreted_encounter: bool = False


def evaluate_contradictions(contradictions: list[Contradiction]) -> dict:
    """Returns {points, scored_items, informational, chart_corrections, analysis}.

    informational entries are always produced for every contradiction —
    including the ones that score — because the clinician should see the
    discrepancy even when it earns nothing.
    """
    informational, chart_corrections, scored_items = [], [], []
    for c in contradictions or []:
        entry = {
            "statement_quote": c.statement_quote,
            "chart_fact": c.chart_fact,
            "chart_source": c.chart_source,
            "materiality": c.materiality,
            "direction": c.direction,
            "encounter_id": c.encounter_id,
            "confirmed": c.confirmed,
            "innocent_explanations": INNOCENT_EXPLANATIONS,
            "targets": "the discrepancy, not the patient",
        }
        if c.chart_possibly_wrong:
            entry["disposition"] = "chart_correction_requested — scores 0; resolve the record first"
            chart_corrections.append(entry)
        elif not c.confirmed:
            entry["disposition"] = "informational only — unconfirmed contradictions never score"
            informational.append(entry)
        elif c.materiality != "material":
            entry["disposition"] = "informational only — minor contradictions never score (patients misremember details)"
            informational.append(entry)
        elif c.interpreted_encounter:
            entry["disposition"] = "informational only — interpreted encounter; possible interpretation artefact"
            informational.append(entry)
        else:
            entry["disposition"] = "candidate for repetition scoring"
            informational.append(entry)

    # --- Repetition + Directionality over qualifying contradictions ---
    qualifying = [c for c in (contradictions or [])
                  if c.confirmed and c.materiality == "material"
                  and not c.chart_possibly_wrong and not c.interpreted_encounter]
    encounters = {c.encounter_id for c in qualifying if c.encounter_id}
    points = 0
    pattern = "none"
    if len(qualifying) >= 2 and len(encounters) >= 2:
        favouring = sum(1 for c in qualifying if c.direction == "favours_more_controlled")
        non_favouring = len(qualifying) - favouring
        repetitions = len(qualifying) - 1  # first instance never scores
        if favouring > non_favouring:
            pattern = "consistent_direction_favouring_more_controlled"
            points = min(CONTRADICTION_CAP, repetitions * POINTS_PER_QUALIFYING_REPETITION)
        else:
            pattern = "random_direction — consistent with ordinary poor recall"
            points = min(3, repetitions * POINTS_PER_RANDOM_REPETITION)
        if points > 0:
            scored_items.append({
                "id": "history_chart_contradiction_mrd",
                "layer": "C",
                "points": points,
                "evidence": (f"{len(qualifying)} clinician-confirmed material contradictions across "
                             f"{len(encounters)} encounters; direction pattern: {pattern}. "
                             f"First instance scored 0 by design (MRD Escalator)"),
                "label": "history-chart contradiction — MRD Escalator (Materiality x Repetition x "
                         "Directionality); Tier 4, UNVALIDATED; capped below objective Layer A factors",
                "innocent_explanations": INNOCENT_EXPLANATIONS,
            })
    analysis = {
        "total_received": len(contradictions or []),
        "qualifying_for_scoring": len(qualifying),
        "distinct_encounters": len(encounters),
        "direction_pattern": pattern,
        "points": points,
        "cap": CONTRADICTION_CAP,
        "rule": "single instance = informational, 0 points, never moves a zone; "
                "minor = never scores; chart-error = correction request, 0 points",
    }
    return {"points": points, "scored_items": scored_items,
            "informational": informational, "chart_corrections": chart_corrections,
            "analysis": analysis}
