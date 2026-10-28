"""Scoring engine — three layers, always reported separately.

Layer A: deterministic published/guideline signals (app.rules).
Layer B: AI chart-review flags, verified quotes only (app.chart_review).
Layer C: expert / lived-experience informed items — Tier 4, UNVALIDATED.
         Capped, disable-able, and NEVER able on its own to produce the
         high zone (enforced here and in tests).

Output is pattern language for a clinician-review queue. It is not an
accusation, it never denies care, and sensitivity/specificity of the
composite are UNKNOWN (unvalidated). Support routing accompanies every
elevated/high output: offer a PDMP check, a clinician conversation,
treatment/referral resources, and — where dose/overlap factors are
present — a prompt to discuss naloxone per clinic policy and CDC 2022
Rec 8 (general guideline information, not a recommendation for a person).
"""
from __future__ import annotations

import json
import os

from .models import Chart
from . import rules
from .contradictions import Contradiction, evaluate_contradictions

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "scoring_config.json")


def load_config(path: str | None = None) -> dict:
    with open(path or _CONFIG_PATH) as f:
        return json.load(f)


# Layer C items a clinician may assert from encounter features. The two
# quote-based pressure items require a verbatim quote in the feature value.
LAYER_C_FEATURE_KEYS = {
    "pressure_tactics_quoted", "story_inconsistency_across_visits",
    "declines_noncontrolled_alternatives", "declines_uds_or_pdmp_consent",
    "travel_distance_anomaly", "cash_pay_insistence_with_insurance",
    "urgency_escalation_between_visits",
}

# Confirmed conversation signals (scribe) -> Layer C items.
CONVERSATION_TO_LAYER_C = {
    "inconsistent_with_chart": "conversation_inconsistent_with_chart",
    "threat_or_coercion": "conversation_threat_or_coercion",
    "contradicts_plan": "conversation_contradicts_plan",
    "admission_sharing_selling": "conversation_admission_sharing_selling",
    "third_party_pressure": "conversation_third_party_pressure",
}

RISK_LANGUAGE = (
    "Pattern language only: this output describes a pattern in recorded data. "
    "False positives happen — chronic pain, cancer care, recovery on MOUD, "
    "language interpretation, fragmented care across clinics, and data errors "
    "can all produce similar patterns in a person at no elevated risk. "
    "False negatives also happen — a lower score does not rule out misuse or "
    "diversion. Sensitivity and specificity of this composite score are "
    "UNKNOWN: it has not been clinically validated."
)

SUPPORT_ACTIONS_BASE = [
    "Route to a clinician for human review before any consequence — the score never acts alone",
    "Offer a PDMP check and a clinician conversation about the pattern",
    "Offer treatment and referral resources without conditioning care on the score",
]
SUPPORT_NALOXONE = ("Discuss naloxone availability per clinic policy — CDC 2022 Rec 8 lists "
                    "higher dosage (>=50 MME/day), concurrent benzodiazepine use, history of "
                    "overdose or substance use disorder among factors for offering naloxone "
                    "(general guideline information, not a recommendation for this individual)")


def layer_c_items(chart: Chart, cfg: dict, confirmed_conversation: list[dict] | None = None) -> list[dict]:
    c_cfg = cfg["layer_c"]
    if not c_cfg.get("enabled", True):
        return []
    pts = c_cfg["points"]
    items: list[dict] = []
    feats = chart.encounter_features or {}
    for key in sorted(LAYER_C_FEATURE_KEYS):
        val = feats.get(key)
        if not val:
            continue
        if key == "cash_pay_insistence_with_insurance" and not feats.get("has_insurance"):
            continue  # cash pay alone is never scored; only insistence despite coverage, clinician-observed
        evidence = val if isinstance(val, str) else "clinician-observed encounter feature"
        items.append({"id": key, "layer": "C", "points": pts[key], "evidence": evidence,
                      "label": "expert-opinion / lived-experience informed — Tier 4, UNVALIDATED"})
    for sig in confirmed_conversation or []:
        if not sig.get("confirmed"):
            continue  # unconfirmed conversation signals NEVER score
        if sig.get("interpreted_encounter"):
            continue  # interpreted-encounter signals held from scoring (interpretation artefacts)
        item_id = CONVERSATION_TO_LAYER_C.get(sig.get("type", ""))
        if item_id and pts.get(item_id):
            items.append({"id": item_id, "layer": "C", "points": pts[item_id],
                          "evidence": f"clinician-confirmed verbatim quote: \"{sig.get('verbatim_quote', '')}\"",
                          "label": "expert-opinion / lived-experience informed — Tier 4, UNVALIDATED; human-confirmed quote"})
    # cap
    total = sum(i["points"] for i in items)
    cap = c_cfg["cap"]
    if total > cap:
        scale_note = f"Layer C raw {total} capped at {cap}"
        excess = total - cap
        for i in reversed(items):
            if excess <= 0:
                break
            cut = min(i["points"], excess)
            i["points"] -= cut
            excess -= cut
        items = [i for i in items if i["points"] > 0]
        for i in items:
            i.setdefault("cap_note", scale_note)
    return items


def score_chart(chart: Chart, cfg: dict | None = None,
                layer_b_flags: list[dict] | None = None,
                confirmed_conversation: list[dict] | None = None,
                contradictions: list[Contradiction] | None = None,
                extra_a_items: list[dict] | None = None) -> dict:
    cfg = cfg or load_config()
    a_items, details = rules.run_layer_a(chart, cfg)
    if chart.documented_prior_diversion:
        a_items.append({"id": "documented_prior_diversion", "layer": "A",
                        "points": cfg["layer_a"]["documented_prior_diversion"]["points"],
                        "evidence": "clinician-entered DOCUMENTED prior diversion incident in the record "
                                    "(original repo SAMHSA flag, kept). A documented fact about the record, "
                                    "presented for clinician review — not a characterisation of the person"})
    for extra in extra_a_items or []:
        a_items.append(extra)
    a_points = sum(i["points"] for i in a_items)

    b_cfg = cfg["layer_b"]
    b_items = []
    for f in layer_b_flags or []:
        if not f.get("verified"):
            continue  # unverified AI flags never score — enforced again here
        b_items.append({"id": f"chart_{f.get('signal', 'unknown')}", "layer": "B",
                        "points": b_cfg["points_per_verified_flag"],
                        "evidence": f"verbatim chart quote: \"{f.get('verbatim_quote', '')}\"",
                        "signal": f.get("signal")})
    b_raw = sum(i["points"] for i in b_items)
    if b_raw > b_cfg["cap"]:
        b_items.append({"id": "layer_b_cap_note", "layer": "B", "points": b_cfg["cap"] - b_raw,
                        "evidence": f"Layer B raw {b_raw} capped at {b_cfg['cap']}"})
    b_points = min(b_raw, b_cfg["cap"])

    c_items = layer_c_items(chart, cfg, confirmed_conversation)
    # MRD Escalator: history-chart contradictions score inside the Layer C
    # cap, with their own sub-cap (12) below objective Layer A factors.
    contra = evaluate_contradictions(contradictions or [])
    c_items = c_items + contra["scored_items"]
    c_points = sum(i["points"] for i in c_items)
    c_cap = cfg["layer_c"]["cap"]
    if c_points > c_cap:
        excess = c_points - c_cap
        for i in reversed(c_items):
            if excess <= 0:
                break
            cut = min(i["points"], excess)
            i["points"] -= cut
            excess -= cut
        c_items = [i for i in c_items if i["points"] > 0]
        c_points = c_cap
    contra_points_scored = sum(i["points"] for i in c_items
                               if i["id"] == "history_chart_contradiction_mrd")

    total = a_points + b_points + c_points
    zones = cfg["zones"]
    sufficient = bool(chart.has_pdmp_data or chart.has_fhir_data
                      or chart.ort_total is not None or chart.comm_total is not None)
    if not sufficient:
        zone = "indeterminate"
        zone_reason = ("insufficient data: no PDMP/FHIR structured data and no clinician-entered "
                       "instrument totals — a score from narrative or encounter features alone "
                       "would not be meaningful")
    elif total >= zones["high_min"]:
        if a_points + b_points < zones["high_min"]:
            zone = "elevated"
            zone_reason = ("zone capped at elevated: Layer C (Tier 4, unvalidated) cannot by itself "
                           "push a case into the high zone")
        else:
            zone = "high"
            zone_reason = "total and Layer A+B both at or above the high threshold"
    elif total >= zones["elevated_min"]:
        zone = "elevated"
        zone_reason = "total at or above the elevated threshold"
    else:
        zone = "lower"
        zone_reason = "total below the elevated threshold"

    support = list(SUPPORT_ACTIONS_BASE)
    if chart.sud_history_documented:
        support.append("Documented substance-use history is present: offer treatment/referral resources "
                       "and naloxone discussion per clinic policy. This history is support context only "
                       "and earned ZERO score points (a diagnosis is not scored)")
    if contra["chart_corrections"]:
        support.append("Resolve the chart-correction request(s) listed under contradictions before "
                       "relying on the affected record entries")
    fired = {i["id"] for i in a_items}
    if fired & {"mme_ge_50", "mme_ge_90", "opioid_benzo_overlap", "ort_high", "comm_elevated"}:
        support.append(SUPPORT_NALOXONE)
    if chart.interpreted_encounter:
        support.append("Language access: this encounter used an interpreter — treat narrative and "
                       "conversation signals with extra caution; do not score interpretation artefacts")

    return {
        "scoring_version": cfg["version"],
        "patient_ref": chart.patient_ref,
        "total": total,
        "zone": zone,
        "zone_reason": zone_reason,
        "pattern_statement": (
            "Pattern consistent with elevated risk — requires clinician review"
            if zone in ("elevated", "high") else
            "No elevated pattern identified in the available data — this does not rule out risk"
            if zone == "lower" else
            "Indeterminate — insufficient structured data to score responsibly"),
        "subscores": {"layer_a_published_guideline": a_points,
                      "layer_b_ai_chart_review": b_points,
                      "layer_c_expert_lived_experience_unvalidated": c_points,
                      "layer_c_of_which_contradictions_mrd": contra_points_scored},
        "contradictions": {"analysis": contra["analysis"],
                           "informational": contra["informational"],
                           "chart_corrections": contra["chart_corrections"]},
        "items": a_items + b_items + c_items,
        "details": details,
        "support_actions": support,
        "risk_language": RISK_LANGUAGE,
        "sensitivity_specificity": "UNKNOWN — composite not clinically validated",
        "human_confirmation_required": True,
        "automated_action": "none — flagging and support routing only; never automated denial of care, accusation, or reporting",
    }
