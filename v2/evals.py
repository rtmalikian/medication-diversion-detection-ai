"""Synthetic eval battery for v2 — >=30 labelled cases.

Labels are PLANTED signals per layer. Metrics measure pipeline correctness
(did the deterministic rules / verified-quote extractor / MRD escalator find
what was planted, and stay quiet on benign look-alikes) — they are NOT
clinical accuracy. Sensitivity/specificity on real patients remain UNKNOWN.

Benign look-alikes are first-class cases: cancer dose escalation
(guideline-exempt context), stable MOUD, an interpreted-encounter
"inconsistency", a single poor-historian contradiction, random-direction
mistakes. Their false-flag rate is reported, not hidden.
"""
from __future__ import annotations

from .contradictions import Contradiction
from .models import Chart, Medication
from .scoring import load_config, score_chart
from .chart_review import review_chart
import asyncio


def med(name, cls, mg=None, pres=None, pharm=None, start=None, disp=(), supply=None, end=None):
    return Medication(name=name, drug_class=cls, daily_mg=mg, prescriber_id=pres,
                      pharmacy_id=pharm, start_date=start, dispense_dates=list(disp),
                      days_supply=supply, end_date=end, source="eval")


def base_chart(**kw):
    kw.setdefault("has_pdmp_data", True)
    kw.setdefault("has_fhir_data", True)
    return Chart(**kw)


def contra(quote="I take three a day", fact="chart: 2 per day", materiality="material",
           direction="favours_more_controlled", encounter="e1", confirmed=True, **kw):
    return Contradiction(statement_quote=quote, chart_fact=fact, materiality=materiality,
                         direction=direction, encounter_id=encounter, confirmed=confirmed, **kw)


CASES: list[dict] = []


def case(name, chart, expected_a=(), expected_b=(), expected_c=(), benign=False,
         contradictions=(), confirmed_conversation=()):
    CASES.append({"name": name, "chart": chart, "expected_a": set(expected_a),
                  "expected_b": set(expected_b), "expected_c": set(expected_c),
                  "benign": benign, "contradictions": list(contradictions),
                  "confirmed_conversation": list(confirmed_conversation)})


# ---------- true-pattern cases (Layer A) ----------
case("a_multi_prescriber", base_chart(medications=[
    med("Oxycodone", "opioid", 10, "Dr A"), med("Oxycodone", "opioid", 10, "Dr B"),
    med("Hydrocodone", "opioid", 10, "Dr C")]), expected_a={"multiple_prescribers"})
case("a_multi_pharmacy", base_chart(medications=[
    med("Hydrocodone", "opioid", 10, "Dr A", "Pharm 1"), med("Hydrocodone", "opioid", 10, "Dr A", "Pharm 2"),
    med("Hydrocodone", "opioid", 10, "Dr A", "Pharm 3")]), expected_a={"multiple_pharmacies"})
case("a_overlap", base_chart(medications=[
    med("Oxycodone", "opioid", 20, "Dr A", start="2026-08-01", end="2026-09-30"),
    med("Lorazepam", "benzodiazepine", 1, "Dr A", start="2026-08-15", end="2026-09-15")]),
    expected_a={"opioid_benzo_overlap"})
case("a_early_refill", base_chart(medications=[
    med("Oxycodone", "opioid", 20, "Dr A", disp=("2026-08-01", "2026-08-20"), supply=30)]),
    expected_a={"early_refill"})
case("a_mme50", base_chart(medications=[med("Oxycodone", "opioid", 40, "Dr A")]),
     expected_a={"mme_ge_50"})  # 40mg x 1.5 = 60 MME
case("a_mme90", base_chart(medications=[med("Morphine", "opioid", 100, "Dr A")]),
     expected_a={"mme_ge_90"})
case("a_escalation", base_chart(medications=[
    med("Hydrocodone", "opioid", 10, "Dr A", start="2026-06-01"),
    med("Hydrocodone", "opioid", 25, "Dr A", start="2026-09-01")]), expected_a={"mme_escalation"})
case("a_ort", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")], ort_total=11),
     expected_a={"ort_high"})
case("a_comm", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")], comm_total=15),
     expected_a={"comm_elevated"})
case("a_prior_diversion", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     documented_prior_diversion=True), expected_a={"documented_prior_diversion"})
case("a_combo_high", base_chart(medications=[
    med("Oxycodone", "opioid", 80, "Dr A", start="2026-08-01", end="2026-09-30", disp=("2026-08-01", "2026-08-19"), supply=30),
    med("Oxycodone", "opioid", 80, "Dr B"), med("Oxycodone", "opioid", 80, "Dr C"),
    med("Diazepam", "benzodiazepine", 5, "Dr B", start="2026-08-10", end="2026-09-10")]),
    expected_a={"multiple_prescribers", "opioid_benzo_overlap", "early_refill", "mme_ge_90"})

# ---------- Layer B (narrative, stub extractor) ----------
case("b_lost", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     narrative="Patient reports he lost my medication on a trip and asks what to do next."),
     expected_b={"chart_lost_medication_report"})
case("b_outside", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     narrative="Note states the patient also receives oxycodone from an urgent care clinic across town."),
     expected_b={"chart_outside_prescriber_mention"})
case("b_uds", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     narrative="Urine drug screen was negative for the prescribed opioid this month."),
     expected_b={"chart_uds_inconsistency_documented"})
case("b_two_flags", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     narrative="He is requesting an early refill again this month. Separately he reports buying tablets from a coworker last week."),
     expected_b={"chart_early_refill_request_narrative", "chart_other_source_mention"})

# ---------- Layer C ----------
case("c_travel_cash", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     encounter_features={"travel_distance_anomaly": "drives 180 miles each way (clinician-observed)",
                         "cash_pay_insistence_with_insurance": True, "has_insurance": True}),
     expected_c={"travel_distance_anomaly", "cash_pay_insistence_with_insurance"})
case("c_declines", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     encounter_features={"declines_uds_or_pdmp_consent": True,
                         "declines_noncontrolled_alternatives": True}),
     expected_c={"declines_uds_or_pdmp_consent", "declines_noncontrolled_alternatives"})
case("c_confirmed_conversation", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     expected_c={"conversation_admission_sharing_selling"},
     confirmed_conversation=[{"type": "admission_sharing_selling", "confirmed": True,
                              "verbatim_quote": "I gave some of my pills to my brother"}])
case("c_mrd_repeated_directional", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     expected_c={"history_chart_contradiction_mrd"},
     contradictions=[contra(encounter="e1"), contra(encounter="e2"), contra(encounter="e3")])

# ---------- benign look-alikes ----------
case("benign_stable", base_chart(medications=[
    med("Oxycodone", "opioid", 10, "Dr A", "Pharm 1", disp=("2026-08-01", "2026-08-31"), supply=30)]), benign=True)
case("benign_cancer_escalation", base_chart(conditions=["Cancer-related pain"],
     medications=[med("Morphine", "opioid", 30, "Dr A", start="2026-06-01"),
                  med("Morphine", "opioid", 120, "Dr A", start="2026-09-01")]), benign=True)
case("benign_moud_stable", base_chart(medications=[
    med("Buprenorphine", "moud", None, "Dr A", "Pharm 1", disp=("2026-08-01", "2026-08-29"), supply=28)],
    sud_history_documented=True), benign=True)
case("benign_benzo_tapered_off", base_chart(medications=[
    med("Oxycodone", "opioid", 10, "Dr A", start="2026-08-01", end="2026-09-30"),
    med("Lorazepam", "benzodiazepine", 1, "Dr A", start="2026-05-01", end="2026-06-30")]), benign=True)
case("benign_cash_no_insurance", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     encounter_features={"cash_pay_insistence_with_insurance": True, "has_insurance": False}), benign=True)
case("benign_single_minor_contradiction", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     benign=True, contradictions=[contra(materiality="minor", direction="neutral")])
case("benign_poor_historian_single", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     benign=True, contradictions=[contra()])
case("benign_random_direction_mistakes", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     benign=True, contradictions=[contra(encounter="e1"),
                                  contra(encounter="e2", direction="favours_less_controlled"),
                                  contra(encounter="e3", direction="neutral")])
case("benign_chart_error", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")]),
     benign=True, contradictions=[contra(chart_possibly_wrong=True)])
case("benign_interpreted_inconsistency", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     interpreted_encounter=True), benign=True,
     contradictions=[contra(interpreted_encounter=True, encounter="e1"),
                     contra(interpreted_encounter=True, encounter="e2")])
case("benign_rural_travel_only_if_unconfirmed", base_chart(medications=[med("Oxycodone", "opioid", 10, "Dr A")],
     narrative="Patient lives rurally and attends the nearest clinic. Visit unremarkable."), benign=True)
case("benign_two_pharmacies", base_chart(medications=[
    med("Oxycodone", "opioid", 10, "Dr A", "Pharm 1"), med("Oxycodone", "opioid", 10, "Dr A", "Pharm 2")]), benign=True)


def run_eval() -> dict:
    cfg = load_config()
    per_layer = {L: {"tp": 0, "fp": 0, "fn": 0} for L in "ABC"}
    benign_total = benign_flagged = 0
    rows = []
    for c in CASES:
        async def _review():
            if c["chart"].narrative:
                return await review_chart(c["chart"].narrative)
            return {"flags": [], "proposed": 0, "dropped": 0}
        review = asyncio.get_event_loop().run_until_complete(_review()) \
            if False else asyncio.run(_review())
        result = score_chart(c["chart"], cfg, layer_b_flags=review["flags"],
                             confirmed_conversation=c["confirmed_conversation"],
                             contradictions=c["contradictions"])
        got = {"A": set(), "B": set(), "C": set()}
        for item in result["items"]:
            if item["points"] > 0:
                got[item["layer"]].add(item["id"])
        for L, exp_key in (("A", "expected_a"), ("B", "expected_b"), ("C", "expected_c")):
            exp = c[exp_key]
            per_layer[L]["tp"] += len(exp & got[L])
            per_layer[L]["fp"] += len(got[L] - exp)
            per_layer[L]["fn"] += len(exp - got[L])
        if c["benign"]:
            benign_total += 1
            flagged = result["zone"] in ("elevated", "high")
            benign_flagged += 1 if flagged else 0
        rows.append({"case": c["name"], "zone": result["zone"], "total": result["total"],
                     "subscores": result["subscores"]})
    metrics = {}
    for L, d in per_layer.items():
        p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 1.0
        r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 1.0
        metrics[L] = {**d, "precision": round(p, 3), "recall": round(r, 3)}
    return {"n_cases": len(CASES), "per_layer": metrics,
            "benign_cases": benign_total, "benign_flagged_elevated_or_high": benign_flagged,
            "benign_false_flag_rate": round(benign_flagged / benign_total, 3) if benign_total else 0,
            "rows": rows,
            "label": "SYNTHETIC planted-signal battery — pipeline correctness only, NOT clinical accuracy"}
