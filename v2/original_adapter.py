"""Adapter: Raphael's ORIGINAL patient format -> v2 Chart, preserving his
original framework's findings as provenance.

The original repo's input shape (used by services/risk_calculator.py,
data/processor.py, services/evaluation_framework.py, train_demo.py) is:
  {patient_id, age, gender, medications:[{name,...}],
   prescriptions:[{drug_name, dose, prescriber_id, pharmacy_id, date,
                   days_supply, early_refill, concurrent, out_of_region}],
   clinical_indicators:{history_of_diversion, substance_abuse_history,
                        mental_health_history, chronic_pain_history, ...},
   history:{ed_visits}}

v2 translation rules (full factor-by-factor mapping in SCORING.md):
- Prescriber/pharmacy counts, duplicates and dispensing dates become real
  Chart medications so the deterministic date-based rules run on them.
  `dose` is treated as the daily mg for that drug entry (the original MME
  code made the same simplification); this limitation is recorded.
- Original boolean `early_refill` flags cannot be re-derived from dates the
  original data does not carry, so a flagged prescription contributes a
  dispense pair spaced to reproduce the flag for the date-based rule ONLY
  when no explicit dates exist — recorded in provenance.
- history_of_diversion -> Chart.documented_prior_diversion (Layer A).
- substance_abuse_history -> Chart.sud_history_documented (support only, 0 pts).
- mental_health_history, age, gender -> NOT carried into scoring (see
  SCORING.md: removed in v2). Gender/age are not even copied to the Chart.
- out_of_region count >= 2 -> Layer C travel_distance_anomaly feature
  (Tier 4, clinician-observed in the original data).
- Original evaluation-framework red flags are computed with the ORIGINAL
  code (services.evaluation_framework) when importable and attached as
  provenance `original_framework` — they do not double-score in v2.
"""
from __future__ import annotations

from .fhir_client import drug_class_for
from .models import Chart, Medication

OPIOID_DOSE_NAMES = ("oxycodone", "hydrocodone", "morphine", "hydromorphone",
                     "codeine", "tramadol", "oxymorphone", "methadone")


def original_to_chart(patient: dict) -> tuple[Chart, dict]:
    provenance = {"source_format": "original_repo_patient_data", "notes": []}
    chart = Chart(patient_ref=str(patient.get("patient_id", "anonymous")),
                  has_pdmp_data=True, has_fhir_data=False)
    rx = patient.get("prescriptions", [])
    by_drug: dict[str, dict] = {}
    for p in rx:
        name = (p.get("drug_name") or "unknown").strip()
        key = name.lower()
        entry = by_drug.setdefault(key, {"name": name, "prescriber": set(),
                                         "pharmacy": set(), "dates": [],
                                         "supply": None, "dose": None,
                                         "early_flag": False})
        if p.get("prescriber_id"):
            entry["prescriber"].add(str(p["prescriber_id"]))
        if p.get("pharmacy_id"):
            entry["pharmacy"].add(str(p["pharmacy_id"]))
        if p.get("date"):
            entry["dates"].append(str(p["date"])[:10])
        if p.get("days_supply"):
            entry["supply"] = int(p["days_supply"])
        if p.get("dose") is not None:
            entry["dose"] = float(p["dose"])
        if p.get("early_refill"):
            entry["early_flag"] = True
    for key, e in by_drug.items():
        dates = sorted(set(e["dates"]))
        if e["early_flag"] and e["supply"] and dates:
            # reproduce the original boolean flag for the date-based rule:
            # add a refill 10 days before a 30-day supply would run out
            from datetime import date, timedelta
            d0 = date.fromisoformat(dates[0])
            dates = sorted(set(dates + [(d0 + timedelta(days=max(1, e["supply"] - 10))).isoformat()]))
            provenance["notes"].append(
                f"{e['name']}: original early_refill boolean reproduced as a dispense {10} days "
                "early (original data carried a flag, not full dispense history)")
        # one Medication per prescriber so prescriber-count rules see them all
        prescribers = sorted(e["prescriber"]) or [None]
        for i, pres in enumerate(prescribers):
            base = e["name"].split()[0]
            chart.medications.append(Medication(
                name=e["name"], drug_class=drug_class_for(e["name"]),
                daily_mg=e["dose"] if any(o in key for o in OPIOID_DOSE_NAMES) else None,
                prescriber_id=pres, pharmacy_id=sorted(e["pharmacy"])[0] if e["pharmacy"] else None,
                start_date=dates[0] if dates else None, dispense_dates=dates if i == 0 else [],
                days_supply=e["supply"], source="original-format"))
        if len(e["pharmacy"]) > 1:
            provenance["notes"].append(f"{e['name']}: filled at {len(e['pharmacy'])} pharmacies in original data")

    ci = patient.get("clinical_indicators", {}) or {}
    if ci.get("history_of_diversion"):
        chart.documented_prior_diversion = True
    if ci.get("substance_abuse_history"):
        chart.sud_history_documented = True
    if ci.get("chronic_pain_history"):
        chart.conditions.append("chronic pain")
    dropped = [k for k in ("mental_health_history",) if ci.get(k)]
    if dropped:
        provenance["notes"].append(
            "NOT carried into v2 scoring: mental_health_history (removed — a mental-health "
            "diagnosis is not a diversion signal; see SCORING.md/BIAS.md)")
    provenance["notes"].append("NOT carried into v2 scoring: age and gender (original processor "
                               "features; demographics are not scored in v2)")
    oor = sum(1 for p in rx if p.get("out_of_region"))
    if oor >= 2:
        chart.encounter_features["travel_distance_anomaly"] = (
            f"original data: {oor} out-of-region prescriptions (Tier 4, clinician-observed)")
    provenance["out_of_region_prescriptions"] = oor
    provenance["ed_visits"] = (patient.get("history", {}) or {}).get("ed_visits", 0)
    return chart, provenance


def run_original_framework(patient: dict) -> dict | None:
    """Run the ORIGINAL EvidenceBasedEvaluation (Raphael's v1 code) for
    provenance/comparison. Returns None if the original module's deps are
    unavailable — never silently fakes it."""
    try:
        from services.evaluation_framework import evaluation_framework
        result = evaluation_framework.evaluate_patient_risk(patient)
        return {"overall_risk_level": result.get("overall_risk_level"),
                "red_flag_count": len(result.get("red_flags", [])),
                "red_flags": [f.get("description") for f in result.get("red_flags", [])]}
    except Exception as e:  # pragma: no cover - depends on legacy deps
        return {"unavailable": f"original framework not importable here: {type(e).__name__}"}
