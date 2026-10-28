"""Layer A — deterministic rules over structured (FHIR/PDMP) chart data.

Every rule returns an item with machine-readable evidence so a clinician can
audit the arithmetic. Rules never see narrative text and never see
protected attributes (the Chart model does not carry them).
"""
from __future__ import annotations

from datetime import date

from .models import Chart, Medication, has_guideline_exempt_context
from .mme import mme_for_drug, MME_EXCLUDED


def _d(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _window(m: Medication) -> tuple[date | None, date | None]:
    start = _d(m.start_date)
    end = _d(m.end_date)
    if end is None and m.dispense_dates:
        last = max(_d(x) for x in m.dispense_dates if _d(x))
        if m.days_supply:
            from datetime import timedelta
            end = last + timedelta(days=m.days_supply)
    return start, end


def overlap_days(a: Medication, b: Medication) -> int:
    a0, a1 = _window(a)
    b0, b1 = _window(b)
    if None in (a0, a1, b0, b1):
        return 0
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0, (hi - lo).days + 1)


def daily_mme(m: Medication) -> float | None:
    if m.drug_class != "opioid" or m.daily_mg is None:
        return None
    return mme_for_drug(m.name, m.daily_mg)


def total_daily_mme(chart: Chart) -> tuple[float, list[str]]:
    total, unknown = 0.0, []
    for m in chart.medications:
        if m.drug_class != "opioid":
            continue
        if m.name.strip().lower() in MME_EXCLUDED or "buprenorphine" in m.name.lower():
            continue
        v = daily_mme(m)
        if v is None:
            if m.daily_mg is not None:
                unknown.append(m.name)
            continue
        total += v
    return total, unknown


def run_layer_a(chart: Chart, cfg: dict) -> tuple[list[dict], dict]:
    """Returns (items, details). Items: {id, layer, points, evidence}."""
    a = cfg["layer_a"]
    items: list[dict] = []
    details: dict = {}
    controlled = [m for m in chart.medications if m.drug_class in ("opioid", "benzodiazepine")]

    prescribers = {m.prescriber_id for m in controlled if m.prescriber_id}
    if len(prescribers) >= a["multiple_prescribers"]["threshold_prescribers"]:
        items.append({"id": "multiple_prescribers", "layer": "A",
                      "points": a["multiple_prescribers"]["points"],
                      "evidence": f"{len(prescribers)} distinct prescribers of controlled medications: {sorted(prescribers)}"})
    details["distinct_prescribers"] = len(prescribers)

    pharmacies = {m.pharmacy_id for m in chart.medications if m.pharmacy_id}
    if len(pharmacies) >= a["multiple_pharmacies"]["threshold_pharmacies"]:
        items.append({"id": "multiple_pharmacies", "layer": "A",
                      "points": a["multiple_pharmacies"]["points"],
                      "evidence": f"{len(pharmacies)} distinct dispensing pharmacies: {sorted(pharmacies)}"})
    details["distinct_pharmacies"] = len(pharmacies)

    opioids = [m for m in chart.medications if m.drug_class == "opioid"]
    benzos = [m for m in chart.medications if m.drug_class == "benzodiazepine"]
    max_overlap = 0
    pair = None
    for o in opioids:
        for b in benzos:
            d = overlap_days(o, b)
            if d > max_overlap:
                max_overlap, pair = d, (o.name, b.name)
    details["opioid_benzo_overlap_days"] = max_overlap
    if max_overlap >= a["opioid_benzo_overlap"]["min_overlap_days"]:
        items.append({"id": "opioid_benzo_overlap", "layer": "A",
                      "points": a["opioid_benzo_overlap"]["points"],
                      "evidence": f"{pair[0]} overlaps {pair[1]} for {max_overlap} day(s) — combination carries an FDA boxed warning (2016) and CDC 2022 Rec 11 advises avoiding co-prescribing whenever possible"})

    # early refills: same-drug dispenses arriving >= early_by_days before
    # the previous days_supply would have run out
    early_events = []
    by_name: dict[str, list[Medication]] = {}
    for m in chart.medications:
        by_name.setdefault(m.name.strip().lower(), []).append(m)
    for name, meds in by_name.items():
        dates = sorted(d for m in meds for d in (_d(x) for x in m.dispense_dates) if d)
        supply = next((m.days_supply for m in meds if m.days_supply), None)
        if not supply or len(dates) < 2:
            continue
        for prev, cur in zip(dates, dates[1:]):
            gap = (cur - prev).days
            if supply - gap >= a["early_refill"]["early_by_days"]:
                early_events.append(f"{name}: refill after {gap}d of a {supply}d supply ({supply - gap}d early) on {cur.isoformat()}")
    details["early_refill_events"] = early_events
    if early_events:
        items.append({"id": "early_refill", "layer": "A",
                      "points": a["early_refill"]["points"],
                      "evidence": "; ".join(early_events)})

    # MME level + trajectory (trajectory uses dispense-ordered daily_mg snapshots
    # encoded as separate Medication entries with start dates)
    mme, unknown = total_daily_mme(chart)
    details["total_daily_mme"] = round(mme, 1)
    details["mme_unknown_drugs"] = unknown
    exempt = has_guideline_exempt_context(chart)
    if exempt:
        details["guideline_exempt_context"] = exempt
    if mme >= 90:
        if exempt:
            details["mme_ge_90_context_noted"] = (
                f"total {mme:.0f} MME/day recorded but NOT scored: CDC 2022 guideline scope excludes {exempt} care contexts")
        else:
            items.append({"id": "mme_ge_90", "layer": "A", "points": a["mme_ge_90"]["points"],
                          "evidence": f"total daily dose {mme:.0f} MME/day (CDC conversion factors). A dose signal for clinician conversation — not a cutoff; CDC 2022 cautions against hard MME thresholds"})
    elif mme >= 50:
        if exempt:
            details["mme_ge_50_context_noted"] = (
                f"total {mme:.0f} MME/day recorded but NOT scored: CDC 2022 guideline scope excludes {exempt} care contexts")
        else:
            items.append({"id": "mme_ge_50", "layer": "A", "points": a["mme_ge_50"]["points"],
                          "evidence": f"total daily dose {mme:.0f} MME/day (CDC conversion factors). CDC 2022 Rec 8 lists >=50 MME/day among factors for offering naloxone — a support prompt, not an accusation"})
    # trajectory: compare earliest vs latest dated opioid entries of same drug
    esc = []
    for name, meds in by_name.items():
        dated = sorted((m for m in meds if m.start_date and m.daily_mg and m.drug_class == "opioid"),
                       key=lambda m: m.start_date)
        if len(dated) >= 2:
            first, last = dated[0], dated[-1]
            if first.daily_mg > 0 and (last.daily_mg - first.daily_mg) / first.daily_mg >= a["mme_escalation"]["min_increase_fraction"]:
                esc.append(f"{name}: {first.daily_mg:.0f} -> {last.daily_mg:.0f} mg/day")
    details["dose_escalations"] = esc
    if esc and not exempt:
        items.append({"id": "mme_escalation", "layer": "A", "points": a["mme_escalation"]["points"],
                      "evidence": "dose escalation >=50%: " + "; ".join(esc)})
    elif esc and exempt:
        details["mme_escalation_context_noted"] = "escalation recorded but NOT scored in guideline-exempt context: " + "; ".join(esc)

    if chart.ort_total is not None and chart.ort_total >= a["ort_high"]["threshold"]:
        items.append({"id": "ort_high", "layer": "A", "points": a["ort_high"]["points"],
                      "evidence": f"clinician-entered Opioid Risk Tool total {chart.ort_total} (>=8 = original high band, Webster & Webster 2005). External validation of the ORT is inconsistent — weigh accordingly"})
    if chart.comm_total is not None and chart.comm_total >= a["comm_elevated"]["threshold"]:
        items.append({"id": "comm_elevated", "layer": "A", "points": a["comm_elevated"]["points"],
                      "evidence": f"clinician-entered COMM total {chart.comm_total} (>=9 = original cut score, Butler et al. 2007). COMM is self-report and can under- or over-state behaviour"})
    return items, details
