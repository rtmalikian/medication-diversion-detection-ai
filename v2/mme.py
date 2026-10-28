"""Morphine Milligram Equivalent (MME) calculation.

Conversion factors are the CDC factors as published in the CDC fact sheet
"Calculating Total Daily Dose of Opioids for Safer Dosage" (CDC Stacks
cdc_38481) and the CDC Clinical Practice Guideline for Prescribing Opioids
for Pain — United States, 2022 (MMWR 2022;71(RR-3):1-95, Table 1).

Limits, stated in the source and honoured here:
- These are estimates; they cannot account for genetics/pharmacokinetics.
- Calculated MME must NOT be used to convert a patient between opioids.
- CDC 2022 explicitly cautions against using MME thresholds as hard cutoffs
  for care decisions; here MME is a risk-conversation signal only.
- Buprenorphine prescribed for opioid use disorder is excluded (CDC MME
  resources do not assign it a factor for this calculation; partial agonist).
- Drugs without a verified factor in the source table return None and are
  reported as "unknown" rather than silently treated as zero risk.
"""
from __future__ import annotations

# CDC conversion factors (see module docstring). Methadone is dose-dependent.
MME_FACTORS: dict[str, float] = {
    "codeine": 0.15,
    "hydrocodone": 1.0,
    "hydromorphone": 4.0,
    "morphine": 1.0,
    "oxycodone": 1.5,
    "oxymorphone": 3.0,
}

# Explicitly excluded from MME totals in this tool, with reason.
MME_EXCLUDED: dict[str, str] = {
    "buprenorphine": "excluded: MOUD / partial agonist; CDC MME calculation does not assign a factor",
    "naloxone": "excluded: not an analgesic opioid",
}


def methadone_factor(daily_mg: float) -> float:
    """CDC sliding scale for methadone."""
    if daily_mg <= 20:
        return 4.0
    if daily_mg <= 40:
        return 8.0
    if daily_mg <= 60:
        return 10.0
    return 12.0


def mme_for_drug(drug_name: str, daily_mg: float) -> float | None:
    """MME/day for one drug, or None if the factor is unknown/excluded."""
    key = (drug_name or "").strip().lower()
    # tolerate combination product names like "oxycodone-acetaminophen"
    for known in list(MME_FACTORS) + ["methadone"] + list(MME_EXCLUDED):
        if known in key:
            key = known
            break
    if key in MME_EXCLUDED:
        return None
    if key == "methadone":
        return daily_mg * methadone_factor(daily_mg)
    factor = MME_FACTORS.get(key)
    if factor is None:
        return None
    return daily_mg * factor


def conversion_factor(drug_name: str, daily_mg: float) -> float | None:
    key = (drug_name or "").strip().lower()
    for known in list(MME_FACTORS) + ["methadone"]:
        if known in key:
            key = known
            break
    if key == "methadone":
        return methadone_factor(daily_mg)
    return MME_FACTORS.get(key)
