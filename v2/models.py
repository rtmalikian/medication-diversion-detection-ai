"""Chart model — the internal representation every layer scores from.

Deliberately EXCLUDED from this model (never scoring inputs): race,
ethnicity, sex, preferred language, disability status, and insurance type
on its own. A test asserts the scorer's output is unchanged when those
attributes are attached to the input payload.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Medication:
    name: str
    drug_class: str = "other"  # opioid | benzodiazepine | moud | other
    daily_mg: float | None = None          # total mg per day, if an analgesic opioid
    prescriber_id: str | None = None
    pharmacy_id: str | None = None
    start_date: str | None = None          # ISO date
    end_date: str | None = None            # ISO date or None = ongoing
    dispense_dates: list[str] = field(default_factory=list)  # ISO dates
    days_supply: int | None = None
    source: str = "fhir"                   # fhir | manual | pdmp-import


@dataclass
class Chart:
    patient_ref: str = "anonymous"
    medications: list[Medication] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    narrative: str = ""                    # free-text chart notes to AI-review
    # Clinician-entered published-instrument totals (instruments themselves
    # are administered/collected outside this tool; totals only)
    ort_total: int | None = None
    comm_total: int | None = None
    # Clinician-observed encounter features for Layer C (expert/lived-experience
    # informed, Tier 4, UNVALIDATED). Keys are defined in scoring_config.json.
    encounter_features: dict[str, Any] = field(default_factory=dict)
    documented_prior_diversion: bool = False  # clinician-entered documented prior incident (original SAMHSA flag, kept as Layer A)
    sud_history_documented: bool = False       # support-routing context only; NEVER scored (see SCORING.md)
    has_pdmp_data: bool = False
    has_fhir_data: bool = False
    interpreted_encounter: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Keys that must never influence scoring. The API strips them on intake and
# the scorer never reads them; they exist here only so tests can prove that.
PROTECTED_KEYS = {
    "race", "ethnicity", "sex", "gender", "preferred_language", "language",
    "disability", "disability_status", "insurance_type", "religion",
    "national_origin", "sexual_orientation",
}


def chart_from_dict(data: dict) -> Chart:
    data = {k: v for k, v in (data or {}).items() if k not in PROTECTED_KEYS}
    meds = [Medication(**{k: v for k, v in m.items() if k in Medication.__dataclass_fields__})
            for m in data.pop("medications", [])]
    fields = {k: v for k, v in data.items() if k in Chart.__dataclass_fields__}
    return Chart(medications=meds, **fields)


# Contexts in which the CDC 2022 guideline states its recommendations do NOT
# apply (cancer-related pain treatment, sickle cell disease, palliative and
# end-of-life care). Dose signals in these contexts are recorded as
# context-noted, not scored — see SCORING.md.
GUIDELINE_EXEMPT_CONTEXTS = ("cancer", "palliative", "end-of-life", "end of life",
                             "sickle cell", "hospice")


def has_guideline_exempt_context(chart: Chart) -> str | None:
    blob = " ".join(chart.conditions).lower()
    for term in GUIDELINE_EXEMPT_CONTEXTS:
        if term in blob:
            return term
    return None
