"""FHIR R4 client + bundle mapper.

Pulls Patient, MedicationRequest, MedicationDispense, Encounter, Observation
and Condition from any FHIR R4 base URL and maps them into the internal
Chart model. Tested against a local mock FHIR server (httpx MockTransport)
and a bundled Synthea-format synthetic bundle — never against real patients.

Mapping notes / limits:
- Dose: dosageInstruction doseAndRate doseQuantity x timing.repeat
  (frequency per period) -> daily mg. If either is missing, daily_mg is None
  and MME rules simply do not fire for that drug (visible in details via
  mme_unknown_drugs when a name has no known factor).
- Drug class: inferred from the medication display text against small
  opioid / benzodiazepine / MOUD lexicons. Production use must replace this
  with RxNorm/ATC terminology binding — stated in ARCHITECTURE.md.
- Prescriber: MedicationRequest.requester reference/display; pharmacy:
  MedicationDispense.performer actor display.
"""
from __future__ import annotations

import httpx

from .models import Chart, Medication

OPIOID_NAMES = ("oxycodone", "hydrocodone", "morphine", "hydromorphone",
                "codeine", "tramadol", "oxymorphone", "methadone",
                "fentanyl", "tapentadol", "meperidine")
BENZO_NAMES = ("alprazolam", "lorazepam", "clonazepam", "diazepam",
               "temazepam", "triazolam", "chlordiazepoxide", "clorazepate")
MOUD_NAMES = ("buprenorphine", "naltrexone")


def drug_class_for(name: str) -> str:
    n = (name or "").lower()
    if any(x in n for x in MOUD_NAMES):
        return "moud"
    if any(x in n for x in BENZO_NAMES):
        return "benzodiazepine"
    if any(x in n for x in OPIOID_NAMES):
        return "opioid"
    return "other"


def _med_name(resource: dict) -> str:
    concept = resource.get("medicationCodeableConcept") or {}
    if concept.get("text"):
        return concept["text"]
    codings = concept.get("coding") or []
    return codings[0].get("display", "unknown") if codings else "unknown"


def _daily_mg(resource: dict) -> float | None:
    try:
        di = (resource.get("dosageInstruction") or [])[0]
        dose = (di.get("doseAndRate") or [])[0]["doseQuantity"]["value"]
        rep = (di.get("timing") or {}).get("repeat") or {}
        per_day = float(rep.get("frequency", 1)) / float(rep.get("period", 1))
        if (rep.get("periodUnit") or "d") != "d":
            return None
        return float(dose) * per_day
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _ref_id(ref: dict | None) -> str | None:
    if not ref:
        return None
    return ref.get("display") or (ref.get("reference") or "").split("/")[-1] or None


def bundle_to_chart(bundle: dict, patient_ref: str = "synthetic") -> Chart:
    entries = [e.get("resource", {}) for e in bundle.get("entry", [])]
    chart = Chart(patient_ref=patient_ref, has_fhir_data=True, has_pdmp_data=True)
    dispenses: dict[str, list[dict]] = {}
    for r in entries:
        if r.get("resourceType") == "MedicationDispense":
            for ar in r.get("authorizingPrescription") or []:
                key = (ar.get("reference") or "").split("/")[-1]
                dispenses.setdefault(key, []).append(r)
    for r in entries:
        rt = r.get("resourceType")
        if rt == "Condition":
            code = r.get("code") or {}
            label = code.get("text") or ((code.get("coding") or [{}])[0].get("display", ""))
            if label:
                chart.conditions.append(label)
        elif rt == "MedicationRequest":
            name = _med_name(r)
            med = Medication(
                name=name,
                drug_class=drug_class_for(name),
                daily_mg=_daily_mg(r),
                prescriber_id=_ref_id(r.get("requester")),
                start_date=(r.get("authoredOn") or "")[:10] or None,
                days_supply=None,
                source="fhir",
            )
            related = dispenses.get(r.get("id", ""), [])
            dates, supplies, pharmacies = [], [], set()
            for d in related:
                if d.get("whenHandedOver"):
                    dates.append(d["whenHandedOver"][:10])
                if d.get("daysSupply", {}).get("value"):
                    supplies.append(int(d["daysSupply"]["value"]))
                for p in d.get("performer") or []:
                    actor = _ref_id(p.get("actor"))
                    if actor:
                        pharmacies.add(actor)
            med.dispense_dates = sorted(dates)
            med.days_supply = supplies[0] if supplies else None
            med.pharmacy_id = sorted(pharmacies)[0] if pharmacies else None
            chart.medications.append(med)
        elif rt == "Observation":
            code = r.get("code") or {}
            label = (code.get("text") or "").lower()
            if "comm" in label and r.get("valueInteger") is not None:
                chart.comm_total = int(r["valueInteger"])
            if "opioid risk" in label and r.get("valueInteger") is not None:
                chart.ort_total = int(r["valueInteger"])
    return chart


class FhirClient:
    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None,
                 headers: dict | None = None, auth=None):
        """auth: optional SMART Backend Services token provider
        (v2.smart_backend.SmartBackendAuth). When given, fetch_chart()
        adds its bearer token to the request headers — this is how the
        client talks to Epic's sandbox (see EPIC_SANDBOX.md)."""
        self.base_url = base_url.rstrip("/")
        self.transport = transport
        self.headers = headers or {"Accept": "application/fhir+json"}
        self.auth = auth

    async def _search(self, client: httpx.AsyncClient, resource: str, patient_id: str) -> list[dict]:
        r = await client.get(f"{self.base_url}/{resource}",
                             params={"patient": patient_id, "_count": "200"})
        r.raise_for_status()
        data = r.json()
        return [e.get("resource", {}) for e in data.get("entry", [])]

    async def fetch_chart(self, patient_id: str) -> Chart:
        headers = dict(self.headers)
        if self.auth is not None:
            headers.update(await self.auth.auth_headers())
        async with httpx.AsyncClient(transport=self.transport, headers=headers,
                                     timeout=30) as client:
            bundle = {"resourceType": "Bundle", "entry": []}
            for resource in ("Patient", "MedicationRequest", "MedicationDispense",
                             "Encounter", "Observation", "Condition"):
                for res in await self._search(client, resource, patient_id):
                    bundle["entry"].append({"resource": res})
            return bundle_to_chart(bundle, patient_ref=f"Patient/{patient_id}")
