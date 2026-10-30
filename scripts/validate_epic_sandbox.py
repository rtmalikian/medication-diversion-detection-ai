#!/usr/bin/env python3
"""Validate the FHIR client + scoring against Epic's FHIR sandbox.

Performs the full Phase 2 path with a registered Backend Systems app:

1. SMART Backend Services token exchange (RS384 JWT client assertion)
   against https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token
2. Chart fetch for one sandbox synthetic patient via the repo's real
   FHIR client (v2/fhir_client.py) with the bearer token wired in.
3. Deterministic Layer A scoring of the returned chart
   (v2/scoring.score_chart, AI chart review off — sandbox narratives
   are not used).
4. Writes a dated Markdown validation report (default:
   docs/epic-sandbox-validation-<date>.md). The report contains NO
   tokens, NO assertion, NO private key material — only the synthetic
   patient's resource counts, mapped medication names/classes, and the
   score result.

Prerequisites (see EPIC_SANDBOX.md for the registration walkthrough):
- A Backend Systems app registered at https://fhir.epic.com with the
  public key (JWK) uploaded and the system/*.read scopes approved.
- Environment:
    EPIC_CLIENT_ID          the app's client id
    EPIC_PRIVATE_KEY_PATH   path to the private key PEM (never in git)
    EPIC_PRIVATE_KEY_KID    kid of the uploaded JWK (optional if the
                            token endpoint resolves a single key)
    EPIC_SCOPES             optional override of the default scopes

Usage:
    python scripts/validate_epic_sandbox.py --patient <sandbox-patient-fhir-id>
    EPIC_PATIENT_ID=<id> python scripts/validate_epic_sandbox.py

Exit codes: 0 = validation ran and report written; 2 = configuration
missing (nothing was sent); 1 = exchange/fetch failed.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v2.fhir_client import FhirClient  # noqa: E402
from v2.scoring import score_chart  # noqa: E402
from v2.smart_backend import (  # noqa: E402
    EPIC_SANDBOX_FHIR_BASE,
    SmartAuthError,
    SmartBackendAuth,
)


async def run(patient_id: str, report_path: str) -> int:
    auth = SmartBackendAuth()
    if not auth.configured:
        print("ERROR: EPIC_CLIENT_ID and EPIC_PRIVATE_KEY_PATH must be set "
              "(see EPIC_SANDBOX.md). Nothing was sent.", file=sys.stderr)
        return 2
    fhir_base = os.environ.get("EPIC_FHIR_BASE_URL", EPIC_SANDBOX_FHIR_BASE)
    try:
        token = await auth.get_token()
    except SmartAuthError as exc:
        print(f"ERROR: token exchange failed: {exc}", file=sys.stderr)
        return 1
    print(f"Token exchange OK (token itself is never printed or stored).")

    client = FhirClient(fhir_base, auth=auth)
    try:
        chart = await client.fetch_chart(patient_id)
    except Exception as exc:  # surface honestly, no retry magic
        print(f"ERROR: FHIR fetch failed: {exc}", file=sys.stderr)
        return 1

    result = score_chart(chart, layer_b_flags=[], confirmed_conversation=[],
                         contradictions=[])
    today = dt.date.today().isoformat()
    lines = [
        f"# Epic sandbox validation — {today}",
        "",
        "Synthetic sandbox patient only. No tokens, assertions, or key",
        "material are recorded in this report.",
        "",
        f"- FHIR base: `{fhir_base}`",
        f"- Patient reference: `{chart.patient_ref}`",
        f"- Token exchange: OK (scopes requested: `{auth.scopes}`)",
        f"- Chart mapped: {len(chart.medications)} medication(s), "
        f"{len(chart.conditions)} condition(s), has_fhir_data={chart.has_fhir_data}",
        "",
        "## Medications (as mapped by v2/fhir_client.py)",
        "",
    ]
    if chart.medications:
        for m in chart.medications:
            lines.append(f"- {m.name} — class={m.drug_class}, "
                         f"daily_mg={m.daily_mg}, source={m.source}")
    else:
        lines.append("- (none returned for this patient)")
    lines += [
        "",
        "## Layer A scoring of the sandbox chart",
        "",
        f"- Zone: **{result['zone']}**, total: {result['total']}",
        f"- Subscores: {json_summary(result)}",
        "",
        "This validates plumbing (auth -> FHIR -> mapping -> scoring) on",
        "Epic sandbox synthetic data. It is NOT clinical validation.",
        "",
    ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Chart: {len(chart.medications)} meds, "
          f"{len(chart.conditions)} conditions. Zone={result['zone']} "
          f"total={result['total']}")
    print(f"Report written to {report_path}")
    return 0


def json_summary(result: dict) -> str:
    subs = result.get("subscores", {})
    return ", ".join(f"{k}={v}" for k, v in subs.items()) or "(none)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default=os.environ.get("EPIC_PATIENT_ID", ""),
                    help="Sandbox patient FHIR id (or set EPIC_PATIENT_ID)")
    ap.add_argument("--report", default=None,
                    help="Report path (default docs/epic-sandbox-validation-<date>.md)")
    args = ap.parse_args()
    if not args.patient:
        print("ERROR: --patient or EPIC_PATIENT_ID is required.", file=sys.stderr)
        return 2
    report = args.report or os.path.join(
        os.path.dirname(__file__), "..", "docs",
        f"epic-sandbox-validation-{dt.date.today().isoformat()}.md")
    return asyncio.run(run(args.patient, report))


if __name__ == "__main__":
    raise SystemExit(main())
