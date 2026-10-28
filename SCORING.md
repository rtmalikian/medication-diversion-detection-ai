# SCORING.md — v2 scoring design, and the full mapping from Raphael's original scoring

**Version:** `scoring-v2-2026-10-28.1` (config: `v2/scoring_config.json` — every weight lives there, traceable to this file)

## Guardrails that govern everything below

- Output is **pattern language for a clinician-review queue**: "pattern consistent with elevated risk — requires clinician review." The system never labels a person, never accuses, never denies care, never reports to law enforcement, and **never acts without a human clinician**.
- **Sensitivity and specificity of the composite score are UNKNOWN.** It has not been clinically validated. Every scored output says so, and explains false-positive *and* false-negative risk.
- Support routing accompanies elevated outputs: offer a PDMP check, a clinician conversation, treatment/referral resources, and — where dose/overlap factors are present — a prompt to *discuss naloxone per clinic policy* (CDC 2022 Rec 8, general guideline information, not a recommendation for an individual). No medication or treatment recommendations, no dosing instructions.
- **Protected attributes are excluded from scoring inputs** (race, ethnicity, sex, language, disability, insurance type alone, national origin, religion): the v2 chart model does not carry them, intake strips them, and a test proves the score is unchanged when they are attached.
- The three layers are **always reported separately**. There is no single opaque blended number in v2 (see mapping row for `risk_calculator._combine_scores`).

---

## Layer A — Published / guideline signals + deterministic FHIR/PDMP rules (objective)

Computed by `v2/rules.py` from structured data only (FHIR/PDMP/original-format). Thresholds marked *app-defined* are operational choices, **not** literature-validated cutoffs.

| Item | Points | Source / basis |
|---|---|---|
| `multiple_prescribers` ≥3 prescribers of controlled meds | 10 | CDC 2022 Rec 9 (review PDMP for dosages/combinations that put the patient at high risk). Threshold *app-defined*. Original: processor `prescriber_count`, ML label +1 |
| `multiple_pharmacies` ≥3 pharmacies | 8 | CDC 2022 Rec 9. Threshold *app-defined*. Original: processor `pharmacy_count`, ML label +1 |
| `opioid_benzo_overlap` ≥1 overlapping day | 15 | FDA boxed warning, Aug 31 2016 (combined use → sedation, respiratory depression, coma, death); CDC 2022 Rec 11 (avoid co-prescribing whenever possible). Original CDC check was name-count based with no dates — **changed** to a true date-overlap computation |
| `early_refill` refill ≥5 days before days-supply runs out | 10 | PDMP dispensing-pattern signal under CDC 2022 Rec 9. Threshold *app-defined*. Original: boolean `early_refill` flags — **changed** to date arithmetic (adapter reproduces legacy booleans where no dates exist, and says so) |
| `mme_ge_50` total ≥50 MME/day | 7 | CDC 2022 Rec 8 lists ≥50 MME/day among factors for offering naloxone. **A support signal, not a cutoff** — CDC 2022 explicitly cautions against hard MME thresholds. Original CDC threshold kept, meaning reframed |
| `mme_ge_90` total ≥90 MME/day | 12 | Higher-dose discussion in CDC 2022 + original DEA threshold (90). Zoning *app-defined*; not a cutoff. Original DEA `HIGH_MME` kept at this tier (the separate original CDC >120 flag is subsumed — see mapping) |
| `mme_escalation` ≥50% dose increase | 8 | Dose-trajectory review under CDC 2022 follow-up recommendations. Threshold *app-defined*. New in v2 |
| `ort_high` clinician-entered ORT total ≥8 | 10 | Webster & Webster 2005, *Pain Med* 6:432–442 (original validation n=185, one pain clinic, c=0.82 M / 0.85 F). **Limits stated on the output:** later re-assessment in a tertiary setting found discrimination no better than chance (c=0.36 M / 0.57 F) — external validation is inconsistent. v2 accepts the clinician-entered *total* only; it does not re-derive ORT items |
| `comm_elevated` clinician-entered COMM total ≥9 | 8 | Butler et al. 2007, *Pain* 130:144–156 (17-item self-report; ≥9 was the original cut score; a later primary-care study suggested ≥13). **Self-report: can under- or over-state behaviour** — stated on the output |
| `documented_prior_diversion` clinician-entered documented prior incident | 12 | Kept from the original repo (SAMHSA `HISTORY_OF_DIVERSION`, CRITICAL; ML label weight +3). Reframed: a documented fact *about the record*, presented for review — not a characterisation of the person |

**MME method:** CDC conversion factors (CDC fact sheet "Calculating Total Daily Dose of Opioids for Safer Dosage"; CDC 2022 Table 1): codeine 0.15, hydrocodone 1, hydromorphone 4, morphine 1, oxycodone 1.5, oxymorphone 3, methadone sliding 4/8/10/12 by daily dose. Buprenorphine excluded (MOUD). Calculated MME must not be used to convert between opioids (CDC). The original simplified table (fentanyl ×100, methadone ×3 flat) was **replaced** — see mapping.

**Guideline-exempt contexts:** CDC 2022 states its recommendations do *not* apply to cancer-related pain treatment, sickle cell disease, palliative or end-of-life care. Dose signals (MME tiers, escalation) in those contexts are recorded as *context-noted, 0 points*.

## Layer B — AI chart review (verified generation)

The gateway LLM extracts red-flag **observations** from narrative text. A flag scores (6 points each, cap 24) **only** with a verbatim quote that a deterministic verifier confirms against the chart text (exact or ≥0.9 token-overlap). Unverified flags are dropped and *counted* (`dropped_unverified` in the API result — the hallucination rate is visible, not hidden). Signals: lost-medication report, early-refill request in narrative, outside-prescriber mention, documented UDS inconsistency, dose-escalation request, other-source mention. Deterministic FHIR rules (Layer A) are computed completely separately, so the AI layer can be audited against them.

## Layer C — Expert / lived-experience informed (Tier 4, UNVALIDATED)

Raphael's prior scoring items from clinical experience running a direct primary care clinic and lived experience in recovery. **Every Layer C item carries the label "expert-opinion / lived-experience informed — Tier 4, UNVALIDATED"** in config, output, and here. Layer C is capped at 15 points total, can be disabled (`layer_c.enabled=false`), and **can never by itself produce the high zone** (enforced in code + tests).

| Item | Points | Origin |
|---|---|---|
| `pressure_tactics_quoted` (requires a verbatim quote) | 5 | Original README "Prescriber pressure tactics" |
| `story_inconsistency_across_visits` (clinician-asserted; see also MRD below) | 5 | Original README "Inconsistent complaints" |
| `declines_uds_or_pdmp_consent` | 4 | Original stewardship practice |
| `urgency_escalation_between_visits` | 4 | Original README "Emergency requests / unusual requests" |
| `declines_noncontrolled_alternatives` | 3 | Clinical experience |
| `travel_distance_anomaly` | 3 | Original README travel-distance factors + processor `out_of_region_prescriptions` |
| `cash_pay_insistence_with_insurance` | 3 | Original README payment factors — **changed**: cash pay alone is *never* scored; only clinician-observed insistence *despite existing coverage*. Insurance status alone is an excluded attribute |
| Confirmed conversation signals (scribe): inconsistency-with-chart 5, threat/coercion wording 5, admission of sharing/selling 5, contradicts-plan 4, third-party pressure 4 | — | v2 ambient scribe; score **only** after a clinician confirms the verbatim quote |

## The MRD Escalator — history–chart contradictions (Raphael, 2026-10-28)

*"Patients have the right to make mistakes and to be poor historians."* A history that contradicts objective chart data is scored on three escalators, never as a flat item (`v2/contradictions.py`):

1. **Materiality** — only contradictions about controlled-substance-relevant facts can ever score. Minor contradictions (dates, names, details) are informational forever: **0 points at any repetition**.
2. **Repetition** — a **single** material contradiction is an informational flag only: **0 points, and it can never by itself move a case up a zone**. Points require repetition across **≥2 separate encounters**.
3. **Directionality** — if repeated material contradictions consistently favour obtaining more controlled substance, each repetition beyond the first earns 4 points. If the mistakes point randomly (mixed or mostly non-favouring) they earn 1 point per repetition, **capped at 3** — that pattern is what ordinary poor recall looks like.

Plus: contradictions score **only after clinician confirmation** of both sides; the points are **capped at 12** (below the single objective overlap item, 15) and live **inside** the Layer C cap; if the clinician marks the chart as possibly wrong, the item becomes a **chart-correction request scoring 0** — the flag targets *the discrepancy, not the patient*; every contradiction output lists the innocent explanations (poor recall, interpretation/language, timing differences, record/chart error). Contradictions from interpreted encounters are held informational (interpretation artefacts).

## Zones

`lower` <15 · `elevated` 15–39 · `high` ≥40 (Layer A+B alone must also reach 40) · `indeterminate` when there is no PDMP/FHIR structured data and no clinician-entered instrument total — narrative or encounter features alone do not justify a zone. Zone thresholds are *app-defined operational triage for a review queue*, not validated risk strata.

---

## Full mapping: original repo → v2

Original sources: `services/evaluation_framework.py` (DEA/CDC/SAMHSA flags), `data/processor.py` (24 ML features), `models/diversion_detection.py` (`_synthesize_label` weights), `services/risk_calculator.py` (`_combine_scores`), README factor lists.

| Original factor | Where it lived | v2 layer | Kept / changed + why |
|---|---|---|---|
| MME >90 (DEA), MME >50 (CDC), MME >120 (CDC) | evaluation_framework | A (`mme_ge_50` 7, `mme_ge_90` 12) | **Kept, consolidated.** >120 tier dropped (redundant above 90). Reframed as support signals, not cutoffs, per CDC 2022's own caution |
| Simplified MME table (fentanyl ×100, methadone ×3) | evaluation_framework `_calculate_mme` | A (`v2/mme.py`) | **Changed** to the CDC factor table incl. methadone sliding scale; oral/transdermal fentanyl not scored by a flat multiplier (CDC table is product-specific) |
| Concurrent benzo + opioid (by drug-name count) | evaluation_framework CDC | A (`opioid_benzo_overlap` 15) | **Kept, strengthened**: true date-overlap from FHIR/dispense windows; FDA boxed warning + CDC Rec 11 cited |
| High prescription volume >100 (DEA "high_volume_prescriber") | evaluation_framework | — | **Removed as a patient factor**: it is a *prescriber*-volume metric misapplied to a patient record. Retained in docs as a prescriber-level concept only |
| Controlled-substance count >5 / Schedule II count >3 (DEA) | evaluation_framework | details only | **Changed to informational counts** (polypharmacy context); raw counts of legitimate prescriptions are not scored |
| Duplicate prescriptions >1 (DEA) | evaluation_framework, processor | A via rules (same-drug multi-source dispensing feeds prescriber/pharmacy/early-refill rules) | **Kept in substance**, computed from dispense data rather than a raw count field |
| Long-term use check (compared a *count* to a 90-*day* threshold — unit bug) | evaluation_framework CDC | details only | **Fixed and descoped**: duration is computed from dates; long-term use alone is not a misuse signal, so it informs context, not points |
| Substance abuse history (SAMHSA flag HIGH; ML label +2) | evaluation_framework, processor, model labels | support routing only | **Changed: 0 points.** A documented SUD history is a reason to *offer support and naloxone* (CDC Rec 8 lists it), not a score against the person. Stigma/false-flag risk documented in BIAS.md |
| Mental health history (SAMHSA flag; processor feature) | evaluation_framework, processor | — | **Removed from scoring.** A mental-health diagnosis is not a diversion signal; scoring it is stigmatising and unsupported |
| Family history of substance abuse | SAMHSA list | via ORT total only | **Changed**: appears only inside the clinician-administered ORT, not as a standalone v2 input |
| History of diversion (SAMHSA CRITICAL; ML label +3) | evaluation_framework, processor, model | A (`documented_prior_diversion` 12) | **Kept**, reframed as a documented record fact for clinician review |
| Young adult 18–25 "vulnerable population" flag | evaluation_framework | — | **Removed.** Age-band profiling is not scored in v2 |
| Age, gender_encoded | processor features | — | **Removed from v2 scoring inputs** (demographics/protected-adjacent). Legacy ML pipeline untouched, but the v2 scorer never reads them |
| prescriber_count / pharmacy_count; doctor_shopping_indicators | processor | A (`multiple_prescribers`/`multiple_pharmacies`) | **Kept, decomposed** — the component counts are scored; the label "doctor shopping" is **not used** in v2 output language |
| early_refill_events (boolean flags) | processor, ML label +1 | A (`early_refill` 10) | **Kept, changed** to date arithmetic vs days-supply |
| out_of_region_prescriptions (ML label +1) | processor | C (`travel_distance_anomaly` 3) | **Kept, demoted to Tier 4**: distance has innocent explanations (rural care, specialist access) |
| concurrent_prescriptions (boolean; ML label +2) | processor | A overlap rule | **Kept, changed** to computed overlap |
| high_dose_prescriptions (raw dose >100) | processor | A MME tiers | **Changed**: raw dose is meaningless across drugs; MME is the auditable unit |
| num_medications / polypharmacy_score / medication_cost / prescription_frequency / ED visits | processor | details / support context | **Not scored.** Utilisation can signal unmet need as easily as risk (BIAS.md) |
| chronic_pain_history | processor | context | **Not scored**; cancer/palliative contexts actively *suppress* dose scoring (CDC scope) |
| ML label heuristic (points table, threshold ≥4) | models `_synthesize_label` | — | **Not carried into v2 scoring.** It was a synthetic-training convenience, never a validated scoring rule. Preserved in the legacy model code |
| `_combine_scores`: 0.6·ML + 0.3·level + 0.1·flags → single 0–1 score | risk_calculator | — | **Replaced in v2** by the transparent layered score. The blend hid its components, and the ML model ships untrained (its probability is not meaningful out of the box). Legacy method preserved in `services/risk_calculator.py` for v1 compatibility; the v2 API does not use it |
| "Pain out of proportion to exam" | README | — | **Not scored.** A subjective impression with high bias risk; clinicians may note it in free text |
| "Provider clinical intuition / gut feeling" | README | — | **Not scored.** Unverifiable and unauditable |
| Lost/stolen reports | README | B (`lost_medication_report`) | **Kept** as a verified-quote narrative observation |
| Prescriber pressure tactics | README | C (`pressure_tactics_quoted`, quote required) | **Kept**, Tier 4 |
| Inconsistent complaints | README | C + **MRD Escalator** | **Kept and refined** by Raphael's materiality/repetition/directionality rule |
| Cash payment / insurance status | README, processor | C (`cash_pay_insistence_with_insurance`) | **Changed**: only insistence despite coverage, clinician-observed; status alone excluded |
| Diversion / "doctor shopping" admissions | README | B/C conversation + chart signals | **Kept** as verbatim-quote, human-confirmed observations only |
| ED visits / emergency requests | README, processor | C (`urgency_escalation_between_visits`, clinician-asserted pattern) + details | **Changed**: raw ED counts not scored |
| `data/processor.py` date handling | processor | — | **Bug fixed in v2** (see CHANGELOG): original crashed on ISO string dates — the format its own tests use |

## Validation roadmap

See `docs/VALIDATION_ROADMAP.md`. Headline: retrospective chart-review validation against clinician-adjudicated cases, Layer C item-level validation or retirement, MRD escalator calibration, fairness audits per BIAS.md, *before* any clinical use.
