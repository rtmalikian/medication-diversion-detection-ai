# BIAS.md — false-positive harms, stigma, and who gets misflagged

A risk flag in a medical record is not a neutral event. It can follow a patient across clinics, colour every future encounter, and — if mishandled — reduce access to legitimate pain care or recovery support. This document names the harms and the design decisions taken against them. It is part of the product, not an appendix.

## The core asymmetry

- A **false negative** means a pattern that needed a conversation didn't get one.
- A **false positive** means a person in pain, or in recovery, is treated with suspicion they did not earn — with documented downstream effects: clinicians anchoring on the flag, prescriptions refused without review, patients avoiding care.
- Design consequence: this tool routes to *conversation and support first*, its thresholds for the high zone require objective structured-data support, and its most subjective layer (Layer C) is capped, labelled, and disable-able.

## Populations at elevated risk of being misflagged

| Population | Why patterns can mislead | Design mitigation |
|---|---|---|
| Chronic pain patients | Dose escalation, multiple specialists, ED visits for flares, and urgent requests are ordinary features of under-treated pain | Utilisation counts and chronic-pain history are **not scored**; support routing, not suspicion, is the default output |
| Cancer / palliative / sickle-cell patients | High and escalating doses are expected care | CDC 2022 scope exclusion implemented in code: dose items are context-noted at **0 points** |
| People in recovery / on MOUD | Buprenorphine in the chart, SUD history, fragmented past care | Buprenorphine **excluded** from MME; documented SUD history earns **0 points** and triggers support offers only |
| Patients with limited English proficiency / interpreted encounters | "Inconsistencies" are frequently interpretation artefacts; records may render statements inaccurately | Conversation signals and contradictions from interpreted encounters are **held informational, never scored**; language-access note printed on output |
| Rural patients | Long travel, few pharmacies/prescribers, cash pay where networks are thin | Travel is a low-weight Tier 4 item requiring clinician assertion; cash pay alone is never scored |
| Patients with fragmented care (uninsured spells, clinic closures, moves) | Multiple prescribers/pharmacies reflect the system, not the patient | Prescriber/pharmacy items are mid-weight, always shown with their counts so a clinician can see *why* |
| People with mental-health conditions | Original v1 scored mental-health history | **Removed from scoring entirely in v2** |

## Language discipline

- Output says "pattern consistent with elevated risk — requires clinician review," never a label on a person. The v1 vocabulary "doctor shopping" does not appear in v2 outputs; the underlying counts are shown instead.
- Every output explains false positives *and* false negatives in plain language, and states that sensitivity/specificity are UNKNOWN.
- History–chart contradictions are framed against the **discrepancy**, with the innocent explanations printed alongside — poor recall, interpretation, timing, and the possibility that **the chart is wrong** (MRD Escalator, SCORING.md).

## What is excluded, on purpose

Race, ethnicity, sex, preferred language, disability status, insurance type alone, national origin, religion: **not scoring inputs** (stripped at intake; a regression test proves score invariance). Age band and gender, which the v1 ML features used, are not read by the v2 scorer.

## Residual risks, honestly

- Clinicians can still over-weight a flag (automation bias). The UI shows subscores and evidence per item, and requires human confirmation before conversation items score — but no software control eliminates this.
- Layer C encodes one clinician's experience. It reflects real patterns *and* one person's blind spots; that is why it is Tier 4, capped, and first in line for item-level validation or retirement (`docs/VALIDATION_ROADMAP.md`).
- PDMP/FHIR data contains errors (wrong patient merges, pharmacy reporting lags). The chart-correction pathway exists because the record is sometimes the problem.
- Any deployment should run the fairness slice of the eval battery on its own population and monitor flag rates by demographic group *for audit purposes* (monitoring is not scoring) before widening use.
