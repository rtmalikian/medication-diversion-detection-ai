# EVALS.md — synthetic eval battery results (run 2026-10-28)

**Label: SYNTHETIC planted-signal battery. These numbers prove pipeline correctness only. They are NOT clinical accuracy. Sensitivity/specificity on real patients are UNKNOWN.**

Reproduce: `python scripts/run_eval.py` → `evals/eval_results.json`. Battery: `v2/evals.py` (31 cases: 19 planted-signal cases across Layers A/B/C, 12 benign look-alikes).

## Per-layer planted-signal detection

| Layer | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| A — deterministic guideline/FHIR rules | 14 | 0 | 0 | 1.000 | 1.000 |
| B — AI chart review (verified quotes) | 5 | 0 | 0 | 1.000 | 1.000 |
| C — expert/lived-experience + MRD + confirmed conversation | 6 | 1 | 0 | 0.857 | 1.000 |

The single Layer C false positive is **by design and reported deliberately**: the benign "random-direction mistakes" case earns the MRD Escalator's token random-direction points (2). It stays in the `lower` zone, but it is counted here as a scored item on a benign case rather than hidden.

## Fairness slice — benign look-alikes

12 benign cases: stable single-prescriber patient; cancer dose escalation (guideline-exempt context — dose items recorded, 0 points); patient stable on buprenorphine MOUD (excluded from MME); benzodiazepine tapered off before opioid start (no overlap); cash pay with **no** insurance (not scored); single minor contradiction; single material contradiction (poor historian); random-direction repeated mistakes; chart-error contradiction (→ correction request); interpreted-encounter inconsistency (held informational); rural narrative; two pharmacies (below threshold).

**Benign cases reaching elevated/high zone: 0 / 12 (false-flag rate 0.0 at zone level).**

## Quote verification (adversarial probe)

A fake LLM proposing 3 chart-review flags (1 with a real quote, 2 with invented quotes) against a test narrative: **proposed 3, verified 1, dropped 2.** The verifier, not the model, decides what scores.

## What this does not show

- Nothing about real-world sensitivity/specificity, calibration, or fairness across demographic groups — that requires the clinician-adjudicated validation set in `docs/VALIDATION_ROADMAP.md`.
- The stub LLM extractor is lexicon-based; a real LLM backend will propose different flags (the verifier contract is what is being tested).
