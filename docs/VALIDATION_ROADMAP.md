# Validation roadmap — before any clinical use

The v2 composite is **unvalidated**. Sensitivity/specificity are UNKNOWN. This roadmap is the work required before the output may influence real care, in dependency order.

## 1. Retrospective clinician-adjudicated validation (gate for everything else)
- Assemble a de-identified retrospective case set (target ≥300 encounters) with two independent clinician adjudicators labelling *documented outcomes* (confirmed diversion/misuse events, confirmed benign explanations), disagreements resolved by a third reviewer; report inter-rater agreement.
- Score blinded; report per-layer and composite sensitivity/specificity with confidence intervals, calibration, and decision-curve analysis at the shipped zone thresholds.
- Pre-register thresholds and analysis before unblinding; publish the protocol in this repo.

## 2. Layer C item-level validation or retirement
- Each Tier 4 item (incl. every MRD parameter: the ≥2-encounter rule, 4/1-point directionality weights, the 12-point cap) is a hypothesis. Items that do not show independent association with adjudicated outcomes are **retired or re-weighted**, with the change logged in CHANGELOG and SCORING.md version bumped.
- MRD-specific: measure how often confirmed material contradictions were adjudicated as (a) genuine risk signals, (b) poor recall, (c) chart error — the chart-error rate is itself a data-quality outcome worth reporting.

## 3. Fairness audit
- Flag rates, false-positive rates, and time-to-review by race, ethnicity, sex, language/interpreter use, disability, insurance type, and rurality — **for audit only** (these attributes are never scoring inputs).
- Pre-set disparity thresholds that pause deployment if crossed; publish the audit method.

## 4. Scribe-specific validation
- STT word-error rate by accent/language group on consented recordings; diarization accuracy; signal-extraction precision against clinician review of transcripts.
- Consent-process audit: confirm the consent gate matches the deploying organisation's consent documentation in 100% of sampled sessions.

## 5. Prospective silent-mode pilot
- Run in shadow mode (flags visible to a study team, not treating clinicians) for a defined period; measure override rates, alert burden per clinician per session, and any change in patient-reported trust.
- Only after 1–5: a limited, governed clinical pilot with an ethics/IRB review, a named clinical owner, and a patient-facing explanation + appeal/correction pathway for flags in the record.

## 6. Ongoing
- Drift monitoring (score distribution, layer contribution mix), quarterly re-audit, incident review for any flag implicated in a care denial — with a standing rule that a flag must never be the sole basis for denial, and any such incident suspends the deployment pending review.
