# EVIDENCE.md — claims, sources, tiers (Raphael's 2026-10-23 evidence standard)

Tier 1 = guideline / systematic review / meta-analysis / authoritative compendium.
Tier 2 = RCT / large cohort / validated measurement study. Tier 3 = small observational.
Tier 4 = expert opinion / vendor / this app's own synthetic results (labelled, never dressed as evidence).

All citations below were verified 2026-10-28 by opening the source or its PubMed/publisher record.

## A. The problem is real

| Claim | Source (full citation + link) | Tier | What it proves | What it does NOT prove |
|---|---|---|---|---|
| Clinicians should review PDMP data before initial and periodically during opioid therapy to identify dosages/combinations that put patients at high risk, and should assess risk and offer naloxone where risk factors are present | Dowell D, Ragan KR, Jones CM, Baldwin GT, Chou R. CDC Clinical Practice Guideline for Prescribing Opioids for Pain — United States, 2022. MMWR Recomm Rep 2022;71(RR-3):1–95. https://pubmed.ncbi.nlm.nih.gov/36327391/ (doi:10.15585/mmwr.rr7103a1) | 1 | PDMP review, risk assessment, and naloxone offers are guideline-recommended components of opioid stewardship (Recs 8, 9, 11) | That any particular scoring algorithm identifies diversion, or that these signals diagnose misuse in an individual |
| Combined opioid + benzodiazepine use causes serious harm including respiratory depression and death, warranting boxed warnings | FDA Drug Safety Communication, Aug 31 2016: "FDA warns about serious risks and death when combining opioid pain or cough medicines with benzodiazepines; requires its strongest warning." https://www.fda.gov/drugs/food-and-drug-administration-overdose-prevention-framework/new-safety-measures-announced-opioid-analgesics-prescription-opioid-cough-products-and | 1 (regulatory) | The combination itself is a recognised, serious risk — an objective chart pattern worth clinician attention regardless of intent | Anything about a specific patient's intent or honesty |
| MME can be computed with published conversion factors to compare total daily opioid dose | CDC. "Calculating Total Daily Dose of Opioids for Safer Dosage" (CDC Stacks cdc_38481) + CDC 2022 guideline Table 1. https://stacks.cdc.gov/view/cdc/38481/cdc_38481_DS1.pdf | 1 | A standardised dose-comparison method; CDC lists ≥50 MME/day among factors for offering naloxone | That MME thresholds are safe/unsafe cutoffs — CDC 2022 explicitly cautions against using them as hard thresholds, and this tool does not |

## B. The solution methods work in principle (with limits)

| Claim | Source | Tier | What it proves | What it does NOT prove |
|---|---|---|---|---|
| The Opioid Risk Tool stratified aberrant-behaviour risk in its original validation | Webster LR, Webster RM. Predicting aberrant behaviors in opioid-treated patients: preliminary validation of the Opioid Risk Tool. Pain Med. 2005;6(6):432–442. https://academic.oup.com/painmedicine/article-abstract/6/6/432/1853982 (doi:10.1111/j.1526-4637.2005.00072.x) | 2–3 | In one pain clinic (n=185), ORT bands discriminated (c=0.82 M / 0.85 F) | Universal validity — a later tertiary-setting re-assessment found discrimination no better than chance (c=0.36 M / 0.57 F; Pain Medicine 2018 re-assessment, https://academic.oup.com/painmedicine/article/19/7/1382/4835592). v2 therefore treats ORT as one clinician-entered input among several and states the inconsistency on the output |
| The COMM measures current aberrant drug-related behaviour in chronic-pain patients on opioid therapy | Butler SF, Budman SH, Fernandez KC, et al. Development and validation of the Current Opioid Misuse Measure. Pain. 2007;130(1–2):144–156. https://doi.org/10.1016/j.pain.2007.01.014 | 2 | A 17-item self-report instrument with validation samples (n=277, n=302 in related testing) and an original cut score of ≥9 | That self-report is complete or truthful in either direction; a later primary-care analysis suggested ≥13. COMM is an input, not a verdict |
| Human deception detection is barely above chance — so "detecting manipulation" from speech is not a scientifically supportable feature | Bond CF Jr, DePaulo BM. Accuracy of deception judgments. Pers Soc Psychol Rev. 2006;10(3):214–234. https://pubmed.ncbi.nlm.nih.gov/16859438/ (206 documents, 24,483 judges; mean accuracy 54%, lies 47%, truths 61%) | 1 | Any system claiming to detect lying/intent from voice or manner is claiming something the literature does not support | That concrete, checkable *content* (a statement contradicting a recorded fact) cannot be extracted — that is the only thing the scribe does, and even that requires human confirmation |
| Verified generation (a claim counts only with a checkable source quote) reduces fabricated-flag risk in extraction pipelines | Method claim — this is an engineering discipline used across Raphael's clinical products (Discharge Decoded, Prior-Auth Co-Pilot), evaluated here only on synthetic fixtures | 4 | On this repo's synthetic battery, hallucinated quotes were dropped (adversarial probe: 3 proposed, 1 verified, 2 dropped) | Any published validation of this implementation |

## C. Claims deliberately NOT made (removed / downgraded during the evidence audit)

- **No prevalence statistic for diversion is quoted.** Specific diversion-rate figures in the original README era were not re-verifiable to a primary source in this pass, so they were removed rather than repeated.
- **Layer C items (including the MRD Escalator) are Tier 4** — author clinical/lived experience. They are labelled UNVALIDATED everywhere, capped, and disable-able. No citation is attached to them because none exists; attaching one would be decoration.
- **Synthea**: the bundled test bundle is a locally constructed FHIR R4 bundle *in Synthea's format*. No claim about Synthea's published validation is made in this repo (its citation was not verified in this pass).
- **SAMHSA/DEA framing**: the original repo described its thresholds as "DEA/SAMHSA guidelines." On audit, several were the author's operational thresholds, not published agency cutoffs. v2 labels every threshold as either guideline-sourced (with the citation above) or *app-defined*. No DEA/SAMHSA threshold is claimed where the underlying document could not be verified.

## What this app has NOT proven

Its own accuracy on real patients is unproven. The eval battery (`EVALS.md`, `evals/eval_results.json`) uses synthetic planted-signal cases and proves pipeline correctness only. Sensitivity and specificity are **UNKNOWN** until the validation work in `docs/VALIDATION_ROADMAP.md` is done. Do not cite the synthetic numbers as clinical performance.
