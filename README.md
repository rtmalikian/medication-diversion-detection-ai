> ## ⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED
>
> **This tool has not been tested clinically and is a proof-of-concept for
> research and software demonstration only. It is not a medical device, it
> does not diagnose, treat, or rule out any condition, and its outputs may
> be wrong — including false positives and false negatives.**
>
> **Do not use it to make decisions about your health.**
> For any medical issue or concern, seek a licensed healthcare professional.
> **In an emergency, go to your nearest emergency room or dial 911 immediately.**

**Product-specific limits (in addition to the banner above):** this system produces *patterns for clinician review*, never labels on people. It never accuses a patient, never denies care, never reports to law enforcement, and never acts without a human clinician. **Sensitivity and specificity are UNKNOWN — the composite score is unvalidated.** False positives occur (chronic pain, cancer care, recovery on MOUD, interpreted encounters, fragmented care, record errors) and false negatives occur (a lower score does not rule out misuse or diversion). Ambient audio features are **consent-first** (no documented consent → processing is refused in code), audio is **not retained by default**, and conversation signals are verbatim quotes a clinician must confirm before they can count. The system detects **no psychological states** — deception detection from speech is near chance in the literature (54%, Bond & DePaulo 2006), so the scribe extracts only concrete, checkable content. See `BIAS.md` and `EVIDENCE.md`.

# Diversion Risk Radar — Medication Diversion Detection AI (v2) | Evidence-Based Controlled-Substance Diversion & Misuse Risk Scoring with FHIR EHR Integration, AI Chart Review, and Ambient Scribe Conversation Signals

**Medication diversion detection, opioid stewardship, PDMP risk signals, controlled-substance monitoring, MME calculation, FHIR R4 EHR integration, ambient clinical scribe, AI chart review, and clinician-review flagging — in one open, test-covered system.**

Built by [Raphael Malikian](https://github.com/rtmalikian) — healthcare founder/operator (bootstrapped and ran a direct primary care clinic as CEO) now building clinical AI. His original scoring (v1, preserved in this repo) came from that clinical practice and from lived experience in recovery; v2 keeps every original factor accounted for — see the factor-by-factor mapping in [`SCORING.md`](SCORING.md).

## What v2 adds over the original repo

| Capability | v1 (original) | v2 (this upgrade) |
|---|---|---|
| Scoring | Gradient Boosting ML + DEA/CDC/SAMHSA flag framework, blended 0.6/0.3/0.1 into one opaque 0–1 score | **Transparent three-layer score**, always reported separately, every weight in `v2/scoring_config.json` traceable to `SCORING.md` (legacy blend preserved for comparison; v2 API doesn't use it) |
| EHR integration | `EHR_INTEGRATION.md` described FHIR/Epic/Cerner plans | **Working FHIR R4 client** (`v2/fhir_client.py`): Patient, MedicationRequest, MedicationDispense, Encounter, Observation, Condition → deterministic rules; tested against a mock FHIR server + Synthea-format bundle |
| AI chart review | — | Embedded **OpenAI-compatible gateway** (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions`; local Ollama/llama.cpp or API-key upstream). LLM flags score **only with a verbatim quote verified against the chart** — hallucinated quotes are dropped and counted |
| Ambient scribe | — | Consent-gated speech-to-text pipeline → diarized transcript → **concrete content signals only** (chart-inconsistent statements, explicit threats/coercion wording, plan contradictions, admissions of sharing/selling, third-party pressure) → **clinician confirmation queue**; only confirmed quotes score |
| Audit & access | Database service | RBAC-lite API keys + **hash-chained audit log** (who viewed/scored whom, tamper-evident) |
| Evidence discipline | "DEA/CDC/SAMHSA guidelines" framing | `EVIDENCE.md` with tiered, verified citations; thresholds labelled guideline-sourced or **app-defined**; `BIAS.md`; eval battery with benign look-alikes (`EVALS.md`) |

## The three layers (never merged silently)

- **Layer A — Published/guideline signals (objective).** CDC 2022 guideline signals computed deterministically from FHIR/PDMP data: multiple prescribers/pharmacies, opioid×benzodiazepine date overlap (FDA boxed warning; CDC Rec 11), early-refill date arithmetic, MME tiers + escalation (CDC factors; a *support signal, not a cutoff*), clinician-entered ORT and COMM totals (validation limits stated on the output), documented prior incidents. Cancer/palliative/sickle-cell contexts suppress dose scoring, per CDC's own scope.
- **Layer B — AI chart review.** Verified-quote narrative observations only (above).
- **Layer C — Expert / lived-experience informed (Tier 4, UNVALIDATED).** Raphael's prior scoring items: quoted pressure tactics, story inconsistency, reluctance toward non-controlled alternatives / UDS / PDMP consent, travel-distance anomalies, cash-pay insistence *despite* coverage (cash alone is never scored), escalating urgency. **Capped at 15, disable-able, and can never alone produce the high zone.**

## The MRD Escalator — scoring history–chart contradictions fairly

A patient's history contradicting objective chart data is a real signal — but patients have the right to make mistakes and to be poor historians. So contradictions are scored by **Materiality × Repetition × Directionality**, never as a flat item:

- **Materiality:** trivial contradictions (dates, details) are informational forever — 0 points.
- **Repetition:** a *single* material contradiction is an informational flag only — **0 points, and it can never by itself move a case up a zone**. Points require repetition across ≥2 separate encounters.
- **Directionality:** mistakes that point both ways (random direction) earn a token amount (≤3 total) — that is what poor recall looks like. A pattern where the "mistakes" consistently favour obtaining more controlled substance earns progressively more (4 per qualifying repetition).
- Contradiction points are **capped at 12** — below the objective opioid×benzo overlap item (15) — and sit inside the Layer C cap. They score only after **clinician confirmation**, every flag prints the **innocent explanations alongside** (poor recall, interpretation, timing, and the possibility that **the chart is wrong**), and a clinician can convert a flag into a **chart-correction request** (0 points) — the flag targets *the discrepancy, not the patient*.

## What this proves — and what it does NOT prove

**It proves** the engineering: FHIR data flows into auditable rules; an LLM's flags are filtered by quote verification; a scribe can surface checkable conversation content under a consent gate with human confirmation; a legacy scoring system can be transparently mapped, corrected, and layered without discarding its author's clinical knowledge. On the synthetic battery (31 cases): Layer A precision/recall 1.0/1.0, Layer B 1.0/1.0, Layer C 0.857/1.0 (the one FP is the MRD random-direction token, reported deliberately), **0 of 12 benign look-alikes reached an elevated zone** — see `EVALS.md`. Those are pipeline-correctness numbers on planted synthetic signals, labelled Tier 4.

**It does NOT prove** clinical accuracy. There is no real-patient validation; sensitivity/specificity are UNKNOWN; no pilot, customer, or deployment is claimed. `docs/VALIDATION_ROADMAP.md` is the required path before any clinical use.

## Epic documentation via open.epic MCP

v2 also integrates Epic's **open.epic MCP server** (`https://mcp.open.epic.com`) — a **documentation/spec-lookup** integration (`v2/epic_mcp.py`; `GET /epic-mcp/tools`, `POST /epic-mcp/docs`) used to retrieve Epic's API specifications for the MedicationRequest / MedicationDispense / Observation APIs the FHIR client consumes. It returns documentation only — **no patient data ever flows through it** (Epic states it is not designated to handle PHI, and an in-code no-PHI guard refuses patient-like queries); patient data stays on the FHIR R4 + SMART path. Full details: the "open.epic MCP" section of [`EHR_INTEGRATION.md`](EHR_INTEGRATION.md); AI-client config example: `epic-mcp.mcp.json.example`.

## Epic sandbox validation (SMART Backend Services)

SMART **Backend Services** auth for Epic's FHIR sandbox is implemented in `v2/smart_backend.py` (RS384 JWT client assertion, token caching) and wired into the FHIR client; `scripts/validate_epic_sandbox.py` runs token exchange → chart fetch → scoring on a sandbox synthetic patient. Mock-tested; **live run pending app registration** — see [`EPIC_SANDBOX.md`](EPIC_SANDBOX.md).

## Quickstart (v2)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-v2.txt
uvicorn app_v2:app --port 8000     # demo keys: viewer/demo-viewer, scorer/demo-scorer, admin/demo-admin
python -m pytest tests_v2/ -q      # 67 tests
python scripts/run_eval.py         # synthetic eval battery -> evals/eval_results.json
```

Configuration: `AI_BACKEND=stub|local|upstream`, `AI_BASE_URL`, `AI_API_KEY`, `AI_STT_BASE_URL`, `MDR_API_KEYS`, `MDR_DB`. Default is the offline stub — PHI-safe pattern is a **local** model; sending charts to a hosted API requires a BAA (see `SECURITY.md`).

Endpoints: gateway `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions` · scoring `POST /score`, `POST /fhir/score`, `POST /fhir/bundle/score`, `POST /v2/score/original-format` (accepts the original repo's patient format and returns the v1 framework result as provenance) · scribe `POST /scribe/transcript`, `POST /scribe/audio`, `GET /queue`, `POST /queue/{id}/confirm` · `GET /audit` (admin), `GET /healthz`, demo UI at `/`.

## Original v1 (preserved)

The original system remains intact and importable: Gradient Boosting model (`models/diversion_detection.py`), 24-feature processor (`data/processor.py`), DEA/CDC/SAMHSA evaluation framework (`services/evaluation_framework.py`), SHAP/LIME explainability (`services/xai_service.py`), risk calculator with the original blend (`services/risk_calculator.py`), the v1 FastAPI app (`main.py`, `/api/v1/*`), `train_demo.py`, and the standalone `ui_demo.html` tool. v2's one legacy code change is a documented bug fix (processor crashed on ISO string dates). **Intellectual property:** this Clinical Risk Modeling Engine is the intellectual property of Raphael Tomas Malikian; attribution required, commercial use by permission (contact the author). The original prototype disclaimer and creator notes are preserved in git history and `CHANGELOG.md`.

## Creator

**Raphael Tomas Malikian** — rtmalikian@gmail.com · Family Medicine background; founded and ran a direct primary care clinic as CEO; lived experience of how damaging drug diversion is to providers, patients, pharmacies, and the community, and of recovery. That experience informs this tool — and its insistence that a flag is a prompt for a conversation and support, never a verdict on a person.

## Keywords

medication diversion detection · drug diversion prevention · controlled substance risk scoring · opioid stewardship · PDMP · prescription drug monitoring · MME morphine milligram equivalents · FHIR R4 · EHR integration · ambient scribe · speech to text clinical · AI chart review · LLM red flags · opioid risk tool · COMM · CDC opioid guideline 2022 · clinician decision support · healthcare AI · clinical AI
