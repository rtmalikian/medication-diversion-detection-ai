# SECURITY.md

## Scope and honest limits

Proof-of-concept research software. **Not HIPAA-certified, not a medical device, not for use with real patient data** until the deployment controls below are implemented by the deploying organisation and the validation roadmap is complete.

## v2 controls implemented

- **RBAC-lite API keys** (`X-API-Key`, roles viewer < scorer < admin) via `MDR_API_KEYS="role:key:actor,..."`. The built-in `demo-*` keys exist so tests/demos run; **change them before any deployment** — they are printed in this file, so treat them as public.
- **Access audit log**: every chart view (FHIR/bundle), score run, scribe processing, and confirmation decision is written to a SQLite **hash-chained** audit table (SHA-256 chain; `GET /audit`, admin only, reports `chain_valid`). Tampering with a stored row breaks the chain — covered by a test.
- **Ambient audio**: processed in memory, **not retained by default** (`audio_retained=False` in responses and audit details). Transcript-only retention. Scribe endpoints refuse requests without a documented consent flag (HTTP 403) — consent is a code gate, not a policy sentence.
- **PHI and model backends**: default backend is `stub`/local. Sending chart text to a hosted LLM API (`AI_BACKEND=upstream`) sends PHI to a third party — only permissible under a BAA with that provider and the deploying organisation's policies. The PHI-safe configuration is a local model (Ollama/llama.cpp + local Whisper-style STT).
- **Protected attributes** are stripped at intake and never stored in the v2 chart model.
- **No automated action**: the API exposes no endpoint that denies, reports, or alters care. Outputs are review-queue flags and support prompts.

## Known gaps before any real deployment

- API keys are shared secrets, not per-user identity: production needs OIDC/SSO, per-actor keys (the audit identity work from Raphael's clinical-llm-safety-harness roadmap), TLS termination, and key rotation.
- SQLite audit storage is single-node; production needs centralised, access-controlled, retention-managed log storage.
- No encryption-at-rest is configured by this repo; the deploying environment must provide it.
- FHIR client supports bearer/API-key headers supplied by configuration; SMART on FHIR OAuth flow is deployment integration work, documented in `EHR_INTEGRATION.md`.

## Reporting

Security issues: contact Raphael Malikian <rtmalikian@gmail.com>. Do not open public issues for vulnerabilities in deployments handling real data.
