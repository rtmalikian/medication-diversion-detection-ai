# Changelog
All notable changes to the Clinical Risk Modeling Engine for medication diversion detection will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added (2026-10-30)
- **open.epic MCP documentation integration** (`v2/epic_mcp.py`; `GET /epic-mcp/tools`, `POST /epic-mcp/docs` in `app_v2.py`): a dependency-light JSON-RPC/Streamable-HTTP client for Epic's remote MCP server at https://mcp.open.epic.com, used to look up Epic API specifications (MedicationRequest / MedicationDispense / Observation) for the FHIR integration. Documentation/spec lookup ONLY — no patient data flows through it; an in-code no-PHI guard refuses patient-like queries, and the feature degrades gracefully when disabled/unreachable. Live-verified 2026-10-30 (server "open.epic MCP" v4.0.9; tools `search_documents`, `get_document`). Plus `epic-mcp.mcp.json.example` AI-client config and an "open.epic MCP" section in `EHR_INTEGRATION.md`. Mock-only tests in `tests_v2/test_epic_mcp.py`.

## [2.0.0] - 2026-10-28 — v2 upgrade (additive; v1 preserved)
### Added
- `v2/` package + `app_v2.py`: transparent three-layer scoring (A guideline/FHIR rules, B verified-quote AI chart review, C expert/lived-experience Tier 4, capped/disable-able) with every weight in `v2/scoring_config.json` and a factor-by-factor mapping from the original scoring in `SCORING.md`.
- The **MRD Escalator** (`v2/contradictions.py`): history–chart contradictions scored by Materiality × Repetition × Directionality; single instance = informational/0 points; chart-error → correction request.
- Embedded OpenAI-compatible gateway: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/audio/transcriptions` (stub / local / API-key upstream backends).
- Real FHIR R4 client (`v2/fhir_client.py`) replacing the plan-only description: Patient/MedicationRequest/MedicationDispense/Encounter/Observation/Condition → Chart; mock-server + Synthea-format bundle tests. `EHR_INTEGRATION.md` now documents the implemented client first.
- Ambient scribe (`v2/scribe.py`): consent gate in code, audio not retained by default, diarizers behind an interface, five concrete content signals, clinician confirmation queue.
- RBAC-lite API keys, SQLite persistence, hash-chained audit log (`v2/store.py`).
- Original-format adapter (`v2/original_adapter.py`): scores the original repo's patient dicts in v2 and returns the original framework's result as provenance.
- Docs: standard proof-of-concept disclaimer at the very top of the README (v1 README did not carry it), `SCORING.md`, `EVIDENCE.md` (tiered, verified citations; unverifiable v1-era claims removed), `BIAS.md`, `EVALS.md`, `SECURITY.md`, `ARCHITECTURE.md`, `docs/VALIDATION_ROADMAP.md`.
- Tests: 67 v2 tests in `tests_v2/` + a 31-case synthetic eval battery (`v2/evals.py`, `scripts/run_eval.py`).
### Changed
- MME: original simplified factors (fentanyl ×100, methadone ×3 flat) replaced by the CDC factor table with methadone sliding scale.
- Concurrent benzo/opioid: name-count check replaced by true date-overlap computation.
- Substance-abuse and mental-health history, young-adult band, age, and gender are **no longer scoring inputs** in v2 (reasons in SCORING.md/BIAS.md). Substance-use history routes support offers at 0 points.
### Fixed
- `data/processor.py` crashed on ISO string prescription dates (the format the original tests use); dates are now parsed. Legacy evaluation framework + processor smoke-verified after the fix.

## [0.1.2] - 2026-06-08
### Security
- Fixed path traversal vulnerability in model upload endpoint: user-controlled filenames could write files outside the models directory via `../` sequences
- Added filename sanitization with strict allowlist for alphanumeric characters, hyphens, underscores, and dots
- Added file extension allowlist (.pkl, .joblib) to prevent upload of arbitrary file types
- Added maximum upload size limit (100 MB) to prevent denial-of-service
- Randomized stored filenames with cryptographic prefix to prevent overwrite attacks
- Added upload validation: uploaded model is tested via joblib.load before confirming success; invalid files are cleaned up
- Added security documentation warning about unsafe pickle deserialization in model loading
- Replaced broken `utils.logging` import with stdlib `logging` in model_management.py

## [0.1.0] - 2025-11-28
### Added
- Initial project structure with api, data, models, services, utils, and tests directories
- FastAPI application with main.py entry point
- Configuration management with config.py
- Requirements file with dependencies (FastAPI, scikit-learn, pandas, SHAP, LIME, etc.)
- ML model implementation using Gradient Boosting for diversion detection
- Data processing module for feature engineering
- Evidence-based evaluation framework using DEA, CDC, and SAMHSA guidelines
- XAI service with SHAP and LIME integration
- Database integration with SQLAlchemy
- API endpoints for risk assessment, patient data, and model management
- Risk calculation service with evidence-based and ML risk scoring
- Patient management service with CRUD operations
- Comprehensive README documentation
- License file (MIT)
- .gitignore for GitHub compatibility
- Contribution guidelines
- Setup configuration for Python packaging
- Public data sources documentation for testing
- EHR integration planning documentation
- Installation script and basic functionality tests
- Synthetic data generation capabilities
- API endpoint testing framework

## [0.1.1] - 2025-11-28
### Fixed
- Changed disclaimer from popup modal to persistent banner on every screen
- Fixed verifyProfessional function to properly show assessment section after verification
- Added proper error checking in verifyProfessional function with element existence validation
- Fixed syntax error in calculateFeatureImportance function that was preventing verification
### Changed
- Removed modal-based disclaimer system
- Updated CSS and JavaScript for persistent disclaimer display
- Enhanced verification flow with better element handling

## [0.0.1] - 2025-11-28
### Added
- Initial project creation with basic directory structure
- Core ML model architecture using Gradient Boosting
- Basic API endpoints for risk assessment
- Initial data processing capabilities
- Basic evaluation framework based on clinical guidelines