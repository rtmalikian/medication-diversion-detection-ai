"""Medication Diversion Detection AI — v2 upgrade package.

v2 wraps Raphael Malikian's original scoring (services/risk_calculator.py,
services/evaluation_framework.py, data/processor.py, models/) — preserved in
this repo and mapped factor-by-factor in SCORING.md — and adds: an embedded
OpenAI-compatible gateway, real FHIR R4 integration, verified-quote AI chart
review, a consent-gated ambient scribe, and the Materiality x Repetition x
Directionality contradiction mechanism.

Pattern language only. Never a label on a person, never an automated
consequence. Sensitivity/specificity of the composite: UNKNOWN/unvalidated.
"""
__version__ = "2.0.0"
SCORING_VERSION = "scoring-v2-2026-10-28.1"
