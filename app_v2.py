"""Medication Diversion Detection AI v2 — API + demo UI (upgrades the original repo).

Auth: RBAC-lite API keys via X-API-Key. Configure with MDR_API_KEYS as
comma-separated role:key:actor triples, e.g.
  MDR_API_KEYS="viewer:vk1:dr-a,scorer:sk1:dr-b,admin:ak1:sysop"
Defaults are demonstration keys (documented as such in SECURITY.md — change
before any real deployment): viewer/demo-viewer, scorer/demo-scorer,
admin/demo-admin.
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from v2 import __version__, gateway, store
from v2.chart_review import review_chart
from v2.epic_mcp import EpicMcpClient, PhiGuardError
from v2.fhir_client import FhirClient, bundle_to_chart
from v2.models import chart_from_dict
from v2.scoring import load_config, score_chart
from v2.scribe import ConsentError, PrefixDiarizer, Transcript, extract_signals

app = FastAPI(title="Medication Diversion Detection AI — v2",
              description="Evidence-based controlled-substance diversion & misuse risk "
                          "scoring for clinician review — pattern flagging and support "
                          "routing only. Proof of concept, not clinically tested.",
              version=__version__)

ROLE_RANK = {"viewer": 1, "scorer": 2, "admin": 3}


def _keys() -> dict[str, tuple[str, str]]:
    raw = os.environ.get("MDR_API_KEYS",
                         "viewer:demo-viewer:demo-viewer,scorer:demo-scorer:demo-scorer,"
                         "admin:demo-admin:demo-admin")
    out = {}
    for part in raw.split(","):
        bits = part.strip().split(":")
        if len(bits) == 3:
            role, key, actor = bits
            out[key] = (role, actor)
    return out


def require_role(min_role: str):
    def dep(request: Request):
        key = request.headers.get("x-api-key", "")
        entry = _keys().get(key)
        if not entry:
            raise HTTPException(401, "missing or invalid API key")
        role, actor = entry
        if ROLE_RANK.get(role, 0) < ROLE_RANK[min_role]:
            raise HTTPException(403, f"requires role >= {min_role}")
        return {"role": role, "actor": actor}
    return dep


# ----------------------------- gateway -----------------------------

class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]


class EmbedRequest(BaseModel):
    model: str | None = None
    input: list[str] | str


@app.get("/v1/models")
async def models(user=Depends(require_role("viewer"))):
    backend = gateway.backend_name()
    return {"object": "list", "data": [
        {"id": os.environ.get("AI_MODEL", "stub-extract-1"), "object": "model", "backend": backend},
        {"id": os.environ.get("AI_EMBED_MODEL", "stub-embed-1"), "object": "model", "backend": backend},
        {"id": os.environ.get("AI_STT_MODEL", "whisper-1"), "object": "model", "backend": backend},
    ]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, user=Depends(require_role("viewer"))):
    store.audit(user["actor"], user["role"], "gateway_chat", "", f"backend={gateway.backend_name()}")
    return await gateway.chat_completion(req.messages, req.model)


@app.post("/v1/embeddings")
async def embeddings(req: EmbedRequest, user=Depends(require_role("viewer"))):
    texts = req.input if isinstance(req.input, list) else [req.input]
    return await gateway.embeddings(texts, req.model)


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str | None = Form(None),
                         user=Depends(require_role("scorer"))):
    audio = await file.read()
    store.audit(user["actor"], user["role"], "gateway_transcribe", file.filename or "",
                f"bytes={len(audio)} audio_retained=False")
    return await gateway.transcribe(audio, file.filename or "audio.wav", model)


# ----------------------------- open.epic MCP (documentation only) -------------
# Epic's open.epic MCP server returns DOCUMENTATION (API specs, playbooks,
# FAQs) — never patient data. Patient data flows only through the FHIR R4
# client above. See EHR_INTEGRATION.md "open.epic MCP" and v2/epic_mcp.py.

def epic_mcp_client() -> EpicMcpClient:
    return EpicMcpClient()


class EpicDocsRequest(BaseModel):
    query: str


@app.get("/epic-mcp/tools")
async def epic_mcp_tools(user=Depends(require_role("viewer"))):
    """List the open.epic MCP documentation tools (degrades gracefully)."""
    return await epic_mcp_client().list_tools()


@app.post("/epic-mcp/docs")
async def epic_mcp_docs(req: EpicDocsRequest, user=Depends(require_role("viewer"))):
    """Search Epic's API documentation via the open.epic MCP server.

    Use case here: retrieve the Epic API specifications for the
    MedicationRequest / MedicationDispense / Observation APIs that
    v2/fhir_client.py consumes, while building or debugging the EHR
    integration. The no-PHI guard in v2/epic_mcp.py refuses any query
    that looks like patient data — documentation-style queries only.
    """
    try:
        return await epic_mcp_client().search_docs(req.query)
    except PhiGuardError as e:
        raise HTTPException(400, str(e))


# ----------------------------- scoring -----------------------------

class ScoreRequest(BaseModel):
    chart: dict
    confirmed_conversation_signals: list[dict] = []
    contradictions: list[dict] = []
    use_ai_chart_review: bool = True


@app.post("/score")
async def score(req: ScoreRequest, user=Depends(require_role("scorer"))):
    chart = chart_from_dict(req.chart)
    review = {"flags": [], "proposed": 0, "dropped": 0}
    if req.use_ai_chart_review and chart.narrative:
        review = await review_chart(chart.narrative)
    from v2.contradictions import Contradiction
    contras = [Contradiction(**{k: v for k, v in c.items() if k in Contradiction.__dataclass_fields__})
               for c in req.contradictions]
    result = score_chart(chart, layer_b_flags=review["flags"],
                         confirmed_conversation=req.confirmed_conversation_signals,
                         contradictions=contras)
    result["layer_b_review"] = {"proposed": review["proposed"],
                                "dropped_unverified": review["dropped"],
                                "verified": len(review["flags"])}
    run_id = store.record_score(user["actor"], chart.patient_ref, result)
    store.audit(user["actor"], user["role"], "score_run", chart.patient_ref,
                f"zone={result['zone']} total={result['total']} run={run_id}")
    return result


class FhirScoreRequest(BaseModel):
    fhir_base_url: str
    patient_id: str
    narrative: str = ""
    encounter_features: dict = {}


@app.post("/fhir/score")
async def fhir_score(req: FhirScoreRequest, user=Depends(require_role("scorer"))):
    client = FhirClient(req.fhir_base_url)
    try:
        chart = await client.fetch_chart(req.patient_id)
    except Exception as e:  # surface FHIR failures honestly
        raise HTTPException(502, f"FHIR fetch failed: {e}")
    chart.narrative = req.narrative
    chart.encounter_features = req.encounter_features
    store.audit(user["actor"], user["role"], "chart_view_fhir", chart.patient_ref, req.fhir_base_url)
    review = await review_chart(chart.narrative) if chart.narrative else {"flags": [], "proposed": 0, "dropped": 0}
    result = score_chart(chart, layer_b_flags=review["flags"])
    result["layer_b_review"] = {"proposed": review["proposed"],
                                "dropped_unverified": review["dropped"],
                                "verified": len(review["flags"])}
    store.record_score(user["actor"], chart.patient_ref, result)
    return result


@app.post("/fhir/bundle/score")
async def bundle_score(bundle: dict, user=Depends(require_role("scorer"))):
    chart = bundle_to_chart(bundle)
    result = score_chart(chart)
    store.audit(user["actor"], user["role"], "chart_view_bundle", chart.patient_ref, "")
    store.record_score(user["actor"], chart.patient_ref, result)
    return result


class OriginalFormatRequest(BaseModel):
    patient: dict
    contradictions: list[dict] = []
    include_original_framework: bool = True


@app.post("/v2/score/original-format")
async def score_original_format(req: OriginalFormatRequest, user=Depends(require_role("scorer"))):
    """Score a patient in Raphael's ORIGINAL repo input format. The original
    framework also runs (provenance only) so v1 and v2 outputs can be compared."""
    from v2.original_adapter import original_to_chart, run_original_framework
    from v2.contradictions import Contradiction
    chart, provenance = original_to_chart(req.patient)
    contras = [Contradiction(**{k: v for k, v in c.items() if k in Contradiction.__dataclass_fields__})
               for c in req.contradictions]
    result = score_chart(chart, contradictions=contras)
    result["adapter_provenance"] = provenance
    if req.include_original_framework:
        result["original_framework_v1"] = run_original_framework(req.patient)
    store.audit(user["actor"], user["role"], "score_run_original_format", chart.patient_ref,
                f"zone={result['zone']} total={result['total']}")
    store.record_score(user["actor"], chart.patient_ref, result)
    return result


# ----------------------------- scribe -----------------------------

class TranscriptRequest(BaseModel):
    patient_ref: str = "anonymous"
    text: str
    consent_documented: bool = False
    interpreted: bool = False
    retain_audio: bool = False          # accepted for API clarity; text path stores no audio
    chart_daily_doses: dict = {}        # e.g. {"oxycodone": 2} tablets/day for contradiction checks
    plan_statements: list[str] = []


@app.post("/scribe/transcript")
async def scribe_transcript(req: TranscriptRequest, user=Depends(require_role("scorer"))):
    if not req.consent_documented:
        raise HTTPException(403, "consent gate: ambient scribe requires documented patient consent "
                                 "(consent_documented=true). Recording without consent is refused.")
    transcript = Transcript(segments=PrefixDiarizer().diarize(req.text),
                            consent_documented=True, interpreted=req.interpreted,
                            audio_retained=False, source="text")
    try:
        signals = extract_signals(transcript, chart_daily_doses=req.chart_daily_doses,
                                  plan_statements=req.plan_statements)
    except ConsentError as e:
        raise HTTPException(403, str(e))
    ids = [store.add_signal(s, patient_ref=req.patient_ref) for s in signals]
    store.audit(user["actor"], user["role"], "scribe_process", req.patient_ref,
                f"signals={len(signals)} audio_retained=False source=text")
    return {"transcript_segments": len(transcript.segments), "audio_retained": False,
            "signals": [{**s, "queue_id": i} for s, i in zip(signals, ids)],
            "note": "Signals are verbatim-quote observations pending clinician confirmation. "
                    "They do not score until confirmed, and this tool detects no psychological states."}


@app.post("/scribe/audio")
async def scribe_audio(file: UploadFile = File(...),
                       consent_documented: bool = Form(False),
                       patient_ref: str = Form("anonymous"),
                       interpreted: bool = Form(False),
                       user=Depends(require_role("scorer"))):
    if not consent_documented:
        raise HTTPException(403, "consent gate: ambient scribe requires documented patient consent. "
                                 "Recording without consent is refused.")
    audio = await file.read()
    result = await gateway.transcribe(audio, file.filename or "audio.wav")
    # audio bytes end here: processed in memory, NOT retained by default.
    transcript = Transcript(segments=PrefixDiarizer().diarize(result["text"]),
                            consent_documented=True, interpreted=interpreted,
                            audio_retained=False, source=f"audio-{result.get('backend', 'stub')}")
    signals = extract_signals(transcript)
    ids = [store.add_signal(s, patient_ref=patient_ref) for s in signals]
    store.audit(user["actor"], user["role"], "scribe_process", patient_ref,
                f"signals={len(signals)} audio_retained=False source=audio backend={result.get('backend')}")
    return {"transcription_backend": result.get("backend"), "audio_retained": False,
            "transcript_text": result["text"],
            "signals": [{**s, "queue_id": i} for s, i in zip(signals, ids)]}


@app.get("/queue")
async def queue(status: str | None = None, user=Depends(require_role("viewer"))):
    return {"signals": store.list_signals(status)}


class ConfirmRequest(BaseModel):
    decision: str  # accept | reject


@app.post("/queue/{signal_id}/confirm")
async def confirm(signal_id: int, req: ConfirmRequest, user=Depends(require_role("scorer"))):
    if req.decision not in ("accept", "reject"):
        raise HTTPException(422, "decision must be 'accept' or 'reject'")
    row = store.confirm_signal(signal_id, req.decision, user["actor"])
    if not row:
        raise HTTPException(404, "signal not found")
    store.audit(user["actor"], user["role"], "signal_confirm", str(signal_id), req.decision)
    out = dict(row)
    out["confirmed"] = row["status"] == "confirmed"
    return out


# ----------------------------- audit / health / UI -----------------------------

@app.get("/audit")
async def audit(limit: int = 100, user=Depends(require_role("admin"))):
    ok, n = store.verify_chain()
    return {"chain_valid": ok, "entries_checked": n, "events": store.list_audit(limit)}


@app.get("/healthz")
async def healthz():
    ok, n = store.verify_chain()
    return {"status": "ok", "version": __version__, "audit_chain_valid": ok, "audit_entries": n,
            "backend": gateway.backend_name(), "scoring_version": load_config()["version"]}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(UI_HTML)


UI_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Medication Diversion Detection AI v2 — clinician-review flagging (proof of concept)</title></head>
<body style="font-family:system-ui;max-width:860px;margin:2rem auto;padding:0 1rem">
<blockquote><h2>⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED</h2>
<p><strong>This tool has not been tested clinically and is a proof-of-concept for
research and software demonstration only. It is not a medical device, it
does not diagnose, treat, or rule out any condition, and its outputs may
be wrong — including false positives and false negatives.</strong></p>
<p><strong>Do not use it to make decisions about your health.</strong>
For any medical issue or concern, seek a licensed healthcare professional.
<strong>In an emergency, go to your nearest emergency room or dial 911 immediately.</strong></p></blockquote>
<h1>Medication Diversion Detection AI — v2</h1>
<p>Pattern flagging for <strong>clinician review</strong> and patient-support routing.
It never labels a person, never denies care, and never acts without a human clinician.
Sensitivity and specificity are <strong>UNKNOWN</strong> — the composite is unvalidated.
False positives (chronic pain, cancer care, recovery on MOUD, interpreted encounters,
fragmented care) and false negatives both occur; a lower score does not rule out risk.</p>
<p>Ambient scribe is <strong>consent-first</strong>: no consent on file, no processing.
Audio is not retained by default. Conversation signals are verbatim quotes that a
clinician must confirm before they can count — the tool detects no psychological states.</p>
<p>API: <code>POST /score</code>, <code>POST /fhir/score</code>, <code>POST /scribe/transcript</code>,
<code>GET /queue</code> — see README. Demo API keys: scorer key <code>demo-scorer</code>.</p>
</body></html>"""
