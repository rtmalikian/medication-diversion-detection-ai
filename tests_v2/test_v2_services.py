"""v2 services: FHIR, verified-quote AI review, scribe, gateway, API, adapter, audit."""
import json
import os

import pytest
from httpx import AsyncClient, MockTransport, Response

from v2.fhir_client import FhirClient, bundle_to_chart, drug_class_for
from v2.verify import verify_quote
from v2.chart_review import review_chart
from v2.scribe import PrefixDiarizer, Transcript, extract_signals, ConsentError
from v2.original_adapter import original_to_chart, run_original_framework
from v2 import store

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SCORER = {"X-API-Key": "demo-scorer"}
VIEWER = {"X-API-Key": "demo-viewer"}
ADMIN = {"X-API-Key": "demo-admin"}


def load_bundle():
    with open(os.path.join(FIXTURES, "synthea_bundle.json")) as f:
        return json.load(f)


# ---- FHIR ----
def test_drug_class_lexicon():
    assert drug_class_for("Oxycodone 10mg") == "opioid"
    assert drug_class_for("Lorazepam 1mg") == "benzodiazepine"
    assert drug_class_for("Buprenorphine 8mg") == "moud"
    assert drug_class_for("Lisinopril") == "other"


def test_bundle_mapping():
    chart = bundle_to_chart(load_bundle())
    names = {m.name for m in chart.medications}
    assert "Oxycodone 10mg" in names
    oxy = next(m for m in chart.medications if m.name.startswith("Oxycodone"))
    assert oxy.daily_mg == 20.0  # 10mg x 2/day
    assert oxy.prescriber_id == "Dr A"
    assert oxy.pharmacy_id == "Pharmacy 1"
    assert oxy.dispense_dates == ["2026-09-01", "2026-09-20"]
    assert chart.comm_total == 12
    assert "Chronic pain" in chart.conditions


@pytest.mark.anyio
async def test_fhir_client_against_mock_server():
    bundle = load_bundle()
    by_type = {}
    for e in bundle["entry"]:
        by_type.setdefault(e["resource"]["resourceType"], []).append(e)

    def handler(request):
        rtype = request.url.path.rsplit("/", 1)[-1]
        return Response(200, json={"resourceType": "Bundle",
                                   "entry": by_type.get(rtype, [])})

    client = FhirClient("http://mock-fhir.local/R4", transport=MockTransport(handler))
    chart = await client.fetch_chart("syn-1")
    assert chart.has_fhir_data and len(chart.medications) == 2


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---- verified generation ----
def test_verify_quote_exact():
    assert verify_quote("he lost my medication on a trip",
                        "Patient reports he lost my medication on a trip today.")


def test_verify_quote_hallucination_rejected():
    assert not verify_quote("patient admitted selling pills to neighbours",
                            "Patient reports he lost my medication on a trip.")


@pytest.mark.anyio
async def test_chart_review_drops_hallucinated_quote():
    async def fake_llm(_text):
        return json.dumps({"flags": [
            {"signal": "lost_medication_report",
             "verbatim_quote": "he lost my medication on a trip", "chart_location": "narrative"},
            {"signal": "other_source_mention",
             "verbatim_quote": "patient confessed to selling pills at a school",
             "chart_location": "narrative"}]})

    out = await review_chart("Patient reports he lost my medication on a trip and asks for help.",
                             chat_fn=fake_llm)
    assert out["proposed"] == 2 and out["dropped"] == 1 and len(out["flags"]) == 1


@pytest.mark.anyio
async def test_chart_review_stub_end_to_end():
    out = await review_chart("Note states the patient also receives oxycodone from an urgent care clinic.")
    assert len(out["flags"]) == 1 and out["flags"][0]["verified"] is True


# ---- scribe ----
def make_transcript(text, **kw):
    kw.setdefault("consent_documented", True)
    return Transcript(segments=PrefixDiarizer().diarize(text), **kw)


def test_scribe_requires_consent_object_level():
    t = Transcript(segments=PrefixDiarizer().diarize("Patient: hello"), consent_documented=False)
    with pytest.raises(ConsentError):
        extract_signals(t)


def test_scribe_inconsistency_is_concrete_and_quoted():
    t = make_transcript("Patient: I take three a day, not two, because two does nothing.")
    sigs = extract_signals(t, chart_daily_doses={"oxycodone": 2})
    inc = [s for s in sigs if s["type"] == "inconsistent_with_chart"]
    assert inc and inc[0]["verbatim_quote"].startswith("I take three a day")
    assert inc[0]["status"] == "pending_confirmation"


def test_scribe_sharing_admission_signal():
    t = make_transcript("Patient: I gave some of my pills to my brother last month.")
    assert any(s["type"] == "admission_sharing_selling" for s in extract_signals(t))


def test_scribe_third_party_pressure():
    t = make_transcript("Other: You have to give him the stronger ones today.")
    assert any(s["type"] == "third_party_pressure" for s in extract_signals(t))


def test_scribe_no_psychological_state_claims():
    t = make_transcript("Patient: I take three a day, not two.")
    for s in extract_signals(t, chart_daily_doses={"oxycodone": 2}):
        assert "lying" not in str(s).lower() and "manipulat" not in str(s).lower()


def test_scribe_interpreted_signals_not_scoring_eligible():
    t = make_transcript("Patient: I gave some of my pills to my brother.", interpreted=True)
    sigs = extract_signals(t)
    assert sigs and all(s["scoring_eligible"] is False for s in sigs)


# ---- API ----
def test_api_auth_required(client):
    assert client.post("/score", json={"chart": {}}).status_code == 401


def test_api_viewer_cannot_score(client):
    assert client.post("/score", json={"chart": {}}, headers=VIEWER).status_code == 403


def test_api_score_roundtrip(client):
    r = client.post("/score", headers=SCORER, json={
        "chart": {"patient_ref": "t1", "has_pdmp_data": True,
                  "medications": [{"name": "Morphine", "drug_class": "opioid", "daily_mg": 100}]},
        "use_ai_chart_review": False})
    assert r.status_code == 200
    body = r.json()
    assert body["subscores"]["layer_a_published_guideline"] == 12
    assert body["sensitivity_specificity"].startswith("UNKNOWN")


def test_api_gateway_models_and_chat(client):
    assert client.get("/v1/models", headers=VIEWER).status_code == 200
    r = client.post("/v1/chat/completions", headers=VIEWER, json={
        "messages": [{"role": "user", "content": "<chart>He lost my medication yesterday.</chart>"}]})
    assert r.status_code == 200
    assert "lost_medication_report" in r.json()["choices"][0]["message"]["content"]


def test_api_gateway_embeddings_shape(client):
    r = client.post("/v1/embeddings", headers=VIEWER, json={"input": ["oxycodone refill"]})
    assert len(r.json()["data"][0]["embedding"]) == 64


def test_api_consent_gate_blocks_scribe(client):
    r = client.post("/scribe/transcript", headers=SCORER, json={
        "text": "Patient: hello", "consent_documented": False})
    assert r.status_code == 403 and "consent" in r.json()["detail"].lower()


def test_api_scribe_queue_confirm_flow(client):
    r = client.post("/scribe/transcript", headers=SCORER, json={
        "text": "Patient: I gave some of my pills to my brother last month.",
        "consent_documented": True, "patient_ref": "scribe-1"})
    assert r.status_code == 200 and r.json()["audio_retained"] is False
    sig = r.json()["signals"][0]
    c = client.post(f"/queue/{sig['queue_id']}/confirm", headers=SCORER, json={"decision": "accept"})
    assert c.status_code == 200 and c.json()["status"] == "confirmed" and c.json()["confirmed"] is True


def test_api_audio_path_stub_not_retained(client):
    with open(os.path.join(FIXTURES, "tiny.wav"), "rb") as f:
        r = client.post("/scribe/audio", headers=SCORER,
                        files={"file": ("tiny.wav", f, "audio/wav")},
                        data={"consent_documented": "true", "patient_ref": "audio-1"})
    assert r.status_code == 200
    assert r.json()["audio_retained"] is False
    assert r.json()["transcription_backend"] == "stub"


def test_api_audio_consent_gate(client):
    with open(os.path.join(FIXTURES, "tiny.wav"), "rb") as f:
        r = client.post("/scribe/audio", headers=SCORER,
                        files={"file": ("tiny.wav", f, "audio/wav")},
                        data={"consent_documented": "false"})
    assert r.status_code == 403


def test_api_transcriptions_endpoint(client):
    with open(os.path.join(FIXTURES, "tiny.wav"), "rb") as f:
        r = client.post("/v1/audio/transcriptions", headers=SCORER,
                        files={"file": ("tiny.wav", f, "audio/wav")})
    assert r.status_code == 200 and "text" in r.json()


def test_api_bundle_score_and_audit_chain(client):
    r = client.post("/fhir/bundle/score", headers=SCORER, json=load_bundle())
    assert r.status_code == 200
    body = r.json()
    ids = {i["id"] for i in body["items"]}
    assert "early_refill" in ids and "comm_elevated" in ids
    a = client.get("/audit", headers=ADMIN)
    assert a.status_code == 200 and a.json()["chain_valid"] is True


def test_api_original_format_endpoint(client):
    patient = {
        "patient_id": "orig-1", "age": 52, "gender": "female",
        "medications": [{"name": "Oxycodone"}],
        "prescriptions": [{"drug_name": "Oxycodone 20mg", "dose": 20, "prescriber_id": "Dr A",
                           "pharmacy_id": "Pharm 1", "date": "2026-08-01", "days_supply": 30}],
        "clinical_indicators": {"history_of_diversion": True, "substance_abuse_history": True,
                                "mental_health_history": True},
        "history": {"ed_visits": 2}}
    r = client.post("/v2/score/original-format", headers=SCORER, json={"patient": patient})
    assert r.status_code == 200
    body = r.json()
    assert "documented_prior_diversion" in {i["id"] for i in body["items"]}
    assert body["original_framework_v1"] is not None
    assert any("mental_health_history" in n for n in body["adapter_provenance"]["notes"])


def test_audit_chain_tamper_detected(tmp_path, monkeypatch):
    db = str(tmp_path / "tamper.sqlite3")
    monkeypatch.setenv("MDR_DB", db)
    store.audit("actor1", "scorer", "score_run", "p1", "zone=lower")
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute("UPDATE audit SET detail='zone=high' WHERE id=1")
    conn.commit(); conn.close()
    ok, _ = store.verify_chain()
    assert ok is False


# ---- original adapter ----
def test_adapter_maps_original_fields():
    patient = {"patient_id": "o1",
               "prescriptions": [
                   {"drug_name": "Oxycodone 10mg", "dose": 10, "prescriber_id": "Dr A",
                    "pharmacy_id": "P1", "date": "2026-08-01", "days_supply": 30},
                   {"drug_name": "Oxycodone 10mg", "dose": 10, "prescriber_id": "Dr B",
                    "pharmacy_id": "P2", "date": "2026-08-05", "days_supply": 30}],
               "clinical_indicators": {"history_of_diversion": True,
                                       "substance_abuse_history": True,
                                       "mental_health_history": True}}
    chart, prov = original_to_chart(patient)
    assert chart.documented_prior_diversion is True
    assert chart.sud_history_documented is True
    assert not hasattr(chart, "gender") or getattr(chart, "gender", None) is None
    assert any("mental_health_history" in n for n in prov["notes"])


def test_original_framework_runs():
    out = run_original_framework({"patient_id": "o2", "prescriptions": [],
                                  "clinical_indicators": {}})
    assert out is not None and (out.get("overall_risk_level") == "LOW" or "unavailable" in out)
