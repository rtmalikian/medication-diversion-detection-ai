"""Embedded OpenAI-compatible gateway.

Endpoints served by app.main: /v1/chat/completions, /v1/models,
/v1/embeddings, /v1/audio/transcriptions.

Backends (env AI_BACKEND):
- "stub" (default): deterministic offline backend. Its chat model performs
  lexicon-based extraction over the supplied chart text (real, auditable
  logic — not a random generator), embeddings are hashing vectors, and STT
  returns a demonstration transcript. Documented as a stub everywhere.
- "local": any OpenAI-compatible local server (Ollama / llama.cpp) at
  AI_BASE_URL (default http://localhost:11434/v1). STT expects a local
  Whisper-style server exposing /audio/transcriptions at AI_STT_BASE_URL.
- "upstream": same as local but AI_BASE_URL/AI_API_KEY point at a hosted
  OpenAI-compatible API. PHI warning: only use an upstream under a BAA —
  see SECURITY.md. PHI-safe default is local.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import httpx

DEMO_TRANSCRIPT = (
    "Clinician: How have the tablets been going since last visit? "
    "Patient: I take three a day, not two, because two does nothing for me. "
    "Patient: My cousin said you have to give me the stronger ones or else I will end up in the emergency room. "
    "Patient: To be honest I gave some of my pills to my brother last month, he was in pain too."
)


def backend_name() -> str:
    return os.environ.get("AI_BACKEND", "stub").lower()


def _base_url() -> str:
    return os.environ.get("AI_BASE_URL", "http://localhost:11434/v1").rstrip("/")


def _api_key() -> str:
    return os.environ.get("AI_API_KEY", "")


# ---------------- embeddings (stub: hashing) ----------------

def embed_stub(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------- chat ----------------

# Lexicon for the stub chat extractor. Maps narrative phrases (as dictated /
# written in real charts) to Layer B signal ids. A flag quotes the exact
# sentence it matched — verified again downstream by app.verify.
STUB_SIGNAL_LEXICON: dict[str, list[str]] = {
    "lost_medication_report": ["lost my medication", "medication was lost", "pills were stolen",
                               "reported the medication was lost", "lost prescription"],
    "early_refill_request_narrative": ["requesting an early refill", "requests early refill",
                                       "asking for an early refill", "ran out early"],
    "outside_prescriber_mention": ["also receives oxycodone from", "also prescribed by an urgent care",
                                   "gets tramadol from another", "second prescriber at"],
    "uds_inconsistency_documented": ["urine drug screen was negative for the prescribed",
                                     "uds negative for prescribed", "urine screen inconsistent"],
    "dose_escalation_request_documented": ["insists on a dose increase", "demanding a higher dose",
                                           "requests a dose increase despite"],
    "other_source_mention": ["reports buying tablets", "obtained tablets from a friend",
                             "using a family member's medication", "bought pills from"],
}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def stub_chat(messages: list[dict]) -> str:
    """Deterministic stub 'LLM': extraction mode when the prompt carries a
    chart in <chart>...</chart>, otherwise echoes a JSON acknowledgement."""
    user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    m = re.search(r"<chart>(.*?)</chart>", user, re.DOTALL)
    if not m:
        return json.dumps({"detail": "stub backend: no <chart> block supplied"})
    chart_text = m.group(1)
    flags = []
    for sent in _sentences(chart_text):
        low = sent.lower()
        for signal, phrases in STUB_SIGNAL_LEXICON.items():
            if any(p in low for p in phrases):
                flags.append({"signal": signal, "verbatim_quote": sent,
                              "chart_location": "narrative"})
                break
    return json.dumps({"flags": flags})


async def chat_completion(messages: list[dict], model: str | None = None) -> dict:
    name = backend_name()
    model = model or os.environ.get("AI_MODEL", "stub-extract-1")
    if name == "stub":
        content = stub_chat(messages)
    else:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{_base_url()}/chat/completions",
                headers={"Authorization": f"Bearer {_api_key()}"} if _api_key() else {},
                json={"model": model, "messages": messages, "temperature": 0},
            )
            r.raise_for_status()
            return r.json()
    return {
        "id": "chatcmpl-stub", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def embeddings(texts: list[str], model: str | None = None) -> dict:
    name = backend_name()
    model = model or os.environ.get("AI_EMBED_MODEL", "stub-embed-1")
    if name == "stub":
        data = [{"object": "embedding", "index": i, "embedding": embed_stub(t)}
                for i, t in enumerate(texts)]
    else:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{_base_url()}/embeddings",
                headers={"Authorization": f"Bearer {_api_key()}"} if _api_key() else {},
                json={"model": model, "input": texts},
            )
            r.raise_for_status()
            return r.json()
    return {"object": "list", "data": data, "model": model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


async def transcribe(audio_bytes: bytes, filename: str = "audio.wav",
                     model: str | None = None) -> dict:
    """STT. Stub returns DEMO_TRANSCRIPT (documented demo content, not a
    transcription of the uploaded audio). Local/upstream POST to a
    Whisper-style /audio/transcriptions endpoint. Audio is processed in
    memory only; retention is controlled by the scribe layer, not here."""
    name = backend_name()
    model = model or os.environ.get("AI_STT_MODEL", "whisper-1")
    if name == "stub":
        return {"text": DEMO_TRANSCRIPT, "model": model, "backend": "stub",
                "note": "STUB STT: returned demonstration transcript, not a transcription of the uploaded audio"}
    base = os.environ.get("AI_STT_BASE_URL", _base_url()).rstrip("/")
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {_api_key()}"} if _api_key() else {},
            files={"file": (filename, audio_bytes)},
            data={"model": model},
        )
        r.raise_for_status()
        out = r.json()
        out.setdefault("backend", name)
        return out
