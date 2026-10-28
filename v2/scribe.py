"""Ambient scribe — consent-first, quote-based, human-confirmed.

Scientific honesty (see EVIDENCE.md): deception detection from speech is
near chance in the literature (Bond & DePaulo 2006: 54% average accuracy).
This module therefore detects NO psychological states. It extracts only
CONCRETE, CHECKABLE content signals, each stored as a verbatim transcript
quote with its segment index and timestamp:

  1. inconsistent_with_chart      — a numeric dose/frequency claim in the
                                    transcript that contradicts chart facts
  2. threat_or_coercion           — an explicit threat/coercion statement
  3. contradicts_plan             — a statement contradicting the agreed
                                    treatment plan (plan statements supplied
                                    by the clinician as checkable facts)
  4. admission_sharing_selling    — an explicit admission of sharing/selling
  5. third_party_pressure         — pressure language from a non-patient
                                    speaker segment

Signals NEVER score directly. They enter a clinician confirmation queue;
only a clinician-confirmed signal (verbatim quote reviewed by a human) may
be passed to the scorer, where it counts as a Layer C observation.
Signals from interpreted encounters are marked scoring_eligible=False:
inconsistency signals in particular may be interpretation artefacts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class ConsentError(PermissionError):
    """Raised when scribe processing is attempted without documented consent."""


@dataclass
class Segment:
    speaker: str            # Patient | Clinician | Other | Unknown
    text: str
    start: float = 0.0
    end: float = 0.0


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    consent_documented: bool = False
    interpreted: bool = False
    audio_retained: bool = False     # default: audio NOT retained
    source: str = "text"             # text | audio-stub | audio-upstream

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


class Diarizer:
    """Interface. Implementations must be labelled honestly — speaker
    attribution from a mono clinic recording is error-prone."""


class PrefixDiarizer(Diarizer):
    """Parses 'Speaker: text' lines; unlabelled sentences in a plain
    transcript are attributed to Unknown (NOT guessed as the patient)."""

    SPEAKERS = ("Patient", "Clinician", "Other")

    def diarize(self, text: str) -> list[Segment]:
        segments: list[Segment] = []
        for chunk in re.split(r"(?<=[.!?])\s+", (text or "").strip()):
            chunk = chunk.strip()
            if not chunk:
                continue
            speaker = "Unknown"
            for sp in self.SPEAKERS:
                if chunk.lower().startswith(sp.lower() + ":"):
                    speaker = sp
                    chunk = chunk[len(sp) + 1:].strip()
                    break
            segments.append(Segment(speaker=speaker, text=chunk))
        return segments


class PauseDiarizer(Diarizer):
    """Stub diarizer over timestamped STT segments: a new speaker is assumed
    after a pause >= pause_seconds; speakers alternate Patient/Clinician.
    This is a heuristic placeholder behind the Diarizer interface, labelled
    as such — it is not a validated diarization model."""

    def __init__(self, pause_seconds: float = 1.5):
        self.pause_seconds = pause_seconds

    def diarize_segments(self, raw: list[dict]) -> list[Segment]:
        out: list[Segment] = []
        speaker = "Clinician"
        prev_end = None
        for seg in raw:
            start = float(seg.get("start", 0.0))
            if prev_end is not None and start - prev_end >= self.pause_seconds:
                speaker = "Patient" if speaker == "Clinician" else "Clinician"
            out.append(Segment(speaker=speaker, text=seg.get("text", ""),
                               start=start, end=float(seg.get("end", start))))
            prev_end = float(seg.get("end", start))
        return out


THREAT_PATTERNS = ["or else", "i will sue", "you'll be sorry", "i'll report you",
                   "end up in the emergency room if", "go to the emergency room if you don't",
                   "i know where you"]
SHARING_PATTERNS = ["gave some of my pills", "gave my pills", "sold my pills",
                    "gave some of my tablets", "shared my medication with",
                    "gave them to my brother", "gave them to my sister",
                    "gave some to my"]
PLAN_PATTERNS = ["i stopped taking the", "i take two at a time", "i take double",
                 "i don't take the", "i threw away the"]
THIRD_PARTY_PATTERNS = ["you have to give", "give him the", "give her the",
                        "he needs the stronger", "she needs the stronger"]

_DOSE_CLAIM = re.compile(r"i take (\w+) (?:tablets?|pills?|capsules?)?\s*(?:a day|per day|daily)", re.I)
_NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _dose_claims(text: str) -> list[int]:
    out = []
    for m in _DOSE_CLAIM.finditer(text.lower()):
        w = m.group(1)
        if w.isdigit():
            out.append(int(w))
        elif w in _NUMWORDS:
            out.append(_NUMWORDS[w])
    return out


def extract_signals(transcript: Transcript, chart_daily_doses: dict[str, int] | None = None,
                    plan_statements: list[str] | None = None) -> list[dict]:
    """Extract concrete content signals. Every signal carries the verbatim
    quote (exact segment text) — no quote, no signal."""
    if not transcript.consent_documented:
        raise ConsentError("ambient scribe requires documented patient consent")
    chart_daily_doses = chart_daily_doses or {}
    chart_doses = set(chart_daily_doses.values())
    signals: list[dict] = []
    for idx, seg in enumerate(transcript.segments):
        low = seg.text.lower()
        base = {"verbatim_quote": seg.text, "segment_index": idx,
                "start": seg.start, "speaker": seg.speaker,
                "status": "pending_confirmation",
                "scoring_eligible": not transcript.interpreted,
                "interpreted_encounter": transcript.interpreted}
        if seg.speaker in ("Patient", "Unknown"):
            for claim in _dose_claims(seg.text):
                if chart_doses and claim not in chart_doses:
                    signals.append({**base, "type": "inconsistent_with_chart",
                                    "detail": f"transcript states {claim} per day; chart records {sorted(chart_doses)} per day"})
                    break
            if any(p in low for p in THREAT_PATTERNS):
                signals.append({**base, "type": "threat_or_coercion",
                                "detail": "explicit threat/coercion wording in transcript (verbatim quote for clinician confirmation; wording alone does not establish intent)"})
            if any(p in low for p in SHARING_PATTERNS):
                signals.append({**base, "type": "admission_sharing_selling",
                                "detail": "statement describing sharing/selling of medication (verbatim quote for clinician confirmation)"})
            if any(p in low for p in PLAN_PATTERNS):
                signals.append({**base, "type": "contradicts_plan",
                                "detail": "statement appears to contradict the agreed plan (clinician to confirm against the actual plan)"})
        if seg.speaker == "Other" and any(p in low for p in THIRD_PARTY_PATTERNS):
            signals.append({**base, "type": "third_party_pressure",
                            "detail": "pressure wording from a non-patient speaker in the room"})
    return signals
