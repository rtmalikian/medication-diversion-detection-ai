"""SQLite persistence + hash-chained audit log + confirmation queue.

For a tool that flags risk patterns, access auditing matters: every chart
view, score run, scribe processing and confirmation decision is written to
an append-only audit table where each row carries the SHA-256 of the
previous row, so tampering is detectable (verify_chain).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time

_lock = threading.Lock()


def _db_path() -> str:
    return os.environ.get("MDR_DB", os.path.join(os.path.dirname(__file__), "..", "data", "drr.sqlite3"))


def connect() -> sqlite3.Connection:
    path = _db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL, actor TEXT NOT NULL, role TEXT NOT NULL,
        action TEXT NOT NULL, subject TEXT, detail TEXT,
        prev_hash TEXT NOT NULL, hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created REAL NOT NULL, type TEXT, verbatim_quote TEXT, segment_index INT,
        start REAL, speaker TEXT, detail TEXT, status TEXT DEFAULT 'pending',
        scoring_eligible INT DEFAULT 1, interpreted_encounter INT DEFAULT 0,
        confirmed_by TEXT, patient_ref TEXT);
    CREATE TABLE IF NOT EXISTS score_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL, actor TEXT, patient_ref TEXT, zone TEXT,
        total INT, result_json TEXT);
    """)
    return conn


def _chain_hash(prev: str, payload: str) -> str:
    return hashlib.sha256(f"{prev}|{payload}".encode()).hexdigest()


def audit(actor: str, role: str, action: str, subject: str = "", detail: str = "") -> str:
    with _lock:
        conn = connect()
        row = conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else "GENESIS"
        ts = time.time()
        h = _chain_hash(prev, f"{ts}|{actor}|{action}|{subject}|{detail}")
        conn.execute("INSERT INTO audit (ts, actor, role, action, subject, detail, prev_hash, hash)"
                     " VALUES (?,?,?,?,?,?,?,?)", (ts, actor, role, action, subject, detail, prev, h))
        conn.commit()
        conn.close()
        return h


def verify_chain() -> tuple[bool, int]:
    conn = connect()
    rows = conn.execute("SELECT * FROM audit ORDER BY id").fetchall()
    conn.close()
    prev = "GENESIS"
    for r in rows:
        expect = _chain_hash(prev, f"{r['ts']}|{r['actor']}|{r['action']}|{r['subject']}|{r['detail']}")
        if expect != r["hash"] or r["prev_hash"] != prev:
            return False, r["id"]
        prev = r["hash"]
    return True, len(rows)


def list_audit(limit: int = 100) -> list[dict]:
    conn = connect()
    rows = conn.execute("SELECT id, ts, actor, role, action, subject, detail, hash FROM audit"
                        " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_signal(sig: dict, patient_ref: str = "") -> int:
    conn = connect()
    cur = conn.execute(
        "INSERT INTO signals (created, type, verbatim_quote, segment_index, start, speaker,"
        " detail, status, scoring_eligible, interpreted_encounter, patient_ref)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (time.time(), sig.get("type"), sig.get("verbatim_quote"), sig.get("segment_index"),
         sig.get("start", 0.0), sig.get("speaker"), sig.get("detail"), "pending",
         1 if sig.get("scoring_eligible", True) else 0,
         1 if sig.get("interpreted_encounter") else 0, patient_ref))
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def list_signals(status: str | None = None) -> list[dict]:
    conn = connect()
    q = "SELECT * FROM signals"
    args: tuple = ()
    if status:
        q += " WHERE status = ?"
        args = (status,)
    rows = conn.execute(q + " ORDER BY id", args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def confirm_signal(signal_id: int, decision: str, reviewer: str) -> dict | None:
    conn = connect()
    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    if not row:
        conn.close()
        return None
    status = "confirmed" if decision == "accept" else "rejected"
    conn.execute("UPDATE signals SET status = ?, confirmed_by = ? WHERE id = ?",
                 (status, reviewer, signal_id))
    conn.commit()
    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    conn.close()
    return dict(row)


def record_score(actor: str, patient_ref: str, result: dict) -> int:
    conn = connect()
    cur = conn.execute("INSERT INTO score_runs (ts, actor, patient_ref, zone, total, result_json)"
                       " VALUES (?,?,?,?,?,?)",
                       (time.time(), actor, patient_ref, result.get("zone"),
                        result.get("total"), json.dumps(result)))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid
