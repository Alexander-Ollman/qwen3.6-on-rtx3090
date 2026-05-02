"""SQLite-backed persistence: sessions, request log, state transitions, last-active."""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


STATE_DIR = Path(os.environ.get("QWEN_CONTROL_STATE_DIR", "/var/qwen-control"))
DB_PATH = STATE_DIR / "qwen-control.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL,
    user_agent  TEXT,
    remote_ip   TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY,
    ts          INTEGER NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    status      INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    profile     TEXT,
    tokens      INTEGER,
    remote_ip   TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);

CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY,
    ts INTEGER NOT NULL,
    from_state TEXT,
    to_state TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_state_transitions_ts ON state_transitions(ts);
"""

REQUESTS_RETENTION_DAYS = 7


def init_db() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()


# --- state key/value -------------------------------------------------------

def get_state(key: str) -> Optional[str]:
    with connect() as conn:
        row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_state(key: str, value: str) -> None:
    now = int(time.time())
    with connect() as conn:
        conn.execute(
            "INSERT INTO state(key, value, updated_at) VALUES(?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now),
        )


# --- sessions --------------------------------------------------------------

def create_session(token: str, ttl_seconds: int, user_agent: str, remote_ip: str) -> None:
    now = int(time.time())
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions(token, created_at, expires_at, user_agent, remote_ip) VALUES(?,?,?,?,?)",
            (token, now, now + ttl_seconds, user_agent, remote_ip),
        )


def lookup_session(token: str) -> Optional[dict]:
    now = int(time.time())
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token = ? AND expires_at > ?", (token, now)
        ).fetchone()
        return dict(row) if row else None


def revoke_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def gc_sessions() -> int:
    now = int(time.time())
    with connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return cur.rowcount or 0


# --- request log -----------------------------------------------------------

def log_request(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: int,
    profile: Optional[str],
    tokens: Optional[int],
    remote_ip: str,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO requests(ts, method, path, status, duration_ms, profile, tokens, remote_ip) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (int(time.time()), method, path, status, duration_ms, profile, tokens, remote_ip),
        )


def recent_requests(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def gc_requests() -> int:
    cutoff = int(time.time()) - REQUESTS_RETENTION_DAYS * 86400
    with connect() as conn:
        cur = conn.execute("DELETE FROM requests WHERE ts < ?", (cutoff,))
        return cur.rowcount or 0


# --- state transitions -----------------------------------------------------

def log_transition(from_state: Optional[str], to_state: str, reason: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO state_transitions(ts, from_state, to_state, reason) VALUES(?,?,?,?)",
            (int(time.time()), from_state, to_state, reason),
        )
