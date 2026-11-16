"""SQLite persistence: API keys, transcriptions, the result cache and usage."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
import zlib

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    created_at REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    per_minute INTEGER NOT NULL DEFAULT 60,
    monthly_seconds INTEGER
);
CREATE TABLE IF NOT EXISTS analyses (
    content_hash TEXT NOT NULL,
    options_key TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    record BLOB NOT NULL,
    audio_seconds REAL NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (content_hash, options_key, engine_version)
);
CREATE TABLE IF NOT EXISTS transcriptions (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    finished_at REAL,
    content_hash TEXT,
    options_key TEXT,
    engine_version TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    cached INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS transcriptions_created ON transcriptions (created_at);
CREATE TABLE IF NOT EXISTS catalog_tracks (
    provider TEXT NOT NULL,
    track_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (provider, track_id)
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    transcription_id TEXT NOT NULL,
    audio_seconds REAL NOT NULL,
    cached INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_key ON usage (key_id, created_at);
CREATE TABLE IF NOT EXISTS scores (
    transcription_id TEXT PRIMARY KEY,
    document BLOB NOT NULL,
    updated_at REAL NOT NULL
);
"""


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        # Databases created before corrections could be saved lack this column.
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(transcriptions)")}
        if "edit_hash" not in columns:
            self._db.execute("ALTER TABLE transcriptions ADD COLUMN edit_hash TEXT")

    def _run(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # ---- API keys ---------------------------------------------------------

    def create_key(self, name: str, per_minute: int = 60, monthly_seconds: int | None = None) -> tuple[str, str]:
        """Create a key. Returns (id, secret); the secret is never stored and cannot be shown again."""
        key_id = "key_" + secrets.token_hex(6)
        secret = "cl_" + secrets.token_urlsafe(32)
        self._run(
            "INSERT INTO api_keys (id, name, key_hash, prefix, created_at, per_minute, monthly_seconds) VALUES (?,?,?,?,?,?,?)",
            (key_id, name, hash_key(secret), secret[:9], time.time(), per_minute, monthly_seconds),
        )
        return key_id, secret

    def find_key(self, secret: str) -> sqlite3.Row | None:
        rows = self._run("SELECT * FROM api_keys WHERE key_hash = ? AND active = 1", (hash_key(secret),))
        return rows[0] if rows else None

    def list_keys(self) -> list[sqlite3.Row]:
        return self._run("SELECT * FROM api_keys ORDER BY created_at")

    def revoke_key(self, key_id: str) -> bool:
        before = self._run("SELECT 1 FROM api_keys WHERE id = ? AND active = 1", (key_id,))
        self._run("UPDATE api_keys SET active = 0 WHERE id = ?", (key_id,))
        return bool(before)

    # ---- Result cache -----------------------------------------------------

    def get_analysis(self, content_hash: str, options_key: str, engine_version: str) -> dict | None:
        rows = self._run(
            "SELECT record FROM analyses WHERE content_hash = ? AND options_key = ? AND engine_version = ?",
            (content_hash, options_key, engine_version),
        )
        return json.loads(zlib.decompress(rows[0]["record"])) if rows else None

    def put_analysis(self, content_hash: str, options_key: str, engine_version: str, record: dict, audio_seconds: float) -> None:
        blob = zlib.compress(json.dumps(record, separators=(",", ":")).encode("utf-8"), 6)
        self._run(
            "INSERT OR REPLACE INTO analyses (content_hash, options_key, engine_version, record, audio_seconds, created_at) VALUES (?,?,?,?,?,?)",
            (content_hash, options_key, engine_version, blob, audio_seconds, time.time()),
        )

    def get_track_hash(self, provider: str, track_id: str) -> str | None:
        rows = self._run("SELECT content_hash FROM catalog_tracks WHERE provider = ? AND track_id = ?", (provider, track_id))
        return rows[0]["content_hash"] if rows else None

    def put_track_hash(self, provider: str, track_id: str, content_hash: str) -> None:
        self._run("INSERT OR REPLACE INTO catalog_tracks (provider, track_id, content_hash) VALUES (?,?,?)", (provider, track_id, content_hash))

    # ---- Transcriptions ---------------------------------------------------

    def create_transcription(self, owner: str, meta: dict, options_key: str, engine_version: str) -> tuple[str, str]:
        """Returns (id, edit token). The token lets its holder save corrections; only its hash is kept."""
        tid = "tr_" + secrets.token_urlsafe(16)
        token = "et_" + secrets.token_urlsafe(24)
        self._run(
            "INSERT INTO transcriptions (id, owner, status, created_at, options_key, engine_version, meta, edit_hash) VALUES (?,?,?,?,?,?,?,?)",
            (tid, owner, "processing", time.time(), options_key, engine_version, json.dumps(meta), hash_key(token)),
        )
        return tid, token

    def finish_transcription(self, tid: str, content_hash: str, cached: bool, meta: dict | None = None) -> None:
        if meta is not None:
            self._run("UPDATE transcriptions SET meta = ? WHERE id = ?", (json.dumps(meta), tid))
        self._run(
            "UPDATE transcriptions SET status = 'completed', finished_at = ?, content_hash = ?, cached = ? WHERE id = ?",
            (time.time(), content_hash, int(cached), tid),
        )

    def fail_transcription(self, tid: str, error: str) -> None:
        self._run("UPDATE transcriptions SET status = 'failed', finished_at = ?, error = ? WHERE id = ?", (time.time(), error[:500], tid))

    def get_transcription(self, tid: str) -> sqlite3.Row | None:
        rows = self._run("SELECT * FROM transcriptions WHERE id = ?", (tid,))
        return rows[0] if rows else None

    def delete_transcription(self, tid: str) -> None:
        self._run("DELETE FROM transcriptions WHERE id = ?", (tid,))
        self._run("DELETE FROM scores WHERE transcription_id = ?", (tid,))

    # ---- Corrected scores -------------------------------------------------

    def get_score(self, tid: str) -> dict | None:
        rows = self._run("SELECT document FROM scores WHERE transcription_id = ?", (tid,))
        return json.loads(zlib.decompress(rows[0]["document"])) if rows else None

    def put_score(self, tid: str, document: dict) -> None:
        blob = zlib.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"), 6)
        self._run("INSERT OR REPLACE INTO scores (transcription_id, document, updated_at) VALUES (?,?,?)", (tid, blob, time.time()))

    def delete_score(self, tid: str) -> None:
        self._run("DELETE FROM scores WHERE transcription_id = ?", (tid,))

    def fail_interrupted(self) -> None:
        """Jobs that were running when the server stopped cannot finish."""
        self._run("UPDATE transcriptions SET status = 'failed', error = 'interrupted|Interrupted by a server restart' WHERE status = 'processing'")

    def purge(self, older_than_days: int) -> None:
        cutoff = time.time() - older_than_days * 86400
        self._run("DELETE FROM transcriptions WHERE created_at < ?", (cutoff,))
        self._run("DELETE FROM scores WHERE transcription_id NOT IN (SELECT id FROM transcriptions)")
        self._run(
            "DELETE FROM analyses WHERE created_at < ? AND content_hash NOT IN (SELECT content_hash FROM transcriptions WHERE content_hash IS NOT NULL)",
            (cutoff,),
        )
        self._run("DELETE FROM catalog_tracks WHERE content_hash NOT IN (SELECT content_hash FROM analyses)")

    # ---- Usage ------------------------------------------------------------

    def record_usage(self, key_id: str, transcription_id: str, audio_seconds: float, cached: bool) -> None:
        self._run(
            "INSERT INTO usage (key_id, created_at, transcription_id, audio_seconds, cached) VALUES (?,?,?,?,?)",
            (key_id, time.time(), transcription_id, audio_seconds, int(cached)),
        )

    def usage_since(self, key_id: str, since: float) -> dict:
        row = self._run(
            "SELECT COUNT(*) AS requests, COALESCE(SUM(audio_seconds), 0) AS seconds, COALESCE(SUM(cached), 0) AS cached FROM usage WHERE key_id = ? AND created_at >= ?",
            (key_id, since),
        )[0]
        return {"requests": row["requests"], "audio_seconds": round(row["seconds"], 1), "cached_requests": row["cached"]}

    def close(self) -> None:
        with self._lock:
            self._db.close()
