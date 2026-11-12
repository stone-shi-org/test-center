"""SQLite result cache for ``llm_coding`` per-problem evaluations.

Mirrors the live_stt ``AccuracyResultCache``: one row per (problem, model, endpoint, generation /
judge / scoring settings) key, so re-running an unchanged configuration skips the LLM calls, the
Piston judging, and the LLM judge for problems already evaluated. New or edited problems change
``problem_sha256`` and therefore miss the cache.

Bump ``CACHE_SCHEMA_VERSION`` when grading logic that is not captured in the key changes (harness
merge, feedback text, Piston status mapping, score computation), so stale results are not reused.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

CACHE_SCHEMA_VERSION = "llm_coding_cache_v1"
DEFAULT_CACHE_FILENAME = "llm_coding_cache.sqlite3"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CodingResultCache:
    """SQLite-backed cache of ``(case_metrics, case_detail, agg_record)`` per problem.

    Errors, truncated responses, and failed judge calls are never cached (the caller decides via
    :func:`is_cacheable`). Safe to use from the tester's worker threads.
    """

    def __init__(self, cache_path: str | Path, enabled: bool = True, skip_read: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.enabled = bool(enabled)
        self.skip_read = bool(skip_read)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.Lock()
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_coding_cache (
                    cache_key TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def _count(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    @staticmethod
    def compute_key(key_payload: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(sha256_hex_key, canonical_json)`` for a key dictionary."""
        enriched = dict(key_payload)
        enriched.setdefault("schema_version", CACHE_SCHEMA_VERSION)
        canonical = json.dumps(enriched, sort_keys=True, separators=(",", ":"), default=str)
        return sha256_text(canonical), canonical

    def get(self, key_payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the cached value, or None on miss / when disabled or skip_read."""
        if not self.enabled or self.skip_read:
            self._count("misses")
            return None
        cache_key, _ = self.compute_key(key_payload)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT schema_version, value_json, created_at FROM llm_coding_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row and row[0] == CACHE_SCHEMA_VERSION:
            try:
                parsed = json.loads(row[1])
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                parsed["cached_at"] = row[2]
                self._count("hits")
                return parsed
        self._count("misses")
        return None

    def put(self, key_payload: Mapping[str, Any], value_payload: Mapping[str, Any]) -> str:
        """Persist a successful evaluation. Write failures are swallowed (cache is best-effort)."""
        cache_key, canonical_key = self.compute_key(key_payload)
        if not self.enabled:
            return cache_key
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_coding_cache (
                        cache_key, schema_version, problem_id, model, key_json, value_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        schema_version = excluded.schema_version,
                        value_json = excluded.value_json,
                        created_at = excluded.created_at
                    """,
                    (
                        cache_key,
                        CACHE_SCHEMA_VERSION,
                        str(key_payload.get("problem_id", "")),
                        str(key_payload.get("model", "")),
                        canonical_key,
                        json.dumps(dict(value_payload), ensure_ascii=False, default=str),
                        time.time(),
                    ),
                )
                conn.commit()
        except sqlite3.Error:
            return cache_key
        self._count("writes")
        return cache_key

    def clear(self) -> int:
        """Delete all entries and return the number of rows removed."""
        if not self.cache_path.exists():
            return 0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM llm_coding_cache")
            conn.commit()
            return int(cur.rowcount or 0)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "skip_read": self.skip_read,
            "path": str(self.cache_path),
            "schema_version": CACHE_SCHEMA_VERSION,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


def is_cacheable(case_detail: Mapping[str, Any]) -> bool:
    """Only cache complete, deterministic-outcome evaluations.

    Skips: evaluation exceptions (runner/LLM transport errors), truncated streams (usually gateway
    timeouts), and judge calls that were attempted but failed.
    """
    if case_detail.get("error"):
        return False
    if case_detail.get("judge_error"):
        return False
    for att in case_detail.get("attempts") or []:
        if isinstance(att, Mapping) and att.get("truncated"):
            return False
    return True
