"""SQLite result cache for ``llm_coding`` per-problem evaluations.

Mirrors the live_stt ``AccuracyResultCache``: one row per (problem, model, endpoint, generation /
judge / scoring settings) key, so re-running an unchanged configuration skips the LLM calls, the
Piston judging, and the LLM judge for problems already evaluated. New or edited problems change
``problem_sha256`` and therefore miss the cache.

Bump ``CACHE_SCHEMA_VERSION`` when grading logic that is not captured in the key changes (harness
merge, feedback text, Piston status mapping), so stale results are not reused.

The key covers only what changes the model's answer or its test results. The LLM judge and the score
weights are deliberately NOT part of it: an entry stores each judge verdict under a fingerprint of
(judge model, endpoint, prompt) in ``judges``, and the case score is recomputed on every hit. So
turning the judge on/off, changing its model or the weights never makes the model run again; at most
the judge is called on code that is already cached.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

CACHE_SCHEMA_VERSION = "llm_coding_cache_v2"
LEGACY_SCHEMA_VERSIONS = ("llm_coding_cache_v1",)
# Fields v1 keys carried that v2 keys do not (judge and scoring settings).
V1_ONLY_KEY_FIELDS = (
    "judge_enabled", "attempt_factors", "weight_correctness", "weight_judge",
    "judge_model", "judge_endpoint_url", "judge_prompt_sha256", "judge_failed_solutions",
)
DEFAULT_CACHE_FILENAME = "llm_coding_cache.sqlite3"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_fingerprint(judge_model: str | None, judge_endpoint_url: str | None, judge_prompt_sha256: str | None) -> str:
    """Identity of a judge setup; a stored verdict is reused only for the same fingerprint."""
    return sha256_text(json.dumps([str(judge_model or ""), str(judge_endpoint_url or ""), str(judge_prompt_sha256 or "")]))[:32]


def is_judge_placeholder(judge: Any) -> bool:
    """A judge result that is a stand-in (empty/unparseable reply or failed request), not a verdict."""
    if not isinstance(judge, dict):
        return False
    just = str(judge.get("justification") or "")
    return bool(judge.get("error")) or just.startswith(("Judge output parse fallback", "Judge request failed"))


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
            try:
                self.migrated = self.migrate_legacy()
            except Exception:  # best effort: legacy rows just stay unused
                self.migrated = 0

    migrated = 0

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

    def migrate_legacy(self) -> int:
        """Convert v1 entries to v2 so nothing has to re-run after the key change.

        v1 keys included the judge/score settings; the entry's embedded judge verdict (when it is a
        real one) moves into ``judges`` under its fingerprint, and the embedded copy is cleared. When
        several v1 entries map to one v2 key, the newest wins and keeps judge verdicts only from
        entries with identical code. A backup copy of the cache file is written first.
        """
        with self._connect() as conn:
            marks = ",".join("?" for _ in LEGACY_SCHEMA_VERSIONS)
            rows = conn.execute(
                "SELECT cache_key, problem_id, model, key_json, value_json, created_at FROM llm_coding_cache "
                f"WHERE schema_version IN ({marks})", LEGACY_SCHEMA_VERSIONS,
            ).fetchall()
            if not rows:
                return 0
            conn.execute("PRAGMA wal_checkpoint(FULL)")
        backup = self.cache_path.with_name(f"{self.cache_path.name}.bak-v1-{int(time.time())}")
        shutil.copy2(self.cache_path, backup)

        groups: dict[str, list[dict[str, Any]]] = {}
        for old_key, pid, model, key_json, value_json, created_at in rows:
            try:
                key = json.loads(key_json)
                value = json.loads(value_json)
                detail = value["case_detail"]
            except (ValueError, TypeError, KeyError):
                continue
            judges: dict[str, Any] = {}
            if key.get("judge_enabled") and isinstance(detail.get("judge"), dict) and not is_judge_placeholder(detail["judge"]):
                fp = judge_fingerprint(key.get("judge_model"), key.get("judge_endpoint_url"), key.get("judge_prompt_sha256"))
                judges[fp] = detail["judge"]
            new_key = {k: v for k, v in key.items() if k not in V1_ONLY_KEY_FIELDS}
            new_key["schema_version"] = CACHE_SCHEMA_VERSION
            detail["judge"] = None
            detail["judge_error"] = None
            metrics = value.get("case_metrics") or {}
            for k in [k for k in metrics if k.startswith("judge_")]:
                del metrics[k]
            attempts = detail.get("attempts") or []
            code_sha = sha256_text(str((attempts[-1] if attempts else {}).get("code") or ""))
            value["judges"] = judges
            new_hash, canonical = self.compute_key(new_key)
            groups.setdefault(new_hash, []).append(
                {"key": new_key, "canonical": canonical, "value": value, "judges": judges, "code": code_sha,
                 "pid": pid, "model": model, "created_at": created_at}
            )
        with self._connect() as conn:
            conn.execute("BEGIN")
            for new_hash, items in groups.items():
                items.sort(key=lambda it: it["created_at"], reverse=True)
                best = items[0]
                for other in items[1:]:
                    if other["code"] == best["code"]:
                        for fp, jd in other["judges"].items():
                            best["judges"].setdefault(fp, jd)
                best["value"]["judges"] = best["judges"]
                conn.execute(
                    "INSERT OR IGNORE INTO llm_coding_cache (cache_key, schema_version, problem_id, model, key_json, value_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (new_hash, CACHE_SCHEMA_VERSION, best["pid"], best["model"], best["canonical"],
                     json.dumps(best["value"], ensure_ascii=False, default=str), best["created_at"]),
                )
            marks = ",".join("?" for _ in LEGACY_SCHEMA_VERSIONS)
            conn.execute(f"DELETE FROM llm_coding_cache WHERE schema_version IN ({marks})", LEGACY_SCHEMA_VERSIONS)
            conn.commit()
        return len(rows)

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

    Skips: evaluation exceptions (runner/LLM transport errors) and truncated streams (usually gateway
    timeouts). A failed judge call does not block caching: the model's answer and test results are
    valid, and the judge verdict is simply missing from the entry until a judge call succeeds.
    """
    if case_detail.get("error"):
        return False
    for att in case_detail.get("attempts") or []:
        if isinstance(att, Mapping) and att.get("truncated"):
            return False
    return True
