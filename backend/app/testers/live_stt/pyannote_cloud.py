"""pyannote.ai cloud diarization as a live-stt tester backend.

Lets the live-stt tester's ``accuracy_diarization`` suite score https://pyannote.ai (cloud) next to
live-stt's own diarizers (e.g. Nemotron-3.5-Transcribe), so runs line up in Compare. The protocol
follows my-meeting-notes' ``app/services/pyannote_cloud.py`` and docs.pyannote.ai:

  1. POST /v1/media/input   -> presigned PUT url for a ``media://`` object key
  2. PUT  <presigned url>    -> raw audio bytes (application/octet-stream, no auth header)
  3. POST /v1/diarize        -> jobId; ``transcription: true`` for speaker-attributed text
  4. GET  /v1/jobs/{jobId}   -> poll until succeeded / failed / canceled

The job output is mapped onto live-stt's ``/v1/audio/diarization`` house JSON so the scoring code
needs no cloud branch:

  - ``segments`` are pyannote.ai's own ``diarization`` turns (overlap included), the right input
    for DER, unlike the merged ``turnLevelTranscription``. Each ``wordLevelTranscription`` word
    is put into the best-overlapping turn of the *same* speaker, so per-speaker text (cpWER) is
    exactly pyannote.ai's own attribution.
  - ``words`` carries the word-level speaker labels. Plain WER is scored on the segment text in
    turn order (``LiveSTTTester._one_pass_transcript_text``), not on words sorted by time.

pyannote.ai bills every succeeded job (20 s minimum), so the tester's SQLite result cache matters:
re-runs over the same meetings are free. Uploaded media and job outputs are deleted by pyannote.ai
itself (48 h / 24 h); nothing here cleans up.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from backend.app.testers.live_stt.client import (
    DiarizeResult,
    DiscoveredModels,
    LiveSTTClientError,
    ModelInfo,
)

PYANNOTE_AI_BASE_URL = "https://api.pyannote.ai/v1"
PYANNOTE_AI_ENGINE = "pyannote_ai"
MODEL_PREFIX = "pyannote.ai/"

# pyannote.ai's own `model` values. Only precision-* support `transcription: true`;
# community-1 is diarization-only (speaker turns, no text) and is scored like local pyannote.
DIARIZE_MODELS = ("precision-2", "precision-3", "community-1")
TRANSCRIBING_MODELS = frozenset({"precision-2", "precision-3"})
DEFAULT_DIARIZE_MODEL = "precision-2"

# `transcriptionConfig.model`: the STT model pyannote.ai runs alongside diarization.
TRANSCRIBE_MODELS = ("parakeet-tdt-0.6b-v3", "faster-whisper-large-v3-turbo")
DEFAULT_TRANSCRIBE_MODEL = "parakeet-tdt-0.6b-v3"

DEFAULT_POLL_INTERVAL_SEC = 5.0
DEFAULT_JOB_TIMEOUT_SEC = 900.0
MAX_RATE_LIMIT_RETRIES = 5
_BACKEND_ALIASES = frozenset({"pyannote_ai", "pyannote-ai", "pyannoteai", "pyannote.ai"})


def is_pyannote_ai_target(target: dict[str, Any], config: dict[str, Any] | None = None) -> bool:
    """True if a target points at the pyannote.ai cloud API (by kind, extra.backend, or host)."""
    extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}
    for value in (target.get("kind"), extra.get("backend"), (config or {}).get("backend")):
        if isinstance(value, str) and value.strip().lower() in _BACKEND_ALIASES:
            return True
    base_url = target.get("base_url") or (config or {}).get("base_url") or ""
    host = (urlparse(str(base_url)).hostname or "").lower()
    return host == "pyannote.ai" or host.endswith(".pyannote.ai")


def model_id_for(diarize_model: str) -> str:
    return f"{MODEL_PREFIX}{diarize_model}"


def pyannote_ai_models() -> list[ModelInfo]:
    """Fixed catalog: pyannote.ai has no /v1/models, just these `model` values."""
    models: list[ModelInfo] = []
    for name in DIARIZE_MODELS:
        raw = {
            "id": model_id_for(name),
            "type": "diarization",
            "engine": PYANNOTE_AI_ENGINE,
            "default": name == DEFAULT_DIARIZE_MODEL,
            "supports_num_speakers_hint": True,
            "produces_transcript": name in TRANSCRIBING_MODELS,
        }
        models.append(
            ModelInfo(
                id=raw["id"],
                type="diarization",
                engine=PYANNOTE_AI_ENGINE,
                is_default=bool(raw["default"]),
                supports_num_speakers_hint=True,
                supports_streaming=False,
                raw=raw,
            )
        )
    return models


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def map_pyannote_output(output: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Map a succeeded pyannote.ai job ``output`` onto live-stt's diarization house JSON."""
    turns_raw = output.get("diarization")
    if not isinstance(turns_raw, list):
        turns_raw = []
    words_raw = output.get("wordLevelTranscription")
    if words_raw is None:
        words_raw = output.get("word_level_transcription")
    turn_text_raw = output.get("turnLevelTranscription")
    if turn_text_raw is None:
        turn_text_raw = output.get("turn_level_transcription")

    # Diarization turns are the segments; fall back to transcript turns if no diarization came back.
    seg_source = turns_raw or (turn_text_raw if isinstance(turn_text_raw, list) else [])
    segments: list[dict[str, Any]] = []
    for t in sorted((t for t in seg_source if isinstance(t, dict)), key=lambda t: _f(t.get("start"))):
        start = _f(t.get("start"))
        end = max(start, _f(t.get("end"), start))
        spk = str(t.get("speaker") or "SPEAKER_00")
        segments.append({"speaker": spk, "label": spk, "start": start, "end": end, "_words": []})

    # Text units to attach: words if available, else transcript turns.
    units_src = words_raw if isinstance(words_raw, list) and words_raw else (
        turn_text_raw if isinstance(turn_text_raw, list) else []
    )
    words: list[dict[str, Any]] = []
    for w in units_src:
        if not isinstance(w, dict):
            continue
        text = str(w.get("text") or "").strip()
        if not text:
            continue
        start = _f(w.get("start"))
        end = max(start, _f(w.get("end"), start))
        words.append({"text": text, "start": start, "end": end, "speaker": str(w.get("speaker") or "")})
    words.sort(key=lambda w: w["start"])

    for w in words:
        best: dict[str, Any] | None = None
        best_key: tuple[int, float, float] | None = None
        mid = 0.5 * (w["start"] + w["end"])
        for seg in segments:
            overlap = max(0.0, min(w["end"], seg["end"]) - max(w["start"], seg["start"]))
            dist = 0.0 if seg["start"] <= mid <= seg["end"] else min(abs(mid - seg["start"]), abs(mid - seg["end"]))
            # Same speaker first, then most overlap, then nearest.
            key = (1 if seg["speaker"] == w["speaker"] else 0, overlap, -dist)
            if best_key is None or key > best_key:
                best, best_key = seg, key
        if best is not None:
            best["_words"].append(w["text"])

    out_segments = []
    for idx, seg in enumerate(segments):
        out_segments.append({
            "id": idx,
            "speaker": seg["speaker"],
            "label": seg["label"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": " ".join(seg["_words"]),
        })

    stats: dict[str, dict[str, float]] = {}
    for seg in out_segments:
        st = stats.setdefault(seg["speaker"], {"dur": 0.0, "n": 0})
        st["dur"] += seg["end"] - seg["start"]
        st["n"] += 1
    speakers = [
        {
            "id": spk,
            "label": spk,
            "total_speech_duration": round(st["dur"], 3),
            "segment_count": int(st["n"]),
        }
        for spk, st in sorted(stats.items(), key=lambda kv: -kv[1]["dur"])
    ]
    duration = max([s["end"] for s in out_segments] + [w["end"] for w in words] + [0.0])

    body: dict[str, Any] = {
        "task": "diarize",
        "model": model_id,
        "num_speakers": len(speakers),
        "segments": out_segments,
        "speakers": speakers,
        "words": words,
        "duration": round(duration, 3),
    }
    warnings = [str(output[k]) for k in ("warning",) if output.get(k)]
    if warnings:
        body["warnings"] = warnings
    return body


class PyannoteCloudClient:
    """Duck-types the parts of ``LiveSTTHTTPClient`` the accuracy_diarization suite uses."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        job_timeout_sec: float = DEFAULT_JOB_TIMEOUT_SEC,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        root = str(base_url or PYANNOTE_AI_BASE_URL).strip().rstrip("/")
        if not root.endswith("/v1"):
            root = f"{root}/v1"
        if transcribe_model not in TRANSCRIBE_MODELS:
            raise LiveSTTClientError(
                f"Unknown pyannote.ai transcribe model {transcribe_model!r}; expected one of {list(TRANSCRIBE_MODELS)}"
            )
        self.root_url = root
        self.api_key = api_key
        self.timeout = float(timeout)
        self.transport = transport
        self.transcribe_model = transcribe_model
        self.poll_interval_sec = max(0.0, float(poll_interval_sec))
        self.job_timeout_sec = max(1.0, float(job_timeout_sec))
        self._sleep = sleep
        self.models_url = f"{root} (built-in pyannote.ai model catalog)"
        self.diarization_url = f"{root}/diarize"
        # Folded into the accuracy cache key: a different STT model is a different result.
        self.cache_tag = f"pyannote_ai:transcribe={transcribe_model}"

    # -- catalog ---------------------------------------------------------------------------
    def discover_models(self) -> DiscoveredModels:
        models = pyannote_ai_models()
        return DiscoveredModels(
            status_code=200,
            latency_ms=0.0,
            raw_body={"object": "list", "data": [m.raw for m in models]},
            diarization_models=models,
        )

    # -- HTTP helpers ----------------------------------------------------------------------
    def _auth(self) -> dict[str, str]:
        if not self.api_key:
            raise LiveSTTClientError(
                "pyannote.ai target has no API key: set the target's auth_env_var (e.g. PYANNOTE_AI_API_KEY) "
                "in test-center's environment"
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    def _request(self, http: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One request with 429 retry (Retry-After honoured, else exponential backoff)."""
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                resp = http.request(method, url, **kwargs)
            except httpx.RequestError as e:
                raise LiveSTTClientError(f"pyannote.ai request {method} {url.split('?')[0]} failed: {e}") from e
            if resp.status_code in (401, 403):
                raise LiveSTTClientError(
                    f"pyannote.ai rejected the API key (status {resp.status_code}) at {url.split('?')[0]}"
                )
            if resp.status_code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                return resp
            retry_after = _f(resp.headers.get("Retry-After"), 0.0)
            self._sleep(retry_after if retry_after > 0 else min(60.0, 2.0 ** attempt))
        return resp  # pragma: no cover - loop always returns

    @staticmethod
    def _error(status_code: int, message: str, latency_ms: float) -> DiarizeResult:
        body = {"error": {"message": message, "type": "pyannote_ai_error"}}
        return DiarizeResult(
            status_code=status_code,
            content_type="application/json",
            body=body,
            raw_text=message,
            latency_ms=latency_ms,
        )

    # -- diarize ---------------------------------------------------------------------------
    def diarize(
        self,
        wav_bytes: bytes,
        model: str | None = None,
        words: list[dict[str, Any]] | None = None,
        include_text: bool = False,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        device: str | None = None,
        response_format: str = "verbose_json",
        filename: str = "fixture.wav",
    ) -> DiarizeResult:
        """Upload, submit, poll, and map one pyannote.ai job. ``words``/``device`` are ignored."""
        model_id = str(model or model_id_for(DEFAULT_DIARIZE_MODEL))
        name = model_id[len(MODEL_PREFIX):] if model_id.startswith(MODEL_PREFIX) else model_id
        t0 = time.perf_counter()

        def elapsed() -> float:
            return round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if name not in DIARIZE_MODELS:
            return self._error(400, f"Unknown pyannote.ai model {model_id!r}; expected one of "
                               f"{[model_id_for(m) for m in DIARIZE_MODELS]}", elapsed())
        auth = self._auth()

        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[:80] or "audio.wav"
        media_url = f"media://test-center/{uuid.uuid4().hex}-{stem}"

        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            # 1. upload slot
            resp = self._request(http, "POST", f"{self.root_url}/media/input", json={"url": media_url},
                                 headers={**auth, "Content-Type": "application/json"})
            if resp.status_code >= 400:
                return self._error(resp.status_code, f"pyannote.ai media/input: {resp.text[:400]}", elapsed())
            try:
                presigned = str(resp.json()["url"])
            except (ValueError, KeyError, TypeError):
                return self._error(502, f"pyannote.ai media/input returned no upload url: {resp.text[:200]}", elapsed())

            # 2. upload (presigned: never send our API key there)
            try:
                put = http.put(presigned, content=wav_bytes, headers={"Content-Type": "application/octet-stream"})
            except httpx.RequestError as e:
                raise LiveSTTClientError(f"pyannote.ai upload failed: {e}") from e
            if put.status_code >= 400:
                return self._error(502, f"pyannote.ai upload returned {put.status_code}: {put.text[:300]}", elapsed())

            # 3. submit
            job: dict[str, Any] = {"url": media_url, "model": name}
            if name in TRANSCRIBING_MODELS:
                job["transcription"] = True
                job["transcriptionConfig"] = {"model": self.transcribe_model}
            if num_speakers is not None:
                job["numSpeakers"] = int(num_speakers)
            else:
                if min_speakers is not None:
                    job["minSpeakers"] = int(min_speakers)
                if max_speakers is not None:
                    job["maxSpeakers"] = int(max_speakers)
            resp = self._request(http, "POST", f"{self.root_url}/diarize", json=job,
                                 headers={**auth, "Content-Type": "application/json"})
            if resp.status_code >= 400:
                return self._error(resp.status_code, f"pyannote.ai diarize: {resp.text[:400]}", elapsed())
            try:
                job_id = str(resp.json()["jobId"])
            except (ValueError, KeyError, TypeError):
                return self._error(502, f"pyannote.ai diarize returned no jobId: {resp.text[:200]}", elapsed())

            # 4. poll
            deadline = time.monotonic() + self.job_timeout_sec
            while True:
                resp = self._request(http, "GET", f"{self.root_url}/jobs/{job_id}", headers=auth)
                if resp.status_code >= 400:
                    return self._error(resp.status_code, f"pyannote.ai job {job_id}: {resp.text[:400]}", elapsed())
                try:
                    status_body = resp.json()
                except ValueError:
                    return self._error(502, f"pyannote.ai job {job_id} returned non-JSON", elapsed())
                status = str(status_body.get("status") or "")
                if status == "succeeded":
                    output = status_body.get("output")
                    if not isinstance(output, dict):
                        return self._error(502, f"pyannote.ai job {job_id} succeeded with no output", elapsed())
                    body = map_pyannote_output(output, model_id)
                    body["pyannote_ai"] = {
                        "job_id": job_id,
                        "model": name,
                        "transcribe_model": self.transcribe_model if name in TRANSCRIBING_MODELS else None,
                    }
                    return DiarizeResult(
                        status_code=200,
                        content_type="application/json",
                        body=body,
                        raw_text="",
                        latency_ms=elapsed(),
                    )
                if status in ("failed", "canceled"):
                    detail = status_body.get("error") or (status_body.get("output") or {}).get("error") or status
                    return self._error(502, f"pyannote.ai job {job_id} {status}: {detail}", elapsed())
                if time.monotonic() > deadline:
                    return self._error(
                        504,
                        f"pyannote.ai job {job_id} still '{status}' after {self.job_timeout_sec:.0f}s "
                        "(raise pyannote_ai_job_timeout_sec)",
                        elapsed(),
                    )
                self._sleep(self.poll_interval_sec)

    def test_connection(self) -> tuple[bool, int | None, float, str]:
        """Authenticate without spending a job: POST /media/input only reserves an upload slot."""
        t0 = time.perf_counter()
        try:
            auth = self._auth()
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                resp = http.post(
                    f"{self.root_url}/media/input",
                    json={"url": f"media://test-center/connection-test-{uuid.uuid4().hex}"},
                    headers={**auth, "Content-Type": "application/json"},
                )
        except (LiveSTTClientError, httpx.RequestError) as e:
            return False, None, round((time.perf_counter() - t0) * 1000.0, 2), str(e)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if resp.status_code in (401, 403):
            return False, resp.status_code, ms, "pyannote.ai rejected this API key"
        if resp.status_code >= 400:
            return False, resp.status_code, ms, f"pyannote.ai returned {resp.status_code}: {resp.text[:300]}"
        return True, resp.status_code, ms, "pyannote.ai API key accepted (no job submitted)"
