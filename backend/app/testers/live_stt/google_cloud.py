"""Google Cloud speech-to-text with diarization as a live-stt tester backend.

Lets the live-stt tester's ``accuracy_diarization`` suite score two Google models next to live-stt's
own diarizers and pyannote.ai, so runs line up in Compare:

  - ``google/gemini-3.5-transcribe`` -- Vertex AI (Gemini Enterprise Agent Platform)
    ``generateContent`` with ``audioTranscriptionConfig{diarization, wordTimestamp}``. Audio is sent
    inline; Vertex caps diarization at 15 min of audio per request. Each response part is one
    speaker turn: ``audioTranscription{speakerLabel, words[{word, startOffset, endOffset}]}``.
  - ``google/chirp_3`` -- Cloud Speech-to-Text V2, ``features.diarizationConfig``. Audio up to
    ``SYNC_MAX_SEC`` goes to synchronous ``recognize`` inline; longer audio needs a GCS bucket
    (``gcs_bucket``): upload, ``batchRecognize`` with an inline response, poll the operation, delete
    the object. Words carry ``speakerLabel``; turns are runs of same-speaker words.

Auth is a service-account key (role ``Vertex AI User`` for Gemini; ``Cloud Speech Client`` for
Chirp 3, plus object create/delete on the bucket for long audio). The target's ``auth_env_var``
names an env var holding the key file path, the key JSON itself, or the JSON base64-encoded. The
project defaults to the key's ``project_id``.

Both models are billed per audio second, so the tester's SQLite result cache matters: re-runs over
the same meetings are free. Note both models were observed to emit short invented phrases on pure
silence ("Good morning."), which shows up as insertions in WER, not as a client error.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
import wave
from typing import Any, Callable
from urllib.parse import quote, urlparse

import httpx

from backend.app.testers.live_stt.client import (
    DiarizeResult,
    DiscoveredModels,
    LiveSTTClientError,
    ModelInfo,
)

GOOGLE_CLOUD_ENGINE = "google_cloud"
MODEL_PREFIX = "google/"
GEMINI_TRANSCRIBE = "gemini-3.5-transcribe"
CHIRP_3 = "chirp_3"
DIARIZE_MODELS = (GEMINI_TRANSCRIBE, CHIRP_3)
DEFAULT_DIARIZE_MODEL = GEMINI_TRANSCRIBE

DEFAULT_GEMINI_LOCATION = "global"  # gemini-3.5-transcribe: global, us, eu
DEFAULT_CHIRP_LOCATION = "us"  # chirp_3 is not served from us-central1
SYNC_MAX_SEC = 59.0  # Speech V2 synchronous recognize accepts at most 60 s of audio
DEFAULT_POLL_INTERVAL_SEC = 5.0
DEFAULT_JOB_TIMEOUT_SEC = 900.0
MAX_RETRIES = 5
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
_BACKEND_ALIASES = frozenset({
    "google-cloud", "google_cloud", "googlecloud", "gcp", "vertex-ai", "vertex_ai", "vertex",
})
_HOST_SUFFIXES = ("aiplatform.googleapis.com", "speech.googleapis.com")


def is_google_cloud_target(target: dict[str, Any], config: dict[str, Any] | None = None) -> bool:
    """True if a target points at Google Cloud speech (by kind, extra.backend, or API host)."""
    extra = target.get("extra") if isinstance(target.get("extra"), dict) else {}
    for value in (target.get("kind"), extra.get("backend"), (config or {}).get("backend")):
        if isinstance(value, str) and value.strip().lower() in _BACKEND_ALIASES:
            return True
    base_url = target.get("base_url") or (config or {}).get("base_url") or ""
    host = (urlparse(str(base_url)).hostname or "").lower()
    return any(host == s or host.endswith("." + s) or host.endswith("-" + s) for s in _HOST_SUFFIXES)


def model_id_for(name: str) -> str:
    return f"{MODEL_PREFIX}{name}"


def google_cloud_models() -> list[ModelInfo]:
    """Fixed catalog. Both models transcribe and diarize in one pass."""
    models: list[ModelInfo] = []
    for name in DIARIZE_MODELS:
        raw = {
            "id": model_id_for(name),
            "type": "diarization",
            "engine": GOOGLE_CLOUD_ENGINE,
            "default": name == DEFAULT_DIARIZE_MODEL,
            # Chirp 3 takes min/max speaker counts; gemini-3.5-transcribe has no speaker-count hint.
            "supports_num_speakers_hint": name == CHIRP_3,
            "produces_transcript": True,
        }
        models.append(
            ModelInfo(
                id=raw["id"],
                type="diarization",
                engine=GOOGLE_CLOUD_ENGINE,
                is_default=bool(raw["default"]),
                supports_num_speakers_hint=bool(raw["supports_num_speakers_hint"]),
                supports_streaming=False,
                raw=raw,
            )
        )
    return models


def load_service_account_info(value: str | None) -> dict[str, Any]:
    """Parse a service-account key given as a file path, raw JSON, or base64-encoded JSON."""
    if not value or not str(value).strip():
        raise LiveSTTClientError(
            "Google Cloud target has no credentials: set the target's auth_env_var (e.g. "
            "GOOGLE_APPLICATION_CREDENTIALS) to an env var holding the service-account key file path or JSON"
        )
    text = str(value).strip()
    info: Any = None
    if text.startswith("{"):
        try:
            info = json.loads(text)
        except ValueError as e:
            raise LiveSTTClientError(f"Google Cloud credentials look like JSON but do not parse: {e}") from e
    elif os.path.isfile(os.path.expanduser(text)):
        try:
            with open(os.path.expanduser(text), encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError) as e:
            raise LiveSTTClientError(f"Cannot read Google Cloud key file {text!r}: {e}") from e
    else:
        try:
            info = json.loads(base64.b64decode(text, validate=True))
        except (ValueError, TypeError):
            raise LiveSTTClientError(
                "Google Cloud credentials are neither an existing key file path, JSON, nor base64 JSON"
            ) from None
    if not isinstance(info, dict) or info.get("type") != "service_account" or not info.get("private_key"):
        raise LiveSTTClientError("Google Cloud credentials are not a service-account key (type 'service_account')")
    return info


def _service_account_token_provider(info: dict[str, Any]) -> Callable[[], str]:
    """OAuth2 access tokens from a service-account key, refreshed shortly before expiry."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as e:  # pragma: no cover - dependency is in requirements.txt
        raise LiveSTTClientError("google-auth is not installed (pip install google-auth requests)") from e
    creds = service_account.Credentials.from_service_account_info(info, scopes=list(_SCOPES))

    def token() -> str:
        if not creds.valid:
            try:
                creds.refresh(Request())
            except Exception as e:  # google.auth.exceptions.RefreshError and transport errors
                raise LiveSTTClientError(f"Google Cloud rejected the service-account key: {e}") from e
        return str(creds.token)

    return token


def _f(value: Any, default: float = 0.0) -> float:
    """Seconds from a number or a protobuf Duration string such as ``"2.400s"``."""
    if isinstance(value, str):
        value = value.strip().rstrip("s")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def wav_duration_sec(wav_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            rate = w.getframerate()
            return w.getnframes() / float(rate) if rate else None
    except (wave.Error, EOFError):
        return None


def _speaker(label: Any) -> str:
    """Provider labels ('spk:0', '1', 'spk_2') -> 'SPEAKER_00'-style; blank -> 'SPEAKER_UNKNOWN'."""
    text = str(label or "").strip()
    digits = "".join(c for c in text if c.isdigit())
    if digits:
        return f"SPEAKER_{int(digits):02d}"
    return f"SPEAKER_{text.upper()}" if text else "SPEAKER_UNKNOWN"


def _house_body(model_id: str, turns: list[dict[str, Any]], words: list[dict[str, Any]]) -> dict[str, Any]:
    """Build live-stt diarization JSON from speaker turns ({speaker,start,end,text}) and words."""
    turns = sorted((t for t in turns if t["end"] >= t["start"]), key=lambda t: (t["start"], t["end"]))
    segments = [
        {
            "id": i,
            "speaker": t["speaker"],
            "label": t["speaker"],
            "start": round(t["start"], 3),
            "end": round(t["end"], 3),
            "text": t["text"],
        }
        for i, t in enumerate(turns)
    ]
    stats: dict[str, dict[str, float]] = {}
    for seg in segments:
        st = stats.setdefault(seg["speaker"], {"dur": 0.0, "n": 0})
        st["dur"] += seg["end"] - seg["start"]
        st["n"] += 1
    speakers = [
        {"id": spk, "label": spk, "total_speech_duration": round(st["dur"], 3), "segment_count": int(st["n"])}
        for spk, st in sorted(stats.items(), key=lambda kv: -kv[1]["dur"])
    ]
    words = sorted(words, key=lambda w: w["start"])
    duration = max([s["end"] for s in segments] + [w["end"] for w in words] + [0.0])
    return {
        "task": "diarize",
        "model": model_id,
        "num_speakers": len(speakers),
        "segments": segments,
        "speakers": speakers,
        "words": words,
        "duration": round(duration, 3),
    }


def map_gemini_response(resp: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Map a gemini-3.5-transcribe ``generateContent`` response: one part per speaker turn."""
    candidates = resp.get("candidates") or []
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") if candidates else None
    turns: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for part in parts or []:
        if not isinstance(part, dict):
            continue
        at = part.get("audioTranscription") or part.get("audio_transcription")
        if not isinstance(at, dict):
            continue
        spk = _speaker(at.get("speakerLabel") or at.get("speaker_label"))
        turn_words = []
        for w in at.get("words") or []:
            if not isinstance(w, dict):
                continue
            text = str(w.get("word") or w.get("text") or "").strip()
            if not text:
                continue
            start = _f(w.get("startOffset", w.get("start_offset")))
            end = max(start, _f(w.get("endOffset", w.get("end_offset")), start))
            wspk = w.get("speakerLabel") or w.get("speaker_label")
            turn_words.append({"text": text, "start": start, "end": end, "speaker": _speaker(wspk) if wspk else spk})
        text = str(at.get("text") or part.get("text") or "").strip() or " ".join(w["text"] for w in turn_words)
        if not text:
            continue
        if turn_words:
            start, end = turn_words[0]["start"], max(w["end"] for w in turn_words)
        else:  # no word timestamps: place the turn after the previous one so ordering survives
            start = end = turns[-1]["end"] if turns else 0.0
        turns.append({"speaker": spk, "start": start, "end": end, "text": text})
        words.extend(turn_words)
    body = _house_body(model_id, turns, words)
    usage = resp.get("usageMetadata")
    if isinstance(usage, dict):
        body["usage"] = usage
    return body


def map_chirp_results(results: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Map Speech V2 ``results`` (sync or batch inline): runs of same-speaker words become turns."""
    words: list[dict[str, Any]] = []
    for res in results or []:
        if not isinstance(res, dict):
            continue
        alts = res.get("alternatives") or []
        alt = alts[0] if alts and isinstance(alts[0], dict) else {}
        for w in alt.get("words") or []:
            if not isinstance(w, dict):
                continue
            text = str(w.get("word") or "").strip()
            if not text:
                continue
            start = _f(w.get("startOffset"))
            end = max(start, _f(w.get("endOffset"), start))
            words.append({"text": text, "start": start, "end": end, "speaker": _speaker(w.get("speakerLabel"))})
    words.sort(key=lambda w: w["start"])
    turns: list[dict[str, Any]] = []
    for w in words:
        if turns and turns[-1]["speaker"] == w["speaker"]:
            turns[-1]["end"] = max(turns[-1]["end"], w["end"])
            turns[-1]["_w"].append(w["text"])
        else:
            turns.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "_w": [w["text"]]})
    for t in turns:
        t["text"] = " ".join(t.pop("_w"))
    return _house_body(model_id, turns, words)


def _language_code(language: str | None) -> str:
    lang = (language or "en").strip()
    if "-" in lang:
        return lang
    return {"en": "en-US", "es": "es-ES", "fr": "fr-FR", "de": "de-DE", "ja": "ja-JP", "zh": "cmn-Hans-CN"}.get(
        lang.lower(), lang
    )


class GoogleCloudClient:
    """Duck-types the parts of ``LiveSTTHTTPClient`` the accuracy_diarization suite uses."""

    def __init__(
        self,
        credentials: str | None = None,
        project: str | None = None,
        gemini_location: str = DEFAULT_GEMINI_LOCATION,
        chirp_location: str = DEFAULT_CHIRP_LOCATION,
        gcs_bucket: str | None = None,
        language: str | None = "en",
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        job_timeout_sec: float = DEFAULT_JOB_TIMEOUT_SEC,
        token_provider: Callable[[], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._credentials = credentials
        self._token_provider = token_provider
        self._project = (project or "").strip() or None
        self.gemini_location = (gemini_location or DEFAULT_GEMINI_LOCATION).strip()
        self.chirp_location = (chirp_location or DEFAULT_CHIRP_LOCATION).strip()
        self.gcs_bucket = (gcs_bucket or "").strip().removeprefix("gs://").strip("/") or None
        self.language_code = _language_code(language)
        self.timeout = float(timeout)
        self.transport = transport
        self.poll_interval_sec = max(0.0, float(poll_interval_sec))
        self.job_timeout_sec = max(1.0, float(job_timeout_sec))
        self._sleep = sleep
        self.root_url = "https://aiplatform.googleapis.com (google cloud speech)"
        self.models_url = "built-in Google Cloud model catalog"
        self.diarization_url = "Vertex AI generateContent / Speech-to-Text V2 recognize"
        # Folded into the accuracy cache key: same meeting, different project/region/language is a new result.
        self.cache_tag = (
            f"google_cloud:gemini@{self.gemini_location}:chirp@{self.chirp_location}:lang={self.language_code}"
        )

    # -- auth ------------------------------------------------------------------------------
    def _resolve_auth(self) -> None:
        if self._token_provider is not None and self._project:
            return
        info = load_service_account_info(self._credentials)
        if not self._project:
            self._project = str(info.get("project_id") or "").strip() or None
        if not self._project:
            raise LiveSTTClientError("Google Cloud key has no project_id; set the target's extra.project")
        if self._token_provider is None:
            self._token_provider = _service_account_token_provider(info)

    @property
    def project(self) -> str:
        self._resolve_auth()
        assert self._project
        return self._project

    def _auth(self) -> dict[str, str]:
        self._resolve_auth()
        assert self._token_provider is not None
        return {"Authorization": f"Bearer {self._token_provider()}"}

    # -- endpoints -------------------------------------------------------------------------
    def _gemini_url(self) -> str:
        loc = self.gemini_location
        host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
        return (f"https://{host}/v1/projects/{self.project}/locations/{loc}"
                f"/publishers/google/models/{GEMINI_TRANSCRIBE}:generateContent")

    def _speech_root(self) -> str:
        loc = self.chirp_location
        host = "speech.googleapis.com" if loc == "global" else f"{loc}-speech.googleapis.com"
        return f"https://{host}/v2"

    def _recognizer(self) -> str:
        return f"projects/{self.project}/locations/{self.chirp_location}/recognizers/_"

    # -- catalog ---------------------------------------------------------------------------
    def discover_models(self) -> DiscoveredModels:
        models = google_cloud_models()
        return DiscoveredModels(
            status_code=200,
            latency_ms=0.0,
            raw_body={"object": "list", "data": [m.raw for m in models]},
            diarization_models=models,
        )

    # -- HTTP helpers ----------------------------------------------------------------------
    def _request(self, http: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """One request with retry on 429/5xx (Retry-After honoured, else exponential backoff)."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = http.request(method, url, **kwargs)
            except httpx.RequestError as e:
                raise LiveSTTClientError(f"Google Cloud request {method} {url.split('?')[0]} failed: {e}") from e
            if resp.status_code not in _RETRY_STATUS or attempt == MAX_RETRIES:
                return resp
            retry_after = _f(resp.headers.get("Retry-After"), 0.0)
            self._sleep(retry_after if retry_after > 0 else min(60.0, 2.0 ** attempt))
        return resp  # pragma: no cover - loop always returns

    @staticmethod
    def _api_message(resp: httpx.Response) -> str:
        try:
            err = resp.json().get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
        except (ValueError, AttributeError):
            pass
        return resp.text[:500]

    @staticmethod
    def _error(status_code: int, message: str, latency_ms: float) -> DiarizeResult:
        body = {"error": {"message": message, "type": "google_cloud_error"}}
        return DiarizeResult(
            status_code=status_code, content_type="application/json", body=body, raw_text=message,
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
        """Transcribe + diarize one WAV with Gemini 3.5 Transcribe or Chirp 3. ``words``/``device`` ignored."""
        model_id = str(model or model_id_for(DEFAULT_DIARIZE_MODEL))
        name = model_id[len(MODEL_PREFIX):] if model_id.startswith(MODEL_PREFIX) else model_id
        t0 = time.perf_counter()

        def elapsed() -> float:
            return round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)

        if name not in DIARIZE_MODELS:
            return self._error(400, f"Unknown Google Cloud model {model_id!r}; expected one of "
                               f"{[model_id_for(m) for m in DIARIZE_MODELS]}", elapsed())
        auth = self._auth()
        with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
            if name == GEMINI_TRANSCRIBE:
                result = self._diarize_gemini(http, auth, wav_bytes, model_id, elapsed)
            else:
                if num_speakers is not None:
                    min_speakers = max_speakers = num_speakers
                result = self._diarize_chirp(http, auth, wav_bytes, model_id, min_speakers, max_speakers,
                                             filename, elapsed)
        return result

    def _ok(self, body: dict[str, Any], extra: dict[str, Any], latency_ms: float) -> DiarizeResult:
        body["google_cloud"] = {"project": self.project, **extra}
        return DiarizeResult(status_code=200, content_type="application/json", body=body, raw_text="",
                             latency_ms=latency_ms)

    def _diarize_gemini(self, http: httpx.Client, auth: dict[str, str], wav_bytes: bytes, model_id: str,
                        elapsed: Callable[[], float]) -> DiarizeResult:
        req = {
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_bytes).decode("ascii")}},
            ]}],
            "generationConfig": {"audioTranscriptionConfig": {
                "diarization": True, "wordTimestamp": True, "languageCodes": [self.language_code],
            }},
        }
        resp = self._request(http, "POST", self._gemini_url(), json=req, headers=auth)
        if resp.status_code >= 400:
            return self._error(resp.status_code, f"Gemini {GEMINI_TRANSCRIBE}: {self._api_message(resp)}", elapsed())
        try:
            data = resp.json()
        except ValueError:
            return self._error(502, f"Gemini {GEMINI_TRANSCRIBE} returned non-JSON", elapsed())
        body = map_gemini_response(data, model_id)
        return self._ok(body, {"model": GEMINI_TRANSCRIBE, "location": self.gemini_location}, elapsed())

    def _chirp_config(self, min_speakers: int | None, max_speakers: int | None) -> dict[str, Any]:
        diar: dict[str, Any] = {}
        if min_speakers is not None:
            diar["minSpeakerCount"] = int(min_speakers)
        if max_speakers is not None:
            diar["maxSpeakerCount"] = int(max_speakers)
        return {
            "autoDecodingConfig": {},
            "model": CHIRP_3,
            "languageCodes": [self.language_code],
            "features": {"diarizationConfig": diar, "enableWordTimeOffsets": True},
        }

    def _diarize_chirp(self, http: httpx.Client, auth: dict[str, str], wav_bytes: bytes, model_id: str,
                       min_speakers: int | None, max_speakers: int | None, filename: str,
                       elapsed: Callable[[], float]) -> DiarizeResult:
        config = self._chirp_config(min_speakers, max_speakers)
        duration = wav_duration_sec(wav_bytes)
        meta = {"model": CHIRP_3, "location": self.chirp_location}
        if duration is None or duration <= SYNC_MAX_SEC:  # unreadable header: let the API decide
            url = f"{self._speech_root()}/{self._recognizer()}:recognize"
            req = {"config": config, "content": base64.b64encode(wav_bytes).decode("ascii")}
            resp = self._request(http, "POST", url, json=req, headers=auth)
            if resp.status_code >= 400:
                return self._error(resp.status_code, f"Chirp 3 recognize: {self._api_message(resp)}", elapsed())
            body = map_chirp_results(resp.json().get("results") or [], model_id)
            return self._ok(body, {**meta, "mode": "sync"}, elapsed())

        if not self.gcs_bucket:
            return self._error(
                400,
                f"Chirp 3 synchronous recognize accepts at most 60 s of audio (this file is "
                f"{duration or 0:.0f} s). Set the target's extra.gcs_bucket for batch mode, or cap the "
                "run with max_audio_duration_sec <= 59.",
                elapsed(),
            )
        return self._diarize_chirp_batch(http, auth, wav_bytes, model_id, config, filename, meta, elapsed)

    def _diarize_chirp_batch(self, http: httpx.Client, auth: dict[str, str], wav_bytes: bytes, model_id: str,
                             config: dict[str, Any], filename: str, meta: dict[str, Any],
                             elapsed: Callable[[], float]) -> DiarizeResult:
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[:80] or "audio.wav"
        obj = f"test-center/{uuid.uuid4().hex}-{stem}"
        bucket = quote(self.gcs_bucket or "", safe="")
        up = self._request(
            http, "POST",
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={quote(obj, safe='')}",
            content=wav_bytes, headers={**auth, "Content-Type": "audio/wav"},
        )
        if up.status_code >= 400:
            return self._error(up.status_code, f"GCS upload to gs://{self.gcs_bucket}: {self._api_message(up)}",
                               elapsed())
        uri = f"gs://{self.gcs_bucket}/{obj}"
        try:
            req = {"config": config, "files": [{"uri": uri}],
                   "recognitionOutputConfig": {"inlineResponseConfig": {}}}
            resp = self._request(http, "POST", f"{self._speech_root()}/{self._recognizer()}:batchRecognize",
                                 json=req, headers=auth)
            if resp.status_code >= 400:
                return self._error(resp.status_code, f"Chirp 3 batchRecognize: {self._api_message(resp)}", elapsed())
            op = resp.json()
            op_name = str(op.get("name") or "")
            deadline = time.monotonic() + self.job_timeout_sec
            while not op.get("done"):
                if time.monotonic() > deadline:
                    return self._error(504, f"Chirp 3 operation {op_name} not done after "
                                       f"{self.job_timeout_sec:.0f}s (raise google_job_timeout_sec)", elapsed())
                self._sleep(self.poll_interval_sec)
                resp = self._request(http, "GET", f"{self._speech_root()}/{op_name}", headers=auth)
                if resp.status_code >= 400:
                    return self._error(resp.status_code, f"Chirp 3 operation {op_name}: {self._api_message(resp)}",
                                       elapsed())
                op = resp.json()
            if op.get("error"):
                return self._error(502, f"Chirp 3 operation {op_name} failed: {op['error']}", elapsed())
            file_result = ((op.get("response") or {}).get("results") or {}).get(uri) or {}
            if file_result.get("error"):
                return self._error(502, f"Chirp 3 failed on {uri}: {file_result['error']}", elapsed())
            transcript = (file_result.get("inlineResult") or {}).get("transcript") or {}
            body = map_chirp_results(transcript.get("results") or [], model_id)
            return self._ok(body, {**meta, "mode": "batch", "operation": op_name}, elapsed())
        finally:
            try:
                http.request("DELETE", f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{quote(obj, safe='')}",
                             headers=auth)
            except httpx.RequestError:
                pass  # best effort; a bucket lifecycle rule can sweep test-center/ leftovers

    def test_connection(self) -> tuple[bool, int | None, float, str]:
        """Mint a token and read gemini-3.5-transcribe's model card. Free: no audio is submitted."""
        t0 = time.perf_counter()
        try:
            auth = self._auth()
            loc = self.gemini_location
            host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
            with httpx.Client(timeout=min(self.timeout, 30.0), transport=self.transport) as http:
                resp = http.get(f"https://{host}/v1beta1/publishers/google/models/{GEMINI_TRANSCRIBE}", headers=auth)
        except (LiveSTTClientError, httpx.RequestError) as e:
            return False, None, round((time.perf_counter() - t0) * 1000.0, 2), str(e)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if resp.status_code >= 400:
            return False, resp.status_code, ms, f"Google Cloud returned {resp.status_code}: {self._api_message(resp)}"
        return True, resp.status_code, ms, (
            f"Service-account key accepted for project {self.project}; {GEMINI_TRANSCRIBE} visible "
            "(no audio submitted)"
        )
