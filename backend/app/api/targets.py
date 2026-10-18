"""API routes for managing test targets.

Targets specify the endpoint, kind, capabilities, and authentication (via environment variable reference only)
for external services being tested or evaluated.
Secrets are never stored in the database or returned by the API.
"""

from datetime import datetime, timezone
import os
import re
import time
from typing import Any
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models.target import (
    Target,
    infer_target_capabilities,
    normalize_capabilities,
)
from backend.app.db.session import get_db
from backend.app.testers.embedding.client import EmbeddingClientError, OpenAIEmbeddingClient
from backend.app.testers.live_stt.client import (
    DEFAULT_ASR_MODEL,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_TTS_MODEL,
    LiveSTTClientError,
    LiveSTTHTTPClient,
)
from backend.app.testers.live_stt.fixtures import make_pcm_wav_bytes
from backend.app.testers.live_stt.grpc_client import (
    LiveSTTGRPCClient,
    resolve_grpc_target,
)
from backend.app.testers.llm.client import ChatClientError, OpenAIChatClient

router = APIRouter(prefix="/targets", tags=["targets"])

_ENV_VAR_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STATUS_CODE_REGEX = re.compile(r"\bstatus\s+(\d{3})\b", re.IGNORECASE)
_DISALLOWED_SECRET_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "token",
    "access_token",
    "auth_token",
    "bearer_token",
    "private_key",
}


def _get_http_transport() -> httpx.BaseTransport | None:
    """Return optional custom httpx transport (used for testing/mocking)."""
    return None


def _redact_secret(text: str, secret: str | None) -> str:
    """Ensure raw API key value is never included in error or diagnostic text."""
    if not text:
        return ""
    if secret and secret.strip():
        return text.replace(secret.strip(), "***REDACTED***")
    return text


def _extract_status_code(error_text: str) -> int | None:
    match = _STATUS_CODE_REGEX.search(error_text or "")
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _validate_auth_env_var(value: str | None) -> str | None:
    if value is None:
        return None
    val = value.strip()
    if not val:
        return None
    # Explicitly check for tokens or secrets starting with common credential prefixes
    if val.startswith(("sk-", "ghp_", "xoxb-", "Bearer ", "token-")):
        raise ValueError(
            "API key must be provided by environment variable reference only (e.g. 'OPENAI_API_KEY'), "
            "never a raw secret or token."
        )
    if not _ENV_VAR_REGEX.match(val):
        raise ValueError(
            "auth_env_var must be a valid environment variable reference only (letters, numbers, underscores), "
            "not a raw API key, token, or secret value."
        )
    return val


def _validate_extra(extra_dict: dict[str, Any] | None) -> dict[str, Any]:
    if not extra_dict:
        return {}
    for key, val in extra_dict.items():
        k_lower = key.lower()
        if k_lower in _DISALLOWED_SECRET_KEYS:
            if isinstance(val, str) and len(val.strip()) > 0:
                raise ValueError(
                    f"Direct secrets are forbidden in target extra ('{key}'). "
                    "Use auth_env_var to reference secrets via environment variables."
                )
    port = extra_dict.get("grpc_port")
    if port is not None and not (isinstance(port, str) and not port.strip()):
        try:
            port_int = int(port)
        except (TypeError, ValueError):
            raise ValueError("extra.grpc_port must be an integer between 1 and 65535") from None
        if isinstance(port, bool) or not 1 <= port_int <= 65535:
            raise ValueError("extra.grpc_port must be an integer between 1 and 65535")
    return extra_dict


class TargetBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Unique target name")
    kind: str = Field(..., min_length=1, max_length=64, description="Target kind (e.g. openai-compatible, live-stt, embedding)")
    base_url: str = Field(..., min_length=1, max_length=1024, description="Base URL of target service")
    auth_env_var: str | None = Field(default=None, description="Environment variable name holding API key (reference only)")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional non-secret target parameters")
    capabilities: list[str] | None = Field(
        default=None,
        description="Supported capabilities for this target (e.g. ['chat', 'embedding', 'asr', 'tts', 'diarization'])",
    )

    @field_validator("auth_env_var")
    @classmethod
    def check_auth_env_var(cls, v: str | None) -> str | None:
        return _validate_auth_env_var(v)

    @field_validator("extra")
    @classmethod
    def check_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_extra(v)

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return normalize_capabilities(v)


class TargetCreate(TargetBase):
    pass


class TargetUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    kind: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=1024)
    auth_env_var: str | None = Field(default=None)
    extra: dict[str, Any] | None = Field(default=None)
    capabilities: list[str] | None = Field(default=None)

    @field_validator("auth_env_var")
    @classmethod
    def check_auth_env_var(cls, v: str | None) -> str | None:
        return _validate_auth_env_var(v)

    @field_validator("extra")
    @classmethod
    def check_extra(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            return _validate_extra(v)
        return v

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return normalize_capabilities(v)


class TargetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    kind: str
    base_url: str
    auth_env_var: str | None
    extra: dict[str, Any]
    capabilities: list[str]
    created_at: datetime
    updated_at: datetime


class TargetTestRequest(BaseModel):
    capabilities: list[str] | None = Field(
        default=None,
        description="Capabilities to test (e.g. ['chat', 'embedding']). Defaults to target's capabilities.",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for the test probe.",
    )
    timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="Timeout in seconds for each capability check.",
    )

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return normalize_capabilities(v)


class AdHocTargetTestRequest(BaseModel):
    name: str = Field(default="adhoc-target", min_length=1, max_length=255)
    kind: str = Field(default="openai-compatible", min_length=1, max_length=64)
    base_url: str = Field(..., min_length=1, max_length=1024)
    auth_env_var: str | None = Field(default=None)
    extra: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] | None = Field(default=None)
    model: str | None = Field(default=None)
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=120.0)

    @field_validator("auth_env_var")
    @classmethod
    def check_auth_env_var(cls, v: str | None) -> str | None:
        return _validate_auth_env_var(v)

    @field_validator("extra")
    @classmethod
    def check_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_extra(v)

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        return normalize_capabilities(v)


class CapabilityTestResult(BaseModel):
    capability: str
    ok: bool
    status_code: int | None = None
    latency_ms: float
    model: str | None = None
    message: str
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class TargetTestResponse(BaseModel):
    target_id: int | None = None
    target_name: str
    kind: str
    base_url: str
    auth_env_var: str | None = None
    auth_configured: bool
    ok: bool
    capabilities_tested: list[str]
    results: list[CapabilityTestResult]
    tested_at: datetime


def _probe_chat_capability(
    base_url: str,
    api_key: str | None,
    extra: dict[str, Any],
    model_override: str | None,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> CapabilityTestResult:
    client = OpenAIChatClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
        transport=transport,
    )
    resolved_model = (
        (model_override or "").strip()
        or str(extra.get("chat_model") or extra.get("model") or "").strip()
        or client.detect_model()
        or "gpt-4o-mini"
    )
    t0 = time.perf_counter()
    try:
        content, usage, elapsed_ms = client.chat(
            messages=[{"role": "user", "content": "Reply with OK."}],
            model=resolved_model,
            max_tokens=8,
        )
        return CapabilityTestResult(
            capability="chat",
            ok=True,
            status_code=200,
            latency_ms=round(elapsed_ms, 2),
            model=resolved_model,
            message=f"Chat completion succeeded ({elapsed_ms:.1f} ms)",
            detail={
                "endpoint": client.endpoint_url,
                "response_preview": content[:200],
                "usage": usage,
            },
        )
    except (ChatClientError, Exception) as exc:
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        err_msg = _redact_secret(str(exc), api_key)
        status_code = _extract_status_code(err_msg)
        return CapabilityTestResult(
            capability="chat",
            ok=False,
            status_code=status_code,
            latency_ms=elapsed_ms,
            model=resolved_model,
            message=f"Chat capability check failed ({elapsed_ms:.1f} ms)",
            error=err_msg,
            detail={"endpoint": client.endpoint_url},
        )


def _probe_embedding_capability(
    base_url: str,
    api_key: str | None,
    extra: dict[str, Any],
    model_override: str | None,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> CapabilityTestResult:
    client = OpenAIEmbeddingClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
        transport=transport,
    )
    resolved_model = (
        (model_override or "").strip()
        or str(extra.get("embedding_model") or extra.get("model") or "").strip()
    )
    if not resolved_model:
        chat_helper = OpenAIChatClient(
            base_url=base_url,
            api_key=api_key,
            timeout=min(timeout_seconds, 5.0),
            transport=transport,
        )
        resolved_model = chat_helper.detect_model() or "text-embedding-3-small"

    raw_dims = extra.get("dimensions")
    dimensions = int(raw_dims) if isinstance(raw_dims, (int, float)) and not isinstance(raw_dims, bool) and int(raw_dims) > 0 else None

    t0 = time.perf_counter()
    try:
        vectors, usage, elapsed_ms = client.embed(
            texts=["target connectivity test"],
            model=resolved_model,
            dimensions=dimensions,
        )
        dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.shape[0] > 0 else 0
        return CapabilityTestResult(
            capability="embedding",
            ok=True,
            status_code=200,
            latency_ms=round(elapsed_ms, 2),
            model=resolved_model,
            message=f"Embedding succeeded ({dim}-d vector in {elapsed_ms:.1f} ms)",
            detail={
                "endpoint": client.endpoint_url,
                "dimensions": dim,
                "vectors_count": int(vectors.shape[0]),
                "usage": usage,
            },
        )
    except (EmbeddingClientError, Exception) as exc:
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        err_msg = _redact_secret(str(exc), api_key)
        status_code = _extract_status_code(err_msg)
        return CapabilityTestResult(
            capability="embedding",
            ok=False,
            status_code=status_code,
            latency_ms=elapsed_ms,
            model=resolved_model,
            message=f"Embedding capability check failed ({elapsed_ms:.1f} ms)",
            error=err_msg,
            detail={"endpoint": client.endpoint_url},
        )


def _probe_live_stt_capability(
    capability: str,
    base_url: str,
    api_key: str | None,
    extra: dict[str, Any],
    model_override: str | None,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> CapabilityTestResult:
    client = LiveSTTHTTPClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
        transport=transport,
    )
    t0 = time.perf_counter()
    try:
        discovered = client.discover_models()
        if capability == "tts":
            resolved_model = (
                (model_override or "").strip()
                or str(extra.get("tts_model") or extra.get("model") or "").strip()
                or (discovered.tts_models[0].id if discovered.tts_models else DEFAULT_TTS_MODEL.id)
            )
            res = client.synthesize_speech(
                input_text="Target capability test.",
                model=resolved_model,
                stream=False,
            )
            if res.status_code == 200 and len(res.audio_bytes) > 0:
                return CapabilityTestResult(
                    capability="tts",
                    ok=True,
                    status_code=200,
                    latency_ms=res.latency_ms,
                    model=resolved_model,
                    message=f"TTS synthesis succeeded ({len(res.audio_bytes)} bytes in {res.latency_ms:.1f} ms)",
                    detail={
                        "endpoint": client.speech_url,
                        "audio_bytes": len(res.audio_bytes),
                    },
                )
            err_msg = _redact_secret(str(res.error_body or f"HTTP status {res.status_code}"), api_key)
            return CapabilityTestResult(
                capability="tts",
                ok=False,
                status_code=res.status_code,
                latency_ms=res.latency_ms,
                model=resolved_model,
                message=f"TTS synthesis failed with status {res.status_code}",
                error=err_msg,
                detail={"endpoint": client.speech_url},
            )

        if capability == "diarization":
            resolved_model = (
                (model_override or "").strip()
                or str(extra.get("diarization_model") or extra.get("model") or "").strip()
                or (discovered.diarization_models[0].id if discovered.diarization_models else DEFAULT_DIARIZATION_MODEL.id)
            )
            wav_bytes = make_pcm_wav_bytes(duration_sec=0.5)
            res = client.diarize(wav_bytes=wav_bytes, model=resolved_model)
            if res.status_code == 200:
                return CapabilityTestResult(
                    capability="diarization",
                    ok=True,
                    status_code=200,
                    latency_ms=res.latency_ms,
                    model=resolved_model,
                    message=f"Speaker diarization succeeded ({res.latency_ms:.1f} ms)",
                    detail={"endpoint": client.diarization_url},
                )
            err_msg = _redact_secret(res.raw_text[:300] or f"HTTP status {res.status_code}", api_key)
            return CapabilityTestResult(
                capability="diarization",
                ok=False,
                status_code=res.status_code,
                latency_ms=res.latency_ms,
                model=resolved_model,
                message=f"Speaker diarization failed with status {res.status_code}",
                error=err_msg,
                detail={"endpoint": client.diarization_url},
            )

        # Default live-stt capability: "asr"
        resolved_model = (
            (model_override or "").strip()
            or str(extra.get("asr_model") or extra.get("model") or "").strip()
            or (discovered.asr_models[0].id if discovered.asr_models else DEFAULT_ASR_MODEL.id)
        )
        wav_bytes = make_pcm_wav_bytes(duration_sec=0.25)
        res = client.transcribe_json(wav_bytes=wav_bytes, model=resolved_model)
        if res.status_code == 200:
            text_out = ""
            if isinstance(res.body, dict):
                text_out = str(res.body.get("text") or "")
            return CapabilityTestResult(
                capability="asr",
                ok=True,
                status_code=200,
                latency_ms=res.latency_ms,
                model=resolved_model,
                message=f"ASR transcription succeeded ({res.latency_ms:.1f} ms)",
                detail={
                    "endpoint": client.transcriptions_url,
                    "transcript_preview": text_out[:200],
                    "discovered_asr_models": [m.id for m in discovered.asr_models],
                },
            )
        err_msg = _redact_secret(res.raw_text[:300] or f"HTTP status {res.status_code}", api_key)
        return CapabilityTestResult(
            capability="asr",
            ok=False,
            status_code=res.status_code,
            latency_ms=res.latency_ms,
            model=resolved_model,
            message=f"ASR transcription failed with status {res.status_code}",
            error=err_msg,
            detail={"endpoint": client.transcriptions_url},
        )
    except (LiveSTTClientError, Exception) as exc:
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        err_msg = _redact_secret(str(exc), api_key)
        status_code = _extract_status_code(err_msg)
        return CapabilityTestResult(
            capability=capability,
            ok=False,
            status_code=status_code,
            latency_ms=elapsed_ms,
            model=model_override or extra.get("model"),
            message=f"{capability.upper()} capability check failed ({elapsed_ms:.1f} ms)",
            error=err_msg,
            detail={"base_url": base_url},
        )


def _probe_grpc_capability(
    base_url: str,
    api_key: str | None,
    extra: dict[str, Any],
    timeout_seconds: float,
) -> CapabilityTestResult:
    """Probe the live-stt gRPC endpoint via StreamingASR.GetServerInfo.

    The address comes from ``resolve_grpc_target``: ``extra.grpc_target`` / ``extra.grpc_host`` +
    ``extra.grpc_port``, otherwise the host of ``base_url`` with ``extra.grpc_port`` or 50051.
    """
    t0 = time.perf_counter()
    grpc_addr: str | None = None
    try:
        grpc_addr = resolve_grpc_target({"base_url": base_url, "extra": extra})
        client = LiveSTTGRPCClient(grpc_addr, api_key=api_key, timeout=timeout_seconds)
        info = client.get_server_info_sync(timeout=timeout_seconds)
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        if info.ok:
            return CapabilityTestResult(
                capability="grpc",
                ok=True,
                status_code=None,
                latency_ms=info.latency_ms,
                model=info.default_model or None,
                message=f"gRPC GetServerInfo succeeded at {grpc_addr} ({info.latency_ms:.1f} ms)",
                detail={
                    "grpc_target": grpc_addr,
                    "version": info.version,
                    "backend": info.backend,
                    "default_model": info.default_model,
                    "max_concurrent_calls": info.max_concurrent_calls,
                    "active_calls": info.active_calls,
                },
            )
        return CapabilityTestResult(
            capability="grpc",
            ok=False,
            status_code=None,
            latency_ms=elapsed_ms,
            message=f"gRPC GetServerInfo failed at {grpc_addr} with {info.grpc_status_code}",
            error=_redact_secret(info.error_message or info.grpc_status_code, api_key),
            detail={"grpc_target": grpc_addr},
        )
    except Exception as exc:
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        return CapabilityTestResult(
            capability="grpc",
            ok=False,
            status_code=None,
            latency_ms=elapsed_ms,
            message=f"gRPC capability check failed ({elapsed_ms:.1f} ms)",
            error=_redact_secret(str(exc), api_key),
            detail={"grpc_target": grpc_addr} if grpc_addr else {},
        )


def _probe_generic_http_capability(
    capability: str,
    base_url: str,
    api_key: str | None,
    extra: dict[str, Any],
    model_override: str | None,
    timeout_seconds: float,
    transport: httpx.BaseTransport | None,
) -> CapabilityTestResult:
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/")
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            resp = client.get(url, headers=headers)
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        ok = 200 <= resp.status_code < 300
        return CapabilityTestResult(
            capability=capability,
            ok=ok,
            status_code=resp.status_code,
            latency_ms=elapsed_ms,
            model=model_override or extra.get("model"),
            message=(
                f"Endpoint reachable ({resp.status_code} in {elapsed_ms:.1f} ms)"
                if ok
                else f"Endpoint returned status {resp.status_code}"
            ),
            error=None if ok else _redact_secret(resp.text[:300] or f"HTTP {resp.status_code}", api_key),
            detail={"endpoint": url},
        )
    except Exception as exc:
        elapsed_ms = round(max(0.01, (time.perf_counter() - t0) * 1000.0), 2)
        err_msg = _redact_secret(str(exc), api_key)
        return CapabilityTestResult(
            capability=capability,
            ok=False,
            status_code=None,
            latency_ms=elapsed_ms,
            model=model_override or extra.get("model"),
            message=f"Endpoint check failed ({elapsed_ms:.1f} ms)",
            error=err_msg,
            detail={"endpoint": url},
        )


def _execute_target_test(
    *,
    target_id: int | None,
    target_name: str,
    kind: str,
    base_url: str,
    auth_env_var: str | None,
    extra: dict[str, Any],
    requested_capabilities: list[str] | None,
    model_override: str | None,
    timeout_seconds: float,
) -> TargetTestResponse:
    api_key: str | None = None
    auth_configured = True
    if auth_env_var:
        raw_env = os.environ.get(auth_env_var)
        if raw_env and raw_env.strip():
            api_key = raw_env.strip()
            auth_configured = True
        else:
            auth_configured = False

    capabilities_to_test = normalize_capabilities(requested_capabilities)
    if not capabilities_to_test:
        capabilities_to_test = infer_target_capabilities(kind, extra)

    transport = _get_http_transport()
    results: list[CapabilityTestResult] = []

    for cap in capabilities_to_test:
        if cap == "chat":
            res = _probe_chat_capability(
                base_url=base_url,
                api_key=api_key,
                extra=extra,
                model_override=model_override,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )
        elif cap == "embedding":
            res = _probe_embedding_capability(
                base_url=base_url,
                api_key=api_key,
                extra=extra,
                model_override=model_override,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )
        elif cap == "grpc":
            res = _probe_grpc_capability(
                base_url=base_url,
                api_key=api_key,
                extra=extra,
                timeout_seconds=timeout_seconds,
            )
        elif cap in ("asr", "tts", "diarization"):
            res = _probe_live_stt_capability(
                capability=cap,
                base_url=base_url,
                api_key=api_key,
                extra=extra,
                model_override=model_override,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )
        else:
            res = _probe_generic_http_capability(
                capability=cap,
                base_url=base_url,
                api_key=api_key,
                extra=extra,
                model_override=model_override,
                timeout_seconds=timeout_seconds,
                transport=transport,
            )
        results.append(res)

    overall_ok = len(results) > 0 and all(r.ok for r in results)
    return TargetTestResponse(
        target_id=target_id,
        target_name=target_name,
        kind=kind,
        base_url=base_url,
        auth_env_var=auth_env_var,
        auth_configured=auth_configured,
        ok=overall_ok,
        capabilities_tested=capabilities_to_test,
        results=results,
        tested_at=datetime.now(timezone.utc),
    )


@router.post("", response_model=TargetResponse, status_code=status.HTTP_201_CREATED)
def create_target(
    req: TargetCreate,
    db: Session = Depends(get_db),
) -> Target:
    """Create a new test target.
    
    API keys must only be referenced by their environment variable name via `auth_env_var`.
    Secrets are never stored or returned by the API.
    """
    # Check uniqueness of name
    existing = db.execute(select(Target).where(Target.name == req.name)).scalars().first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Target with name '{req.name}' already exists",
        )

    extra_data = dict(req.extra or {})
    if req.capabilities is not None:
        extra_data["capabilities"] = req.capabilities
    elif "capabilities" in extra_data and isinstance(extra_data["capabilities"], list):
        extra_data["capabilities"] = normalize_capabilities([str(c) for c in extra_data["capabilities"]])

    target = Target(
        name=req.name,
        kind=req.kind,
        base_url=req.base_url,
        auth_env_var=req.auth_env_var,
        extra=extra_data,
    )
    db.add(target)
    db.commit()
    db.refresh(target)
    return target


@router.get("", response_model=list[TargetResponse])
def list_targets(
    kind: str | None = Query(None, description="Filter targets by kind"),
    limit: int = Query(50, ge=1, le=200, description="Max targets to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> list[Target]:
    """List test targets with optional filtering by kind."""
    stmt = select(Target).order_by(Target.id.asc()).offset(offset).limit(limit)
    if kind:
        stmt = stmt.where(Target.kind == kind)
    return list(db.execute(stmt).scalars().all())


@router.post("/test", response_model=TargetTestResponse)
def test_adhoc_target(
    req: AdHocTargetTestRequest,
) -> TargetTestResponse:
    """Test connectivity and selected capabilities for an unsaved/ad-hoc target configuration."""
    return _execute_target_test(
        target_id=None,
        target_name=req.name,
        kind=req.kind,
        base_url=req.base_url,
        auth_env_var=req.auth_env_var,
        extra=req.extra,
        requested_capabilities=req.capabilities,
        model_override=req.model,
        timeout_seconds=req.timeout_seconds,
    )


@router.get("/{target_id}", response_model=TargetResponse)
def get_target(
    target_id: int,
    db: Session = Depends(get_db),
) -> Target:
    """Get details of a specific target by ID."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")
    return target


@router.post("/{target_id}/test", response_model=TargetTestResponse)
def test_target(
    target_id: int,
    req: TargetTestRequest | None = Body(default=None),
    db: Session = Depends(get_db),
) -> TargetTestResponse:
    """Test connectivity and capabilities (e.g. chat, embedding, asr, tts, diarization) for a saved target."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    test_req = req or TargetTestRequest()
    return _execute_target_test(
        target_id=target.id,
        target_name=target.name,
        kind=target.kind,
        base_url=target.base_url,
        auth_env_var=target.auth_env_var,
        extra=target.extra or {},
        requested_capabilities=test_req.capabilities,
        model_override=test_req.model,
        timeout_seconds=test_req.timeout_seconds,
    )


@router.put("/{target_id}", response_model=TargetResponse)
@router.patch("/{target_id}", response_model=TargetResponse)
def update_target(
    target_id: int,
    req: TargetUpdate,
    db: Session = Depends(get_db),
) -> Target:
    """Update target fields."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    update_data = req.model_dump(exclude_unset=True)
    capabilities_update = update_data.pop("capabilities", None)

    # Check name collision if name is being changed
    if "name" in update_data and update_data["name"] != target.name:
        existing = db.execute(select(Target).where(Target.name == update_data["name"])).scalars().first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Target with name '{update_data['name']}' already exists",
            )

    for field, val in update_data.items():
        setattr(target, field, val)

    if capabilities_update is not None:
        merged_extra = dict(target.extra or {})
        merged_extra["capabilities"] = capabilities_update
        target.extra = merged_extra
    elif "extra" in update_data and isinstance(target.extra, dict) and isinstance(target.extra.get("capabilities"), list):
        merged_extra = dict(target.extra)
        merged_extra["capabilities"] = normalize_capabilities([str(c) for c in merged_extra["capabilities"]])
        target.extra = merged_extra

    db.commit()
    db.refresh(target)
    return target


@router.delete("/{target_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_target(
    target_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a target. References from test definitions are automatically set to NULL."""
    target = db.get(Target, target_id)
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target {target_id} not found")

    db.delete(target)
    db.commit()

