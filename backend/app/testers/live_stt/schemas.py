"""Response schema and contract validators for live-stt HTTP endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from typing import Any
import wave


@dataclass
class SchemaCheckResult:
    """Outcome of validating an HTTP response against its expected schema."""
    valid: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def summary_message(self) -> str:
        if self.valid:
            return f"Passed all {len(self.checks)} schema checks"
        return "Schema validation failed: " + "; ".join(self.errors)


def _record_check(
    checks: list[dict[str, Any]],
    errors: list[str],
    name: str,
    passed: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def validate_models_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
    strict_catalog: bool = False,
) -> SchemaCheckResult:
    """Validate GET /v1/models OpenAI-style model catalog response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    if strict_catalog or status_code != 200:
        _record_check(
            checks,
            errors,
            "http_status_200",
            status_code == 200,
            f"HTTP status {status_code}",
        )
        _record_check(
            checks,
            errors,
            "content_type_json",
            "application/json" in (content_type or "").lower(),
            f"Content-Type={content_type!r}",
        )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    obj_type = body.get("object")
    _record_check(
        checks,
        errors,
        "object_is_list",
        obj_type == "list",
        f"object={obj_type!r}",
    )

    data = body.get("data")
    is_list = isinstance(data, list) and len(data) > 0
    _record_check(
        checks,
        errors,
        "data_is_non_empty_list",
        is_list,
        f"Found {len(data)} models" if isinstance(data, list) else f"Expected list, got {type(data).__name__}",
    )

    if isinstance(data, list):
        all_have_id = all(
            isinstance(item, dict) and isinstance(item.get("id"), str) and bool(item["id"].strip())
            for item in data
        )
        _record_check(
            checks,
            errors,
            "items_have_valid_id",
            all_have_id,
            "All model entries have non-empty string 'id'"
            if all_have_id
            else "One or more model entries missing valid 'id'",
        )

        if strict_catalog and len(data) > 0:
            all_have_obj = all(
                isinstance(item, dict) and item.get("object") == "model"
                for item in data
            )
            _record_check(
                checks,
                errors,
                "items_have_model_object",
                all_have_obj,
                "All model entries have object='model'"
                if all_have_obj
                else "One or more model entries missing object='model'",
            )

            valid_types = {"asr", "diarization", "tts"}
            all_have_type = all(
                isinstance(item, dict) and item.get("type") in valid_types
                for item in data
            )
            _record_check(
                checks,
                errors,
                "items_have_valid_type",
                all_have_type,
                "All model entries have type in {'asr', 'diarization', 'tts'}"
                if all_have_type
                else "One or more model entries missing valid 'type'",
            )

            asr_items = [
                item for item in data
                if isinstance(item, dict) and item.get("type") == "asr"
            ]
            has_asr = len(asr_items) > 0
            _record_check(
                checks,
                errors,
                "catalog_has_asr_models",
                has_asr,
                f"Found {len(asr_items)} ASR model(s)",
            )
            if has_asr:
                asr_meta_ok = all(
                    isinstance(item.get("engine"), str)
                    and bool(item["engine"].strip())
                    and isinstance(item.get("default"), bool)
                    for item in asr_items
                )
                _record_check(
                    checks,
                    errors,
                    "asr_models_have_engine_and_default",
                    asr_meta_ok,
                    "All ASR models define string 'engine' and boolean 'default'"
                    if asr_meta_ok
                    else "One or more ASR models missing 'engine' or boolean 'default'",
                )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_transcription_json_response(
    body: Any,
    content_type: str = "application/json",
    status_code: int = 200,
) -> SchemaCheckResult:
    """Validate POST /v1/audio/transcriptions (response_format='json') response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    has_no_error = "error" not in body
    _record_check(
        checks,
        errors,
        "no_error_field",
        has_no_error,
        "No 'error' field present" if has_no_error else f"Error field present: {body.get('error')}",
    )

    has_text = "text" in body and isinstance(body.get("text"), str)
    _record_check(
        checks,
        errors,
        "text_field_is_string",
        has_text,
        f"text={body.get('text')!r}" if has_text else "Missing or non-string 'text' field",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_transcription_verbose_json_response(
    body: Any,
    content_type: str = "application/json",
    status_code: int = 200,
) -> SchemaCheckResult:
    """Validate POST /v1/audio/transcriptions (response_format='verbose_json') response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    task_ok = isinstance(body.get("task"), str) and len(body["task"]) > 0
    _record_check(
        checks,
        errors,
        "task_field",
        task_ok,
        f"task={body.get('task')!r}",
    )

    lang_ok = isinstance(body.get("language"), str)
    _record_check(
        checks,
        errors,
        "language_field",
        lang_ok,
        f"language={body.get('language')!r}",
    )

    dur = body.get("duration")
    dur_ok = isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur >= 0
    _record_check(
        checks,
        errors,
        "duration_field",
        dur_ok,
        f"duration={dur!r}",
    )

    text_ok = isinstance(body.get("text"), str)
    _record_check(
        checks,
        errors,
        "text_field",
        text_ok,
        f"text={body.get('text')!r}" if text_ok else "Missing or non-string 'text' field",
    )

    words = body.get("words")
    words_is_list = isinstance(words, list)
    _record_check(
        checks,
        errors,
        "words_is_list",
        words_is_list,
        f"words count={len(words)}" if isinstance(words, list) else f"Expected list for 'words', got {type(words).__name__}",
    )

    if isinstance(words, list) and len(words) > 0:
        word_items_valid = True
        bad_word_detail = ""
        for idx, w in enumerate(words):
            if not isinstance(w, dict):
                word_items_valid = False
                bad_word_detail = f"words[{idx}] is not an object"
                break
            w_str = w.get("word")
            w_start = w.get("start")
            w_end = w.get("end")
            w_conf = w.get("confidence", 1.0)
            if not isinstance(w_str, str):
                word_items_valid = False
                bad_word_detail = f"words[{idx}].word is not a string"
                break
            if not (isinstance(w_start, (int, float)) and not isinstance(w_start, bool) and w_start >= 0):
                word_items_valid = False
                bad_word_detail = f"words[{idx}].start invalid: {w_start!r}"
                break
            if not (isinstance(w_end, (int, float)) and not isinstance(w_end, bool) and w_end >= w_start):
                word_items_valid = False
                bad_word_detail = f"words[{idx}].end ({w_end!r}) < start ({w_start!r})"
                break
            if not (isinstance(w_conf, (int, float)) and not isinstance(w_conf, bool)):
                word_items_valid = False
                bad_word_detail = f"words[{idx}].confidence invalid: {w_conf!r}"
                break

        _record_check(
            checks,
            errors,
            "word_items_schema",
            word_items_valid,
            f"All {len(words)} word items have valid word/start/end/confidence" if word_items_valid else bad_word_detail,
        )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_transcription_sse_response(
    events: list[dict[str, Any]],
    saw_done: bool,
    content_type: str = "text/event-stream",
    status_code: int = 200,
) -> SchemaCheckResult:
    """Validate POST /v1/audio/transcriptions (stream=true SSE) response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )
    _record_check(
        checks,
        errors,
        "content_type_event_stream",
        "text/event-stream" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )
    _record_check(
        checks,
        errors,
        "sse_events_non_empty",
        len(events) > 0,
        f"Parsed {len(events)} JSON SSE data event(s)",
    )

    done_events = [
        e for e in events
        if isinstance(e, dict)
        and e.get("type") == "transcript.text.done"
        and isinstance(e.get("text"), str)
    ]
    has_done_event = len(done_events) >= 1
    _record_check(
        checks,
        errors,
        "transcript_text_done_event",
        has_done_event,
        f"Found transcript.text.done event with text={done_events[0].get('text')!r}"
        if has_done_event
        else "Missing event with type='transcript.text.done' and string 'text'",
    )

    _record_check(
        checks,
        errors,
        "sse_done_sentinel",
        saw_done,
        "Received terminal 'data: [DONE]' line" if saw_done else "Missing terminal 'data: [DONE]' line",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_diarization_response(
    body: Any,
    content_type: str = "application/json",
    status_code: int = 200,
    expect_text: bool = False,
) -> SchemaCheckResult:
    """Validate POST /v1/audio/diarization house JSON response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    task_val = body.get("task")
    _record_check(
        checks,
        errors,
        "task_is_diarize",
        task_val == "diarize",
        f"task={task_val!r}",
    )

    model_val = body.get("model")
    _record_check(
        checks,
        errors,
        "model_field_present",
        isinstance(model_val, str) and len(model_val.strip()) > 0,
        f"model={model_val!r}",
    )

    num_speakers = body.get("num_speakers")
    num_speakers_ok = isinstance(num_speakers, int) and not isinstance(num_speakers, bool) and num_speakers >= 1
    _record_check(
        checks,
        errors,
        "num_speakers_positive_int",
        num_speakers_ok,
        f"num_speakers={num_speakers!r}",
    )

    segments = body.get("segments")
    segments_ok = isinstance(segments, list) and len(segments) >= 1
    _record_check(
        checks,
        errors,
        "segments_non_empty_list",
        segments_ok,
        f"segments count={len(segments)}" if isinstance(segments, list) else f"Expected list, got {type(segments).__name__}",
    )

    if isinstance(segments, list) and len(segments) > 0:
        seg_schema_ok = True
        seg_err = ""
        for idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                seg_schema_ok = False
                seg_err = f"segments[{idx}] is not an object"
                break
            s_id = seg.get("id")
            s_spk = seg.get("speaker")
            s_lbl = seg.get("label")
            s_start = seg.get("start")
            s_end = seg.get("end")
            s_txt = seg.get("text")
            if not (isinstance(s_id, int) and not isinstance(s_id, bool)):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].id invalid: {s_id!r}"
                break
            if not (isinstance(s_spk, str) and len(s_spk) > 0):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].speaker invalid: {s_spk!r}"
                break
            if not (isinstance(s_lbl, str) and len(s_lbl) > 0):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].label invalid: {s_lbl!r}"
                break
            if not (isinstance(s_start, (int, float)) and not isinstance(s_start, bool) and s_start >= 0):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].start invalid: {s_start!r}"
                break
            if not (isinstance(s_end, (int, float)) and not isinstance(s_end, bool) and s_end > s_start):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].end ({s_end!r}) <= start ({s_start!r})"
                break
            if not isinstance(s_txt, str):
                seg_schema_ok = False
                seg_err = f"segments[{idx}].text is not a string"
                break

        _record_check(
            checks,
            errors,
            "segments_item_schema",
            seg_schema_ok,
            f"All {len(segments)} segments have valid id/speaker/label/start/end/text" if seg_schema_ok else seg_err,
        )

        if expect_text:
            has_aligned_text = any(
                isinstance(seg, dict) and isinstance(seg.get("text"), str) and bool(seg["text"].strip())
                for seg in segments
            )
            _record_check(
                checks,
                errors,
                "segments_words_text_aligned",
                has_aligned_text,
                "Segments populated with aligned word text" if has_aligned_text else "Expected non-empty segment text when 'words' supplied",
            )

    speakers = body.get("speakers")
    speakers_ok = (
        isinstance(speakers, list)
        and len(speakers) >= 1
        and (not num_speakers_ok or len(speakers) == num_speakers)
    )
    _record_check(
        checks,
        errors,
        "speakers_list_matches_num_speakers",
        speakers_ok,
        f"speakers count={len(speakers) if isinstance(speakers, list) else 'N/A'}, num_speakers={num_speakers!r}",
    )

    if isinstance(speakers, list) and len(speakers) > 0:
        spk_schema_ok = True
        spk_err = ""
        for idx, spk in enumerate(speakers):
            if not isinstance(spk, dict):
                spk_schema_ok = False
                spk_err = f"speakers[{idx}] is not an object"
                break
            if not (isinstance(spk.get("id"), str) and spk["id"]):
                spk_schema_ok = False
                spk_err = f"speakers[{idx}].id invalid"
                break
            if not (isinstance(spk.get("label"), str) and spk["label"]):
                spk_schema_ok = False
                spk_err = f"speakers[{idx}].label invalid"
                break
            dur = spk.get("total_speech_duration")
            if not (isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur >= 0):
                spk_schema_ok = False
                spk_err = f"speakers[{idx}].total_speech_duration invalid: {dur!r}"
                break
            cnt = spk.get("segment_count")
            if not (isinstance(cnt, int) and not isinstance(cnt, bool) and cnt >= 1):
                spk_schema_ok = False
                spk_err = f"speakers[{idx}].segment_count invalid: {cnt!r}"
                break

        _record_check(
            checks,
            errors,
            "speakers_item_schema",
            spk_schema_ok,
            f"All {len(speakers)} speaker summaries valid" if spk_schema_ok else spk_err,
        )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def parse_wav_audio_info(audio_bytes: bytes) -> dict[str, Any] | None:
    """Parse WAV header and frame metadata from raw bytes, supporting single or multi-RIFF chunks."""
    if not audio_bytes or not audio_bytes.startswith(b"RIFF"):
        return None
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            nframes = wf.getnframes()
            duration_sec = nframes / float(rate) if rate > 0 else 0.0
            return {
                "sample_rate": rate,
                "channels": channels,
                "sample_width": width,
                "n_frames": nframes,
                "duration_sec": round(duration_sec, 4),
            }
    except Exception:
        return None


def validate_tts_response(
    audio_bytes: bytes,
    content_type: str,
    status_code: int = 200,
    response_format: str = "wav",
    expected_sample_rate: int | None = None,
    is_streaming: bool = False,
    chunk_count: int = 1,
) -> tuple[SchemaCheckResult, float]:
    """Validate POST /v1/audio/speech response and compute synthesized audio duration in seconds."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )

    fmt = response_format.lower().strip()
    expected_ct = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
    ct_lower = (content_type or "").lower()
    ct_ok = expected_ct in ct_lower or "audio/" in ct_lower
    _record_check(
        checks,
        errors,
        "content_type_audio",
        ct_ok,
        f"Content-Type={content_type!r} (expected {expected_ct!r})",
    )

    non_empty = isinstance(audio_bytes, (bytes, bytearray)) and len(audio_bytes) > 0
    _record_check(
        checks,
        errors,
        "non_empty_audio_payload",
        non_empty,
        f"Received {len(audio_bytes)} audio bytes",
    )

    if is_streaming:
        _record_check(
            checks,
            errors,
            "streaming_chunks_received",
            chunk_count >= 1,
            f"Received {chunk_count} chunk(s)",
        )

    sr = expected_sample_rate or 24000
    duration_sec = 0.0

    if fmt == "wav" and not is_streaming:
        wav_info = parse_wav_audio_info(audio_bytes)
        wav_valid = wav_info is not None and wav_info["n_frames"] > 0 and wav_info["sample_rate"] > 0
        _record_check(
            checks,
            errors,
            "valid_riff_wav_header",
            wav_valid,
            f"WAV {wav_info['sample_rate']}Hz, {wav_info['channels']}ch, {wav_info['sample_width']*8}-bit, {wav_info['duration_sec']:.3f}s"
            if wav_info and wav_valid
            else "Invalid or empty RIFF WAV payload",
        )
        if wav_info and wav_valid:
            duration_sec = float(wav_info["duration_sec"])
            if expected_sample_rate is not None:
                sr_match = wav_info["sample_rate"] == expected_sample_rate
                _record_check(
                    checks,
                    errors,
                    "sample_rate_matches_model",
                    sr_match,
                    f"WAV sample_rate={wav_info['sample_rate']}Hz (expected {expected_sample_rate}Hz)",
                )
    elif fmt == "pcm":
        pcm_valid = len(audio_bytes) >= 2 and (len(audio_bytes) % 2 == 0)
        _record_check(
            checks,
            errors,
            "valid_pcm16_frame_alignment",
            pcm_valid,
            f"{len(audio_bytes)} PCM16 bytes ({len(audio_bytes) // 2} samples)",
        )
        duration_sec = round((len(audio_bytes) / 2.0) / float(sr), 4) if sr > 0 else 0.0
    else:
        # For streamed WAV or compressed formats (mp3/opus/aac/flac), check if RIFF WAV fallback or estimate duration
        wav_info = parse_wav_audio_info(audio_bytes)
        if wav_info and wav_info["duration_sec"] > 0:
            duration_sec = float(wav_info["duration_sec"])
        else:
            duration_sec = round(max(0.05, len(audio_bytes) / float(sr * 2)), 4)

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors), duration_sec


def _is_non_neg_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 0


def _is_pos_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and val >= 1


def _is_opt_int(val: Any) -> bool:
    return val is None or (isinstance(val, int) and not isinstance(val, bool))


def validate_health_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/health response contract.

    Per live-stt contract:
    - HTTP 200 when status == 'ok' (even when capacity.used == capacity.total!)
    - HTTP 503 when status in ('draining', 'degraded')
    """
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_valid",
        status_code in (200, 503),
        f"HTTP status {status_code} (expected 200 or 503)",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    status_val = body.get("status")
    status_valid = status_val in ("ok", "draining", "degraded")
    _record_check(
        checks,
        errors,
        "status_enum_valid",
        status_valid,
        f"status={status_val!r}",
    )

    if status_valid:
        expected_http = 200 if status_val == "ok" else 503
        _record_check(
            checks,
            errors,
            "http_status_matches_health_status",
            status_code == expected_http,
            f"status={status_val!r} -> HTTP {status_code} (expected {expected_http})",
        )

    backend_val = body.get("backend")
    _record_check(
        checks,
        errors,
        "backend_field_string",
        isinstance(backend_val, str) and len(backend_val.strip()) > 0,
        f"backend={backend_val!r}",
    )

    backends = body.get("backends")
    backends_ok = (
        isinstance(backends, dict)
        and all(
            k in backends and isinstance(backends.get(k), str) and len(backends[k].strip()) > 0
            for k in ("parakeet", "whisper", "parakeet_tdt")
        )
    )
    _record_check(
        checks,
        errors,
        "backends_map_schema",
        backends_ok,
        f"backends={backends!r}",
    )

    model_val = body.get("model")
    _record_check(
        checks,
        errors,
        "model_field_string",
        isinstance(model_val, str) and len(model_val.strip()) > 0,
        f"model={model_val!r}",
    )

    capacity = body.get("capacity")
    cap_ok = (
        isinstance(capacity, dict)
        and _is_non_neg_int(capacity.get("used"))
        and _is_pos_int(capacity.get("total"))
        and isinstance(capacity.get("admitting"), bool)
    )
    _record_check(
        checks,
        errors,
        "capacity_schema",
        cap_ok,
        f"capacity={capacity!r}",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_stats_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/stats detailed runtime telemetry and admission counter response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(
        checks,
        errors,
        "http_status_200",
        status_code == 200,
        f"HTTP status {status_code}",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    top_counters_ok = (
        _is_non_neg_int(body.get("active_calls"))
        and _is_non_neg_int(body.get("active_workers"))
        and _is_pos_int(body.get("max_concurrent_calls"))
        and _is_pos_int(body.get("max_workers"))
        and isinstance(body.get("draining"), bool)
    )
    _record_check(
        checks,
        errors,
        "top_level_concurrency_fields",
        top_counters_ok,
        f"active_calls={body.get('active_calls')!r}, max_concurrent_calls={body.get('max_concurrent_calls')!r}, "
        f"active_workers={body.get('active_workers')!r}, max_workers={body.get('max_workers')!r}, "
        f"draining={body.get('draining')!r}",
    )

    gpu = body.get("gpu")
    gpu_ok = (
        isinstance(gpu, dict)
        and _is_non_neg_int(gpu.get("reserve_mb"))
        and _is_opt_int(gpu.get("free_vram_mb"))
        and _is_opt_int(gpu.get("total_vram_mb"))
        and _is_opt_int(gpu.get("used_vram_mb"))
        and _is_opt_int(gpu.get("utilization_pct"))
    )
    _record_check(
        checks,
        errors,
        "gpu_telemetry_schema",
        gpu_ok,
        f"gpu={gpu!r}",
    )

    transcribe = body.get("transcribe")
    transcribe_ok = (
        isinstance(transcribe, dict)
        and _is_non_neg_int(transcribe.get("active"))
        and _is_non_neg_int(transcribe.get("completed_total"))
        and _is_non_neg_int(transcribe.get("failed_total"))
        and _is_non_neg_int(transcribe.get("rejected_vram_total"))
        and _is_non_neg_int(transcribe.get("rejected_capacity_total"))
        and isinstance(transcribe.get("active_requests"), list)
    )
    _record_check(
        checks,
        errors,
        "transcribe_section_schema",
        transcribe_ok,
        f"transcribe={transcribe!r}",
    )

    diarization = body.get("diarization")
    diar_ok = (
        isinstance(diarization, dict)
        and _is_non_neg_int(diarization.get("active"))
        and _is_non_neg_int(diarization.get("completed_total"))
        and _is_non_neg_int(diarization.get("failed_total"))
        and _is_non_neg_int(diarization.get("rejected_vram_total"))
        and isinstance(diarization.get("active_requests"), list)
    )
    _record_check(
        checks,
        errors,
        "diarization_section_schema",
        diar_ok,
        f"diarization={diarization!r}",
    )

    tts = body.get("tts")
    tts_ok = (
        isinstance(tts, dict)
        and _is_non_neg_int(tts.get("active"))
        and _is_non_neg_int(tts.get("completed_total"))
        and _is_non_neg_int(tts.get("failed_total"))
        and _is_non_neg_int(tts.get("rejected_vram_total"))
        and _is_non_neg_int(tts.get("rejected_capacity_total"))
        and isinstance(tts.get("active_requests"), list)
        and (tts.get("warm_worker") is None or isinstance(tts.get("warm_worker"), dict))
    )
    _record_check(
        checks,
        errors,
        "tts_section_schema",
        tts_ok,
        f"tts={tts!r}",
    )

    pool = body.get("stt_worker_pool")
    pool_ok = (
        isinstance(pool, dict)
        and isinstance(pool.get("standby_workers"), list)
        and _is_non_neg_int(pool.get("standby_count"))
    )
    _record_check(
        checks,
        errors,
        "stt_worker_pool_schema",
        pool_ok,
        f"stt_worker_pool={pool!r}",
    )

    warmed = body.get("warmed_workers")
    _record_check(
        checks,
        errors,
        "warmed_workers_list",
        isinstance(warmed, list),
        f"warmed_workers count={len(warmed) if isinstance(warmed, list) else 'N/A'}",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_version_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/version response contract."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    for key in ("hash", "timestamp", "parakeet_ref", "whisper_ref"):
        val = body.get(key)
        _record_check(
            checks,
            errors,
            f"{key}_is_string",
            isinstance(val, str) and len(val.strip()) > 0,
            f"{key}={val!r}",
        )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_config_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/config read-only runtime settings response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    _record_check(
        checks,
        errors,
        "max_concurrent_calls_positive_int",
        _is_pos_int(body.get("max_concurrent_calls")),
        f"max_concurrent_calls={body.get('max_concurrent_calls')!r}",
    )
    _record_check(
        checks,
        errors,
        "default_model_string",
        isinstance(body.get("default_model"), str) and bool(body["default_model"].strip()),
        f"default_model={body.get('default_model')!r}",
    )
    backends_present = all(
        isinstance(body.get(k), str) and bool(body[k].strip())
        for k in ("parakeet_backend", "whisper_backend")
    )
    _record_check(
        checks,
        errors,
        "backend_settings_present",
        backends_present,
        f"parakeet_backend={body.get('parakeet_backend')!r}, whisper_backend={body.get('whisper_backend')!r}",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_settings_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/settings read-only settings metadata response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    settings_obj = body.get("settings")
    _record_check(
        checks,
        errors,
        "settings_is_object",
        isinstance(settings_obj, dict) and len(settings_obj) > 0,
        f"settings keys={len(settings_obj) if isinstance(settings_obj, dict) else 'N/A'}",
    )

    hot_reload = body.get("hot_reload_fields")
    _record_check(
        checks,
        errors,
        "hot_reload_fields_list",
        isinstance(hot_reload, list) and len(hot_reload) > 0 and all(isinstance(x, str) for x in hot_reload),
        f"hot_reload_fields count={len(hot_reload) if isinstance(hot_reload, list) else 'N/A'}",
    )

    restart_req = body.get("restart_required_fields")
    _record_check(
        checks,
        errors,
        "restart_required_fields_list",
        isinstance(restart_req, list) and len(restart_req) > 0 and all(isinstance(x, str) for x in restart_req),
        f"restart_required_fields count={len(restart_req) if isinstance(restart_req, list) else 'N/A'}",
    )

    _record_check(
        checks,
        errors,
        "store_available_bool",
        isinstance(body.get("store_available"), bool),
        f"store_available={body.get('store_available')!r}",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


# Field contracts for GET /api/backends, matching live-stt's admin_http.py
# (stable since LS-5). "flavor" is the cpu/cuda build; there is no "device" or "name".
_REGISTERED_BACKEND_FIELDS: dict[str, type | tuple[type, ...]] = {
    "engine": str,
    "flavor": str,
    "repo": str,
    "tag": str,
    "wired": bool,
}
_INSTALLED_BACKEND_FIELDS: dict[str, type | tuple[type, ...]] = {
    "engine": str,
    "flavor": str,
    "version": str,
    "path": str,
    "active": bool,
}


def _backend_entry_problems(
    entries: Any,
    label: str,
    fields: dict[str, type | tuple[type, ...]],
    max_reported: int = 5,
) -> list[str]:
    """Return human-readable problems for a list of backend entries (empty list means valid)."""
    if not isinstance(entries, list):
        return [f"{label} is not a list (got {type(entries).__name__})"]
    problems: list[str] = []
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            problems.append(f"{label}[{idx}] is not an object")
            continue
        missing = [k for k in fields if k not in item]
        wrong = [
            f"{k}={type(item[k]).__name__}"
            for k, typ in fields.items()
            if k in item and not isinstance(item[k], typ)
        ]
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if wrong:
            parts.append(f"wrong type {', '.join(wrong)}")
        if parts:
            problems.append(f"{label}[{idx}] {'; '.join(parts)}")
    if len(problems) > max_reported:
        extra = len(problems) - max_reported
        problems = problems[:max_reported] + [f"... and {extra} more"]
    return problems


def validate_backends_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
) -> SchemaCheckResult:
    """Validate GET /api/backends read-only backend store status response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    _record_check(
        checks,
        errors,
        "store_available_bool",
        isinstance(body.get("store_available"), bool),
        f"store_available={body.get('store_available')!r}",
    )

    store_available = body.get("store_available")
    registered = body.get("registered")
    reg_problems = _backend_entry_problems(registered, "registered", _REGISTERED_BACKEND_FIELDS)
    if isinstance(registered, list) and not registered and store_available is not False:
        reg_problems.insert(0, "registered is empty while store_available is not false")
    _record_check(
        checks,
        errors,
        "registered_backends_list",
        not reg_problems,
        (
            f"registered count={len(registered)}"
            if not reg_problems
            else "; ".join(reg_problems)
        ),
    )

    installed = body.get("installed")
    inst_problems = _backend_entry_problems(installed, "installed", _INSTALLED_BACKEND_FIELDS)
    _record_check(
        checks,
        errors,
        "installed_backends_list",
        not inst_problems,
        (
            f"installed count={len(installed)}"
            if not inst_problems
            else "; ".join(inst_problems)
        ),
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_engine_status_response(
    body: Any,
    status_code: int = 200,
    content_type: str = "application/json",
    expected_engines: tuple[str, ...] | None = None,
) -> SchemaCheckResult:
    """Validate GET /api/diarization-engines or GET /api/tts-engines read-only response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    engines = body.get("engines")
    engines_ok = (
        isinstance(engines, dict)
        and len(engines) > 0
        and all(
            isinstance(info, dict) and isinstance(info.get("installed"), bool)
            for info in engines.values()
        )
    )
    _record_check(
        checks,
        errors,
        "engines_map_schema",
        engines_ok,
        f"engines keys={list(engines.keys()) if isinstance(engines, dict) else 'N/A'}",
    )

    if expected_engines and isinstance(engines, dict):
        has_expected = all(eng in engines for eng in expected_engines)
        _record_check(
            checks,
            errors,
            "expected_engines_present",
            has_expected,
            f"Found engines={list(engines.keys())} (expected {list(expected_engines)})",
        )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_prometheus_metrics_response(
    raw_text: str,
    status_code: int = 200,
    content_type: str = "text/plain",
) -> SchemaCheckResult:
    """Validate GET /metrics Prometheus text exposition response and label safety."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_text_plain",
        "text/plain" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    text_str = raw_text if isinstance(raw_text, str) else ""
    non_empty = len(text_str.strip()) > 0
    _record_check(
        checks,
        errors,
        "non_empty_metrics_body",
        non_empty,
        f"Metrics payload length={len(text_str)} chars",
    )

    has_help_and_type = "# HELP " in text_str and "# TYPE " in text_str
    _record_check(
        checks,
        errors,
        "prometheus_help_and_type_headers",
        has_help_and_type,
        "Found '# HELP' and '# TYPE' lines" if has_help_and_type else "Missing Prometheus '# HELP' or '# TYPE' headers",
    )

    required_metrics = (
        "live_stt_streams_active",
        "live_stt_transcribe_requests_total",
        "live_stt_diarization_requests_total",
        "live_stt_tts_requests_total",
        "live_stt_admission_rejections_total",
    )
    missing_metrics = [m for m in required_metrics if m not in text_str]
    _record_check(
        checks,
        errors,
        "core_live_stt_metrics_present",
        len(missing_metrics) == 0,
        f"All {len(required_metrics)} core live_stt_* metric families present"
        if not missing_metrics
        else f"Missing metric families: {missing_metrics}",
    )

    forbidden_labels = ("stream_id=", "call_id=", "call_ref=")
    found_forbidden = [lbl for lbl in forbidden_labels if lbl in text_str]
    _record_check(
        checks,
        errors,
        "no_high_cardinality_pii_labels",
        len(found_forbidden) == 0,
        "No forbidden per-stream PII labels found"
        if not found_forbidden
        else f"Forbidden high-cardinality labels found: {found_forbidden}",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def validate_dashboard_html_response(
    raw_text: str,
    status_code: int = 200,
    content_type: str = "text/html",
) -> SchemaCheckResult:
    """Validate GET /dashboard read-only HTML admin dashboard response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    _record_check(checks, errors, "http_status_200", status_code == 200, f"HTTP status {status_code}")
    _record_check(
        checks,
        errors,
        "content_type_html",
        "text/html" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    text_lower = (raw_text or "").lower()
    is_html = "<html" in text_lower or "<!doctype html" in text_lower
    _record_check(
        checks,
        errors,
        "valid_html_document",
        is_html,
        f"HTML length={len(raw_text or '')} chars",
    )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)


def extract_error_message(body: Any) -> str | None:
    """Extract error message string from either nested {'error': {'message': ...}} or flat {'error': '...'}."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    elif isinstance(err, str) and err.strip():
        return err.strip()
    return None


def validate_error_response(
    body: Any,
    status_code: int,
    content_type: str = "application/json",
    expected_status: int | tuple[int, ...] = 400,
    shape: str = "any",  # "nested" | "flat" | "any"
    expected_substring: str | None = None,
) -> SchemaCheckResult:
    """Validate an error HTTP response status code, JSON content-type, error body shape, and message substring."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    allowed_statuses = (expected_status,) if isinstance(expected_status, int) else tuple(expected_status)
    _record_check(
        checks,
        errors,
        "http_status_matches_expected",
        status_code in allowed_statuses,
        f"HTTP status {status_code} (expected {allowed_statuses})",
    )
    _record_check(
        checks,
        errors,
        "content_type_json",
        "application/json" in (content_type or "").lower(),
        f"Content-Type={content_type!r}",
    )

    is_dict = isinstance(body, dict)
    _record_check(
        checks,
        errors,
        "body_is_object",
        is_dict,
        "Response body is a JSON object" if is_dict else f"Expected dict, got {type(body).__name__}",
    )
    if not is_dict:
        return SchemaCheckResult(valid=False, checks=checks, errors=errors)

    has_error_key = "error" in body
    _record_check(
        checks,
        errors,
        "has_error_field",
        has_error_key,
        "Found top-level 'error' key" if has_error_key else f"Missing 'error' key in {list(body.keys())}",
    )

    err_val = body.get("error")
    norm_shape = (shape or "any").strip().lower()
    if norm_shape == "nested":
        shape_ok = (
            isinstance(err_val, dict)
            and isinstance(err_val.get("message"), str)
            and bool(err_val["message"].strip())
        )
        shape_detail = (
            f"error.message={err_val.get('message')!r}"
            if isinstance(err_val, dict)
            else f"Expected nested dict {'{message: str}'}, got {type(err_val).__name__}"
        )
    elif norm_shape == "flat":
        shape_ok = isinstance(err_val, str) and bool(err_val.strip())
        shape_detail = (
            f"error={err_val!r}"
            if isinstance(err_val, str)
            else f"Expected flat string error, got {type(err_val).__name__}"
        )
    else:
        msg = extract_error_message(body)
        shape_ok = msg is not None
        shape_detail = f"error message={msg!r}" if msg is not None else f"Invalid error field shape: {err_val!r}"

    _record_check(
        checks,
        errors,
        "error_body_shape",
        shape_ok,
        shape_detail,
    )

    extracted_msg = extract_error_message(body) or ""
    if expected_substring:
        sub_ok = expected_substring.lower() in extracted_msg.lower()
        _record_check(
            checks,
            errors,
            "error_message_contains_expected",
            sub_ok,
            f"Message {extracted_msg!r} contains {expected_substring!r}"
            if sub_ok
            else f"Expected {expected_substring!r} in error message {extracted_msg!r}",
        )

    return SchemaCheckResult(valid=len(errors) == 0, checks=checks, errors=errors)

