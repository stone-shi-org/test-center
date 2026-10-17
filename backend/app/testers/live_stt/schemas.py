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


def validate_models_response(body: Any) -> SchemaCheckResult:
    """Validate GET /v1/models OpenAI-style model catalog response."""
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

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
        all_have_id = all(isinstance(item, dict) and isinstance(item.get("id"), str) and bool(item["id"].strip()) for item in data)
        _record_check(
            checks,
            errors,
            "items_have_valid_id",
            all_have_id,
            "All model entries have non-empty string 'id'" if all_have_id else "One or more model entries missing valid 'id'",
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
