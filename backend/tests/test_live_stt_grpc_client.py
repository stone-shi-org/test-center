"""Comprehensive unit and integration tests for the live-stt gRPC client & streaming harness (TES-30).

Covers:
1. Vendored proto/livestt/v1/asr.proto schema verification and zero dependency on live-stt code
2. Build-time stub generation (scripts/gen_proto.sh and pb.generate_proto_stubs / ensure_proto_stubs)
3. gRPC target resolution (resolve_grpc_target) across URLs, host:port, target dicts, and config overrides
4. WAV / raw PCM16 extraction, format validation, and frame chunking (20ms, 100ms, 160ms)
5. Real-time (speed=1.0), faster-than-real-time (speed>1.0), and unpaced burst (speed=0 / realtime=False) pacing
6. GetServerInfo RPC (async and sync) and error handling
7. Bidirectional Transcribe streaming over a real loopback grpc.aio server:
   - Ready, incremental partial (TranscriptDelta), EOU, EOB, Warning, Recycled, and Final events
   - Monotonic and wall-clock timestamps on every partial/final event
   - Word-level timestamps and audio offset tracking
   - Latency metrics (TTFP, ready latency, finalization latency, total latency, RTF)
8. gRPC error contract (INVALID_ARGUMENT, RESOURCE_EXHAUSTED, UNAVAILABLE) and LiveSTTGRPCError
9. LiveSTTStreamingHarness concurrent session execution, auth metadata forwarding, and CLI runner
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import pytest

from backend.app.testers.base import TesterCancelledError
from backend.app.testers.live_stt import (
    DEFAULT_CHUNK_MS,
    DEFAULT_GRPC_PORT,
    DEFAULT_SAMPLE_RATE_HZ,
    LiveSTTGRPCClient,
    LiveSTTGRPCError,
    LiveSTTStreamingHarness,
    MockLiveSTTGRPCServer,
    MockStreamingASRServicer,
    asr_pb2,
    asr_pb2_grpc,
    ensure_proto_stubs,
    extract_pcm16_audio,
    find_proto_root,
    generate_proto_stubs,
    get_audio_fixture,
     get_bad_audio_fixture,
    is_wav_bytes,
    iter_paced_pcm_chunks,
    load_asr_stubs,
    make_pcm_wav_bytes,
    resolve_grpc_target,
    slice_pcm16_chunks,
)
from backend.app.testers.live_stt.grpc_client import _cli_main


REPO_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# 1. Vendored .proto, Build-Time Stub Generation, and Independence Checks
# ===========================================================================


def test_vendored_proto_exists_and_has_expected_definitions() -> None:
    proto_root = find_proto_root()
    proto_file = proto_root / "livestt" / "v1" / "asr.proto"
    assert proto_file.is_file(), f"Expected vendored proto at {proto_file}"

    content = proto_file.read_text(encoding="utf-8")
    assert 'syntax = "proto3";' in content
    assert "package livestt.v1;" in content
    assert "service StreamingASR" in content
    assert "rpc Transcribe(stream TranscriptionRequest) returns (stream TranscriptionEvent);" in content
    assert "rpc GetServerInfo(ServerInfoRequest) returns (ServerInfoResponse);" in content
    assert "message StreamConfig" in content
    assert "message TranscriptDelta" in content
    assert "message Final" in content


def test_gen_proto_script_and_dynamic_stub_generation(tmp_path: Path) -> None:
    # 1. Test programmatic generation into an isolated temp directory
    v1_dir = generate_proto_stubs(output_dir=tmp_path)
    assert (v1_dir / "asr_pb2.py").is_file()
    assert (v1_dir / "asr_pb2_grpc.py").is_file()
    assert (v1_dir / "asr_pb2.pyi").is_file()
    assert (v1_dir / "__init__.py").is_file()
    assert (tmp_path / "livestt" / "__init__.py").is_file()

    # 2. Test scripts/gen_proto.sh execution
    gen_script = REPO_ROOT / "scripts" / "gen_proto.sh"
    assert gen_script.is_file()
    proc = subprocess.run(
        ["bash", str(gen_script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Generated Python gRPC stubs" in proc.stdout

    # 3. Verify loaded stubs expose all expected protobuf & gRPC classes
    pb2, pb2_grpc = load_asr_stubs()
    assert pb2 is asr_pb2
    assert pb2_grpc is asr_pb2_grpc
    assert hasattr(asr_pb2, "StreamConfig")
    assert hasattr(asr_pb2, "TranscriptionRequest")
    assert hasattr(asr_pb2, "TranscriptionEvent")
    assert hasattr(asr_pb2, "TranscriptDelta")
    assert hasattr(asr_pb2, "Ready")
    assert hasattr(asr_pb2, "Final")
    assert hasattr(asr_pb2, "Warning")
    assert hasattr(asr_pb2, "Recycled")
    assert hasattr(asr_pb2_grpc, "StreamingASRStub")
    assert hasattr(asr_pb2_grpc, "StreamingASRServicer")


def test_no_dependency_on_live_stt_codebase() -> None:
    """Verify no Python file in backend/ imports from the external live_stt package."""
    backend_dir = REPO_ROOT / "backend"
    offending_imports: list[str] = []

    for py_file in backend_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "live_stt" or alias.name.startswith("live_stt."):
                        offending_imports.append(f"{py_file}:{node.lineno} -> import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "live_stt" or node.module.startswith("live_stt."):
                    offending_imports.append(f"{py_file}:{node.lineno} -> from {node.module} import ...")

    assert offending_imports == [], f"Found forbidden imports from live_stt: {offending_imports}"


def test_plan_md_open_item_resolved() -> None:
    plan_text = (REPO_ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8")
    assert "proto/livestt/v1/asr.proto" in plan_text
    assert "Resolved" in plan_text


# ===========================================================================
# 2. Target Address Resolution Tests
# ===========================================================================


def test_resolve_grpc_target_variants() -> None:
    assert resolve_grpc_target("127.0.0.1:50051") == "127.0.0.1:50051"
    assert resolve_grpc_target("localhost") == f"localhost:{DEFAULT_GRPC_PORT}"
    assert resolve_grpc_target("grpc://10.100.0.50:50161") == "10.100.0.50:50161"
    assert resolve_grpc_target("http://10.100.0.50:8000", {"grpc_port": 50051}) == "10.100.0.50:50051"
    assert (
        resolve_grpc_target(
            {"base_url": "http://10.100.0.50:8000", "extra": {"grpc_port": 50161}}
        )
        == "10.100.0.50:50161"
    )
    assert (
        resolve_grpc_target(
            {"base_url": "http://10.100.0.50:8000", "extra": {"grpc_target": "10.100.0.50:50055"}}
        )
        == "10.100.0.50:50055"
    )
    assert (
        resolve_grpc_target(
            {"base_url": "http://10.100.0.50:8000"},
            {"grpc_host": "192.168.1.10", "grpc_port": 60000},
        )
        == "192.168.1.10:60000"
    )
    assert resolve_grpc_target("[::1]:50051") == "[::1]:50051"

    with pytest.raises(ValueError, match="missing"):
        resolve_grpc_target({})


# ===========================================================================
# 3. Audio Extraction & Chunking Tests
# ===========================================================================


def test_extract_pcm16_audio_and_chunking(tmp_path: Path) -> None:
    short_fix = get_audio_fixture("short_speech")
    assert is_wav_bytes(short_fix.wav_bytes)

    # 1. From AudioFixture
    ext_fix = extract_pcm16_audio(short_fix)
    assert ext_fix.source_format == "fixture"
    assert ext_fix.fixture_id == "short_speech"
    assert ext_fix.sample_rate_hz == 16000
    assert ext_fix.channels == 1
    assert ext_fix.sample_width == 2
    assert abs(ext_fix.duration_sec - 1.5) < 0.01
    assert len(ext_fix.pcm_bytes) == int(1.5 * 16000 * 2)

    # 2. From WAV bytes
    ext_wav = extract_pcm16_audio(short_fix.wav_bytes)
    assert ext_wav.source_format == "wav"
    assert ext_wav.pcm_bytes == ext_fix.pcm_bytes

    # 3. From file path on disk
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(short_fix.wav_bytes)
    ext_path = extract_pcm16_audio(wav_path)
    assert ext_path.source_format == "wav"
    assert ext_path.pcm_bytes == ext_fix.pcm_bytes

    # 4. From raw PCM bytes (no RIFF header)
    raw_pcm = b"\x00\x00" * 16000  # 1.0s at 16kHz int16 mono
    assert not is_wav_bytes(raw_pcm)
    ext_pcm = extract_pcm16_audio(raw_pcm)
    assert ext_pcm.source_format == "pcm"
    assert abs(ext_pcm.duration_sec - 1.0) < 1e-6

    # 5. Slice into 20ms chunks (640 bytes per chunk)
    chunks_20ms = slice_pcm16_chunks(ext_fix.pcm_bytes, chunk_ms=20.0)
    assert len(chunks_20ms) == 75  # 1500ms / 20ms
    assert all(len(c) == 640 for c in chunks_20ms)
    assert b"".join(chunks_20ms) == ext_fix.pcm_bytes

    # 6. Slice into 160ms chunks (5120 bytes per chunk)
    chunks_160ms = slice_pcm16_chunks(ext_fix.pcm_bytes, chunk_ms=160.0)
    assert b"".join(chunks_160ms) == ext_fix.pcm_bytes


def test_extract_pcm16_audio_rejects_bad_wav_formats() -> None:
    with pytest.raises(ValueError, match="16000Hz"):
        extract_pcm16_audio(get_bad_audio_fixture("wrong_sample_rate_8khz").wav_bytes)

    with pytest.raises(ValueError, match="mono"):
        extract_pcm16_audio(get_bad_audio_fixture("stereo_2ch_wav").wav_bytes)

    with pytest.raises(ValueError, match="16-bit"):
        extract_pcm16_audio(get_bad_audio_fixture("wrong_bit_depth_8bit").wav_bytes)

    corrupt_riff = b"RIFF\x24\x00\x00\x00WAVENOT_VALID_FMT"
    with pytest.raises(ValueError, match="Failed to parse WAV"):
        extract_pcm16_audio(corrupt_riff)


# ===========================================================================
# 4. Pacing Tests (Real-Time, Faster-Than-Real-Time, Unpaced Burst)
# ===========================================================================


@pytest.mark.asyncio
async def test_iter_paced_pcm_chunks_deterministic_schedule() -> None:
    """Verify exact drift-compensated sleep schedule for 1x, 2x, 4x, and unpaced speeds."""
    # 5 chunks of 100ms each (3200 bytes at 16kHz int16 mono) = 0.5s total audio
    chunks = [b"\x00\x00" * 1600 for _ in range(5)]

    for speed_mult, expected_step in [(1.0, 0.10), (2.0, 0.05), (4.0, 0.025)]:
        virtual_now = 100.0
        sleeps: list[float] = []

        def fake_clock() -> float:
            return virtual_now

        async def fake_sleep(delay: float) -> None:
            nonlocal virtual_now
            sleeps.append(round(delay, 6))
            virtual_now += delay

        yielded = []
        async for idx, data, dur, cum in iter_paced_pcm_chunks(
            chunks,
            speed=speed_mult,
            realtime=True,
            sleep_fn=fake_sleep,
            clock_fn=fake_clock,
        ):
            yielded.append((idx, len(data), round(dur, 3), round(cum, 3)))

        assert len(yielded) == 5
        assert len(sleeps) == 4  # chunk 0 is immediate; chunks 1..4 sleep
        assert all(abs(s - expected_step) < 1e-5 for s in sleeps), (
            f"speed={speed_mult}: expected sleeps of {expected_step}s, got {sleeps}"
        )

    # Unpaced mode (speed=0 or realtime=False): zero sleeps
    sleeps_unpaced: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps_unpaced.append(delay)

    items = [
        item
        async for item in iter_paced_pcm_chunks(
            chunks, speed=0.0, realtime=False, sleep_fn=record_sleep
        )
    ]
    assert len(items) == 5
    assert sleeps_unpaced == []


@pytest.mark.asyncio
async def test_streaming_client_wall_clock_pacing_realtime_vs_faster() -> None:
    """Stream 0.40s of audio at 1x real-time, 4x faster-than-real-time, and unpaced over loopback gRPC."""
    wav_400ms = make_pcm_wav_bytes(sample_rate=16000, channels=1, sample_width=2, duration_sec=0.40)

    async with MockLiveSTTGRPCServer() as server:
        harness = LiveSTTStreamingHarness(server.target, default_chunk_ms=40.0)

        # 1. Real-time (speed=1.0): 10 chunks of 40ms -> 9 intervals of 40ms ~= 360ms send duration
        res_1x = await harness.run_session(wav_400ms, speed=1.0, realtime=True)
        assert res_1x.ok
        assert res_1x.chunks_sent == 10
        assert res_1x.send_duration_ms >= 300.0

        # 2. Faster than real-time (speed=4.0): 9 intervals of 10ms ~= 90ms send duration
        res_4x = await harness.run_session(wav_400ms, speed=4.0, realtime=True)
        assert res_4x.ok
        assert res_4x.chunks_sent == 10
        assert 50.0 <= res_4x.send_duration_ms < res_1x.send_duration_ms * 0.65

        # 3. Unpaced burst (speed=0 / realtime=False): < 50ms send duration
        res_burst = await harness.run_session(wav_400ms, speed=0.0, realtime=False)
        assert res_burst.ok
        assert res_burst.chunks_sent == 10
        assert res_burst.send_duration_ms < 50.0


# ===========================================================================
# 5. GetServerInfo RPC Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_get_server_info_async() -> None:
    servicer = MockStreamingASRServicer(
        default_model="realtime_eou_120m-v1",
        backend="cuda",
        version="abc1234",
        parakeet_abi_version=6,
        max_concurrent_calls=4,
    )
    async with MockLiveSTTGRPCServer(servicer) as server:
        async with LiveSTTGRPCClient(server.target) as client:
            info = await client.get_server_info()
            assert info.ok is True
            assert info.grpc_status_code == "OK"
            assert info.default_model == "realtime_eou_120m-v1"
            assert info.backend == "cuda"
            assert info.version == "abc1234"
            assert info.parakeet_abi_version == 6
            assert info.max_concurrent_calls == 4
            assert info.active_calls == 0
            assert info.latency_ms > 0


def test_sync_wrappers_server_info_and_session() -> None:
    servicer = MockStreamingASRServicer(
        default_model="realtime_eou_120m-v1",
        backend="cuda",
        version="abc1234",
        parakeet_abi_version=6,
        max_concurrent_calls=4,
    )
    with MockLiveSTTGRPCServer(servicer) as server:
        harness = LiveSTTStreamingHarness(server.target, default_realtime=False)
        sync_info = harness.get_server_info_sync()
        assert sync_info.ok is True
        assert sync_info.default_model == "realtime_eou_120m-v1"
        assert sync_info.to_dict()["backend"] == "cuda"

        sync_session = harness.run_session_sync("short_speech")
        assert sync_session.ok is True
        assert sync_session.text == "hello world testing"
        assert len(sync_session.partials) >= 2
        assert sync_session.final_event is not None


# ===========================================================================
# 6. Bidirectional Transcribe Streaming & Timestamped Event Recording
# ===========================================================================


@pytest.mark.asyncio
async def test_transcribe_records_partial_and_final_events_with_timestamps() -> None:
    servicer = MockStreamingASRServicer(
        emit_eou_every_n_deltas=2,
        emit_eob_on_first_delta=True,
        emit_warning={
            "code": asr_pb2.WARNING_CODE_FALLING_BEHIND,
            "message": "decode slightly behind",
            "at_sec": 0.16,
            "behind_sec": 0.25,
        },
        emit_recycled_at_sec=0.80,
        final_delay_sec=0.01,
    )
    async with MockLiveSTTGRPCServer(servicer) as server:
        received_callback_events: list[str] = []
        harness = LiveSTTStreamingHarness(
            server.target,
            api_key="test-bearer-token",
            default_chunk_ms=20.0,
            default_speed=10.0,  # 10x real-time for fast test execution while exercising pacing
        )
        res = await harness.run_session(
            "short_speech",
            model="realtime_eou_120m-v1",
            language="en-US",
            call_id="call-tes30-test",
            enable_word_timestamps=True,
            wait_for_ready=True,
            on_event=lambda e: received_callback_events.append(e.kind),
        )

    assert res.ok is True
    assert res.grpc_status_code == "OK"
    assert res.call_id == "call-tes30-test"
    assert res.model == "realtime_eou_120m-v1"
    assert abs(res.audio_duration_sec - 1.5) < 0.01
    assert res.chunks_sent == 75
    assert res.bytes_sent == 48000

    # Verify auth metadata reached server
    assert servicer.received_Metadata
    assert servicer.received_Metadata[0].get("authorization") == "Bearer test-bearer-token"

    # Verify event sequence: Ready first, Final last, Partials (delta) in between
    assert res.events[0].is_ready
    assert res.events[-1].is_final
    assert len(res.partials) >= 2
    assert received_callback_events == [e.kind for e in res.events]

    # Verify partial and final transcripts & word timestamps
    assert res.partial_texts == ["hello", " world"]
    assert res.final_text == " testing"
    assert res.text == "hello world testing"
    assert [w.text for w in res.words] == ["hello", "world", "testing"]
    assert all(w.end_sec > w.start_sec for w in res.words)

    # Verify in-band control events were recorded with timestamps
    assert len(res.warnings) == 1
    assert res.warnings[0].detail["code"] == "WARNING_CODE_FALLING_BEHIND"
    assert len(res.recycled_events) == 1
    assert res.recycled_events[0].detail["reason"] == "RECYCLE_REASON_RSS_THRESHOLD"
    assert res.worker_generations == 2
    assert len(res.eob_events) == 1
    assert len(res.eou_events) == 1

    # Verify monotonic and wall-clock timestamps on every event
    prev_elapsed = 0.0
    for idx, evt in enumerate(res.events):
        assert evt.seq == idx
        assert evt.elapsed_ms >= prev_elapsed
        prev_elapsed = evt.elapsed_ms
        assert evt.timestamp_unix >= res.started_at_unix
        assert evt.timestamp_iso.endswith("Z")
        if evt.is_partial:
            assert evt.event_type == "partial"
            assert evt.since_first_audio_ms is not None
            assert evt.audio_offset_sec is not None and evt.audio_offset_sec > 0
        if evt.is_final:
            assert evt.event_type == "final"
            assert evt.since_half_close_ms is not None and evt.since_half_close_ms >= 5.0
            assert abs((evt.audio_offset_sec or 0.0) - 1.5) < 0.01

    # Verify session-level timing & RTF metrics
    assert res.ready_latency_ms is not None and res.ready_latency_ms >= 0.0
    assert res.first_partial_ms is not None and res.first_partial_ms > res.ready_latency_ms
    assert res.ttfp_ms == res.first_partial_ms
    assert res.ttfp_from_first_audio_ms is not None and res.ttfp_from_first_audio_ms > 0.0
    assert res.final_latency_ms is not None and res.final_latency_ms >= 5.0
    assert res.total_latency_ms >= res.final_latency_ms
    assert res.rtf > 0.0
    assert len(res.inter_partial_latencies_ms) == len(res.partials) - 1

    # Verify clean JSON serialization via to_dict()
    serialized = res.to_dict(include_chunk_sends=True)
    assert serialized["ok"] is True
    assert serialized["partial_count"] == 2
    assert serialized["word_count"] == 3
    assert len(serialized["chunk_sends"]) == 75
    json_str = json.dumps(serialized)
    assert "hello world testing" in json_str


# ===========================================================================
# 7. Error Handling & Protocol Contract Tests over gRPC
# ===========================================================================


@pytest.mark.asyncio
async def test_transcribe_error_paths_and_status_codes() -> None:
    async with MockLiveSTTGRPCServer() as server:
        harness = LiveSTTStreamingHarness(server.target, default_realtime=False)

        # 1. Audio before StreamConfig -> INVALID_ARGUMENT
        res_no_cfg = await harness.run_session("short_speech", send_config=False)
        assert res_no_cfg.ok is False
        assert res_no_cfg.grpc_status_code == "INVALID_ARGUMENT"
        assert "StreamConfig" in (res_no_cfg.error_message or "")

        # 2. Duplicate StreamConfig -> INVALID_ARGUMENT
        res_dup_cfg = await harness.run_session("short_speech", send_duplicate_config=True)
        assert res_dup_cfg.ok is False
        assert res_dup_cfg.grpc_status_code == "INVALID_ARGUMENT"
        assert "twice" in (res_dup_cfg.error_message or "")

        # 3. Unsupported sample rate -> INVALID_ARGUMENT
        res_bad_sr = await harness.run_session(
            b"\x00\x00" * 1600,
            sample_rate_hz=8000,
            validate_wav=False,
        )
        assert res_bad_sr.ok is False
        assert res_bad_sr.grpc_status_code == "INVALID_ARGUMENT"
        assert "16000" in (res_bad_sr.error_message or "")

        # 4. Unsupported encoding (MULAW) -> INVALID_ARGUMENT
        res_bad_enc = await harness.run_session(
            "short_speech",
            encoding=asr_pb2.AUDIO_ENCODING_MULAW,
        )
        assert res_bad_enc.ok is False
        assert res_bad_enc.grpc_status_code == "INVALID_ARGUMENT"
        assert "AUDIO_ENCODING_MULAW" in (res_bad_enc.error_message or "")

        # 5. Unknown model -> INVALID_ARGUMENT
        res_unk_model = await harness.run_session("short_speech", model="does-not-exist")
        assert res_unk_model.ok is False
        assert res_unk_model.grpc_status_code == "INVALID_ARGUMENT"

        # 6. Batch-only model (whisper) -> INVALID_ARGUMENT
        res_whisper = await harness.run_session("short_speech", model="whisper-large-v3-turbo")
        assert res_whisper.ok is False
        assert res_whisper.grpc_status_code == "INVALID_ARGUMENT"
        assert "batch-only" in (res_whisper.error_message or "")

        # 7. raise_on_grpc_error=True raises LiveSTTGRPCError
        with pytest.raises(LiveSTTGRPCError) as exc_info:
            await harness.run_session(
                "short_speech",
                model="whisper-large-v3-turbo",
                raise_on_grpc_error=True,
            )
        assert exc_info.value.status_code == "INVALID_ARGUMENT"
        assert exc_info.value.partial_result is not None

        # 8. Server draining -> UNAVAILABLE
        server.servicer.draining = True
        res_drain = await harness.run_session("short_speech")
        assert res_drain.ok is False
        assert res_drain.grpc_status_code == "UNAVAILABLE"


@pytest.mark.asyncio
async def test_cancellation_aborts_streaming_session() -> None:
    async with MockLiveSTTGRPCServer() as server:
        harness = LiveSTTStreamingHarness(server.target, default_chunk_ms=20.0, default_speed=2.0)
        chunks_seen = 0

        def cancel_after_3() -> None:
            nonlocal chunks_seen
            chunks_seen += 1
            if chunks_seen > 3:
                raise TesterCancelledError("Run cancelled by test")

        with pytest.raises(TesterCancelledError):
            await harness.run_session("medium_speech", cancel_check=cancel_after_3)


# ===========================================================================
# 8. Concurrent Sessions & CLI Entrypoint Tests
# ===========================================================================


@pytest.mark.asyncio
async def test_concurrent_sessions_and_admission_limit() -> None:
    servicer = MockStreamingASRServicer(
        max_concurrent_calls=2,
        decode_delay_sec=0.015,
    )
    async with MockLiveSTTGRPCServer(servicer) as server:
        harness = LiveSTTStreamingHarness(
            server.target,
            default_chunk_ms=160.0,
            default_speed=10.0,
        )
        burst = await harness.run_concurrent_sessions(4, "short_speech")

    assert burst.total_sessions == 4
    assert burst.ok_count == 2
    assert burst.resource_exhausted_count == 2
    assert servicer.peak_active_calls == 2
    assert servicer.active_calls == 0


@pytest.mark.asyncio
async def test_cli_entrypoint_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    async with MockLiveSTTGRPCServer() as server:
        exit_code = await _cli_main(
            [
                "--target",
                server.target,
                "--audio",
                "short_speech",
                "--speed",
                "0",
                "--json",
            ]
        )
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["text"] == "hello world testing"
    assert payload["partial_count"] >= 2
