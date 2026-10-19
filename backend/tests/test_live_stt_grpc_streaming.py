"""Unit and integration tests for TES-31: live-stt gRPC streaming & long-call stability suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backend.app.testers.base import TestContext
from backend.app.testers.live_stt import (
    DEFAULT_GRPC_STREAMING_MODEL,
    LiveSTTTester,
    LiveSTTStreamingHarness,
    MockLiveSTTGRPCServer,
    build_long_call_audio_fixture,
    compute_grpc_streaming_summary,
    evaluate_long_call_stability,
    evaluate_streaming_session_contract,
    get_audio_fixture,
    infer_model_info,
    is_batch_only_grpc_model,
    normalize_live_stt_suites,
    run_clean_shutdown_and_channel_reuse_check,
    run_protocol_and_drain_contract_check,
    run_reconnect_after_abort_check,
    select_grpc_streaming_models,
)


def _make_ctx(tmp_path: Path, run_id: int = 3101) -> TestContext:
    artifacts_dir = tmp_path / f"run_{run_id}"
    return TestContext(
        run_id=run_id,
        artifacts_dir=artifacts_dir,
        emit_stdout=False,
    )


# ===========================================================================
# 1. Unit Tests: Suite Normalization, Model Selection, Long-Call Synthesis
# ===========================================================================


def test_grpc_suite_aliases_and_opt_in_behavior():
    """Verify suite normalization expands gRPC streaming and long-call aliases while keeping long_call opt-in."""
    tester = LiveSTTTester()
    assert "grpc_streaming" in tester.suites
    assert "grpc_long_call" in tester.suites
    # Default 7 suites are the fast HTTP suites; grpc_long_call is strictly opt-in
    assert tester.suites[:7] == [
        "transcriptions_json",
        "transcriptions_sse",
        "diarization",
        "tts",
        "admin",
        "error_handling",
        "concurrency",
    ]

    assert normalize_live_stt_suites(["grpc"], tester.suites) == ["grpc_streaming"]
    assert normalize_live_stt_suites(["grpc_transcribe"], tester.suites) == ["grpc_streaming"]
    assert normalize_live_stt_suites(["long_call"], tester.suites) == ["grpc_long_call"]
    assert normalize_live_stt_suites(["long_calls"], tester.suites) == ["grpc_long_call"]
    assert normalize_live_stt_suites(["grpc_all"], tester.suites) == ["grpc_streaming", "grpc_long_call"]


def test_select_grpc_streaming_models_filters_batch_only_models():
    """Verify batch-only models (e.g. whisper-large-v3-turbo, parakeet-tdt) are excluded from gRPC streaming."""
    assert is_batch_only_grpc_model("whisper-large-v3-turbo") is True
    assert is_batch_only_grpc_model("parakeet-tdt-0.6b-v3") is True
    assert is_batch_only_grpc_model("realtime_eou_120m-v1") is False
    assert is_batch_only_grpc_model("parakeet-ctc-0.6b") is False
    assert is_batch_only_grpc_model("nemotron-3.5-asr-streaming-0.6b-q8_0") is False

    models = [
        infer_model_info({"id": "realtime_eou_120m-v1", "type": "asr", "supports_streaming": True}),
        infer_model_info({"id": "whisper-large-v3-turbo", "type": "asr", "supports_streaming": False}),
        infer_model_info({"id": "parakeet-tdt-0.6b-v2", "type": "asr", "supports_streaming": True}),
    ]
    selected = select_grpc_streaming_models(asr_models=models, server_info=None, config={})
    assert [m.id for m in selected] == ["realtime_eou_120m-v1"]

    # Explicit override via grpc_models
    explicit = select_grpc_streaming_models(
        asr_models=models,
        server_info=None,
        config={"grpc_models": ["parakeet-ctc-0.6b", "realtime_eou_120m-v1"]},
    )
    assert [m.id for m in explicit] == ["parakeet-ctc-0.6b", "realtime_eou_120m-v1"]


def test_build_long_call_audio_fixture_rebases_word_timestamps():
    """Verify build_long_call_audio_fixture synthesizes multi-cycle PCM WAV with monotonic word timestamps."""
    fix = build_long_call_audio_fixture(duration_sec=10.0)
    assert fix.duration_sec == 10.0
    assert fix.sample_rate == 16000
    assert fix.channels == 1
    assert len(fix.wav_bytes) > 44 + 16000 * 2 * 9
    assert len(fix.words) >= 12

    prev_start = -1.0
    for w in fix.words:
        assert w.start_sec >= prev_start
        assert w.end_sec >= w.start_sec
        assert w.end_sec <= 10.0 + 1e-6
        prev_start = w.start_sec


# ===========================================================================
# 2. Direct Streaming Metrics, Reconnect/Shutdown, Protocol & Long-Call Checks
# ===========================================================================


def test_grpc_streaming_metrics_ttfp_final_latency_rtf_and_summary():
    """Verify TTFP, finalization latency, RTF, and compute_grpc_streaming_summary over a gRPC stream."""
    with MockLiveSTTGRPCServer(max_concurrent_calls=4, final_delay_sec=0.01) as server:
        harness = LiveSTTStreamingHarness(server.target)
        fix = get_audio_fixture("medium_speech")

        s1 = harness.run_session_sync(
            fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            language="en",
            chunk_ms=100.0,
            realtime=True,
            speed=10.0,
        )
        s2 = harness.run_session_sync(
            fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            language="en",
            chunk_ms=100.0,
            realtime=True,
            speed=25.0,
        )

        passed1, _, errors1, _ = evaluate_streaming_session_contract(
            s1, expected_duration_sec=fix.duration_sec
        )
        passed2, _, errors2, _ = evaluate_streaming_session_contract(
            s2, expected_duration_sec=fix.duration_sec
        )
        assert passed1, errors1
        assert passed2, errors2

        assert s1.ttfp_ms is not None and s1.ttfp_ms > 0.0
        assert s1.ttfp_from_first_audio_ms is not None and s1.ttfp_from_first_audio_ms >= 0.0
        assert s1.final_latency_ms is not None and s1.final_latency_ms >= 5.0
        assert s1.rtf > 0.0

        summary = compute_grpc_streaming_summary([s1, s2])
        assert summary["ttfp_mean_ms"] > 0.0
        assert summary["ttfp_p50_ms"] > 0.0
        assert summary["ttfp_p95_ms"] >= summary["ttfp_p50_ms"]
        assert summary["final_latency_mean_ms"] >= 5.0
        assert summary["rtf_mean"] > 0.0
        assert summary["sessions_ok"] == 2.0
        assert summary["total_partials"] == float(s1.partial_count + s2.partial_count)
        assert summary["total_words"] == float(s1.word_count + s2.word_count)


def test_grpc_reconnect_after_abort_and_clean_shutdown_channel_reuse():
    """Verify mid-stream cancellation releases server slots and clean half-close reuses channel."""
    with MockLiveSTTGRPCServer(max_concurrent_calls=2, final_delay_sec=0.005) as server:
        harness = LiveSTTStreamingHarness(server.target)
        fix = get_audio_fixture("medium_speech")

        # 1. Mid-stream abort + immediate reconnect
        reconn = run_reconnect_after_abort_check(
            harness,
            audio=fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            language="en",
            chunk_ms=100.0,
            abort_after_chunks=4,
            speed=8.0,
            realtime=True,
        )
        assert reconn.passed, reconn.errors
        assert reconn.aborted_chunks_sent == 4
        assert reconn.active_calls_after_abort == 0
        assert reconn.reconnect_session.ok is True
        assert reconn.reconnect_session.final_event is not None
        assert server.servicer.aborted_calls >= 1

        # 2. Clean half-close shutdown + sequential channel reuse
        shutdown_res = run_clean_shutdown_and_channel_reuse_check(
            harness,
            audio=fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            language="en",
            chunk_ms=100.0,
            speed=15.0,
            realtime=True,
        )
        assert shutdown_res["passed"], shutdown_res["errors"]
        assert len(shutdown_res["sessions"]) == 2
        assert shutdown_res["active_calls_after"] == 0


def test_grpc_protocol_errors_and_server_drain_behavior():
    """Verify INVALID_ARGUMENT protocol violations and UNAVAILABLE / WARNING_CODE_SERVER_DRAINING on drain."""
    from backend.app.testers.live_stt import asr_pb2

    with MockLiveSTTGRPCServer(max_concurrent_calls=2) as server:
        harness = LiveSTTStreamingHarness(server.target)
        fix = get_audio_fixture("short_speech")

        proto_res = run_protocol_and_drain_contract_check(
            harness,
            model=DEFAULT_GRPC_STREAMING_MODEL,
        )
        assert proto_res["passed"], proto_res["errors"]
        assert len(proto_res["checks"]) == 6

        # Also verify server draining behavior (UNAVAILABLE on new calls + WARNING_CODE_SERVER_DRAINING)
        server.servicer.draining = True
        drain_sess = harness.run_session_sync(
            fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            realtime=False,
            raise_on_grpc_error=False,
        )
        assert drain_sess.ok is False
        assert drain_sess.grpc_status_code == "UNAVAILABLE"
        server.servicer.draining = False

        server.servicer.emit_warning = {
            "code": int(asr_pb2.WARNING_CODE_SERVER_DRAINING),
            "message": "Server draining for maintenance",
        }
        warn_sess = harness.run_session_sync(
            fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            realtime=False,
        )
        assert warn_sess.ok is True
        assert len(warn_sess.warnings) == 1
        assert warn_sess.warnings[0].detail.get("code") == "WARNING_CODE_SERVER_DRAINING"


def test_grpc_long_call_stability_with_worker_recycling():
    """Verify long-call streaming handles periodic worker recycling (Recycled events) and rebased timestamps."""
    with MockLiveSTTGRPCServer(
        max_concurrent_calls=2,
        recycle_every_sec=3.0,
        final_delay_sec=0.005,
    ) as server:
        harness = LiveSTTStreamingHarness(server.target)
        long_fix = build_long_call_audio_fixture(duration_sec=10.0)

        session = harness.run_session_sync(
            long_fix,
            model=DEFAULT_GRPC_STREAMING_MODEL,
            language="en",
            chunk_ms=100.0,
            realtime=True,
            speed=20.0,
        )
        info_after = harness.get_server_info_sync()
        assert info_after.active_calls == 0

        report = evaluate_long_call_stability(
            session,
            expected_duration_sec=10.0,
            max_rtf=2.0,
            max_drift_ratio=5.0,
        )
        assert report.passed, report.errors
        assert report.recycled_count >= 2
        assert report.worker_generations >= 3
        assert report.partial_count >= 6
        assert report.word_count >= 8
        assert report.word_timestamps_monotonic is True
        assert report.recycled_rebased_properly is True


# ===========================================================================
# 3. End-to-End LiveSTTTester Execution: grpc_streaming & grpc_long_call
# ===========================================================================


def test_live_stt_tester_grpc_streaming_suite_end_to_end(tmp_path: Path):
    """Run LiveSTTTester with suites=['grpc_streaming'] against MockLiveSTTGRPCServer."""
    with MockLiveSTTGRPCServer(max_concurrent_calls=2, final_delay_sec=0.005) as grpc_server:
        tester = LiveSTTTester()
        ctx = _make_ctx(tmp_path, run_id=3102)

        result = tester.run(
            config={
                "suites": ["grpc_streaming"],
                "grpc_target": grpc_server.target,
                "discover_models": False,
                "grpc_speed": 8.0,
                "grpc_burst_speed": 20.0,
                "repetitions": 1,
            },
            target={
                "name": "mock-grpc-live-stt",
                "grpc_target": grpc_server.target,
            },
            ctx=ctx,
        )

        assert result.status == "pass", [
            (c.case_id, c.status, c.detail.get("errors")) for c in result.cases if c.status != "pass"
        ]

        case_ids = [c.case_id for c in result.cases]
        assert "grpc_streaming:get_server_info" in case_ids
        assert f"grpc_streaming:{DEFAULT_GRPC_STREAMING_MODEL}:short_speech" in case_ids
        assert f"grpc_streaming:{DEFAULT_GRPC_STREAMING_MODEL}:medium_speech" in case_ids
        assert f"grpc_streaming:{DEFAULT_GRPC_STREAMING_MODEL}:fast_paced_burst" in case_ids
        assert "grpc_streaming:concurrent_within_budget" in case_ids
        assert "grpc_streaming:concurrent_over_budget_and_recovery" in case_ids
        assert "grpc_streaming:reconnect_after_client_abort" in case_ids
        assert "grpc_streaming:clean_half_close_shutdown" in case_ids
        assert "grpc_streaming:protocol_error_contract" in case_ids

        # Verify opt-in grpc_long_call did NOT run when only grpc_streaming was requested
        assert not any(c.suite == "grpc_long_call" for c in result.cases)

        # Verify gRPC streaming aggregate metrics
        assert "grpc_ttfp_mean_ms" in result.metrics
        assert result.metrics["grpc_ttfp_mean_ms"]["value"] > 0.0
        assert "grpc_ttfp_p50_ms" in result.metrics
        assert "grpc_ttfp_p95_ms" in result.metrics
        assert "grpc_final_latency_mean_ms" in result.metrics
        assert result.metrics["grpc_final_latency_mean_ms"]["value"] > 0.0
        assert "grpc_rtf_mean" in result.metrics
        assert result.metrics["grpc_rtf_mean"]["value"] > 0.0

        # Verify report artifact contains gRPC Transcribe section
        report_txt = (ctx.artifacts_dir / "live_stt_report.txt").read_text(encoding="utf-8")
        assert "gRPC Transcribe" in report_txt
        assert "TTFP" in report_txt


def test_live_stt_tester_grpc_long_call_opt_in_suite_end_to_end(tmp_path: Path):
    """Run LiveSTTTester with opt-in grpc_long_call suite and verify worker rotation & metrics."""
    with MockLiveSTTGRPCServer(
        max_concurrent_calls=2,
        recycle_every_sec=3.0,
        final_delay_sec=0.005,
    ) as grpc_server:
        tester = LiveSTTTester()
        ctx = _make_ctx(tmp_path, run_id=3103)

        result = tester.run(
            config={
                "suites": ["grpc_streaming"],
                "include_long_call": True,
                "grpc_target": grpc_server.target,
                "discover_models": False,
                "grpc_speed": 10.0,
                "grpc_burst_speed": 25.0,
                "long_call_duration_sec": 10.0,
                "long_call_speed": 20.0,
            },
            target={
                "name": "mock-grpc-long-call",
                "grpc_target": grpc_server.target,
            },
            ctx=ctx,
        )

        assert result.status == "pass", [
            (c.case_id, c.status, c.detail.get("errors")) for c in result.cases if c.status != "pass"
        ]

        long_cases = [c for c in result.cases if c.suite == "grpc_long_call"]
        assert len(long_cases) == 1
        lc = long_cases[0]
        assert lc.case_id == f"grpc_long_call:{DEFAULT_GRPC_STREAMING_MODEL}:sustained_stream"
        assert lc.status == "pass"
        assert lc.metrics["worker_generations"]["value"] >= 3
        assert lc.metrics["recycled_count"]["value"] >= 2
        assert lc.metrics["word_count"]["value"] >= 8
        assert lc.detail["stability"]["word_timestamps_monotonic"] is True

        # Verify aggregate long-call metrics
        assert result.metrics["grpc_long_call_duration_sec"]["value"] >= 9.5
        assert result.metrics["grpc_long_call_worker_generations"]["value"] >= 3
        assert result.metrics["grpc_long_call_recycled_count"]["value"] >= 2
        assert "grpc_long_call_rtf" in result.metrics

        report_txt = (ctx.artifacts_dir / "live_stt_report.txt").read_text(encoding="utf-8")
        assert "gRPC Long-Call" in report_txt


# ===========================================================================
# Real-speech fixtures and VRAM-limited within-budget concurrency
# ===========================================================================


def test_classify_within_budget_rejections():
    from backend.app.testers.live_stt.streaming import classify_within_budget_rejections

    def sess(ok: bool, code: str, msg: str | None = None):
        return type("S", (), {"ok": ok, "grpc_status_code": code, "error_message": msg})()

    out = classify_within_budget_rejections(
        [
            sess(True, "OK"),
            sess(False, "RESOURCE_EXHAUSTED", "insufficient VRAM for m: 120MB free, 250MB required"),
            sess(False, "RESOURCE_EXHAUSTED", "insufficient VRAM for m: 100MB free, 250MB required"),
            sess(False, "RESOURCE_EXHAUSTED", "at capacity, try again shortly"),
            sess(False, "UNAVAILABLE", "connection refused"),
        ]
    )
    assert (out["vram_limited"], out["capacity_rejected"], out["other_failed"]) == (2, 1, 1)
    vram_group = next(g for g in out["groups"] if g["kind"] == "vram_limited")
    assert vram_group["count"] == 2 and vram_group["sample_message"].startswith("insufficient VRAM")


def test_grpc_concurrent_within_budget_accepts_vram_limited_rejections(tmp_path: Path):
    """Run 31: 20 streams <= max_concurrent_calls=20, GPU fits 14 -> clean VRAM rejections are not a failure."""
    with MockLiveSTTGRPCServer(max_concurrent_calls=4, vram_capacity_calls=2, final_delay_sec=0.005) as grpc_server:
        result = LiveSTTTester().run(
            config={
                "suites": ["grpc_streaming"],
                "grpc_target": grpc_server.target,
                "discover_models": False,
                "grpc_speed": 8.0,
                "repetitions": 1,
                "speech_fixture": "synthetic",
            },
            target={"name": "mock-grpc-vram", "grpc_target": grpc_server.target},
            ctx=_make_ctx(tmp_path, run_id=3110),
        )

    within = next(c for c in result.cases if c.case_id == "grpc_streaming:concurrent_within_budget")
    assert within.status == "pass", within.detail.get("errors")
    d = within.detail
    assert d["concurrency"] == 4 and d["ok_count"] == 2
    assert d["vram_limited_count"] == 2 and d["capacity_rejected_count"] == 0 and d["other_failed_count"] == 0
    assert within.metrics["vram_limited_sessions"]["value"] == 2
    assert any(ch["name"] == "vram_limited_sessions_rejected_cleanly" for ch in d["checks"])
    assert "limited by VRAM" in d["eval_message"]


def test_grpc_streaming_uses_real_speech_fixtures_with_wer(tmp_path: Path):
    from backend.app.testers.live_stt.dataset import build_notsofar_speech_fixture
    from backend.tests.test_live_stt_accuracy import _create_long_synthetic_notsofar_split

    ds_root = _create_long_synthetic_notsofar_split(tmp_path / "NOTSOFAR-1")
    short = build_notsofar_speech_fixture("short_speech", dataset_dir=ds_root)
    assert short is not None

    # The mock servicer "recognizes" exactly the short excerpt's words
    with MockLiveSTTGRPCServer(
        max_concurrent_calls=2, final_delay_sec=0.005, words_sequence=list(short.words),
        repeat_words_for_long_audio=False,
    ) as grpc_server:
        result = LiveSTTTester().run(
            config={
                "suites": ["grpc_streaming"],
                "grpc_target": grpc_server.target,
                "discover_models": False,
                "grpc_speed": 8.0,
                "grpc_burst_speed": 20.0,
                "repetitions": 1,
                "dataset_dir": str(ds_root),
                "speech_wer_threshold": 5.0,
            },
            target={"name": "mock-grpc-real-speech", "grpc_target": grpc_server.target},
            ctx=_make_ctx(tmp_path, run_id=3111),
        )

    per_model = {
        c.case_id.rsplit(":", 1)[-1]: c
        for c in result.cases
        if c.case_id.startswith(f"grpc_streaming:{DEFAULT_GRPC_STREAMING_MODEL}:")
    }
    assert set(per_model) == {"short_speech", "medium_speech", "fast_paced_burst"}
    for c in per_model.values():
        assert c.detail["fixture_source"] == "notsofar"
        assert c.detail["fixture_id"].startswith("notsofar:MTG_91001:")
        assert any(ch["name"] == "speech_wer_within_threshold" for ch in c.detail["checks"])
    assert per_model["short_speech"].metrics["wer"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert per_model["short_speech"].status == "pass"
