"""Tests for NOTSOFAR-1 dataset loader and live-stt accuracy suites (WER, cpWER, DER, speaker count, result cache)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wave

import httpx
import pytest

from backend.app.api.compare import _extract_model_agnostic_case_id
from backend.app.testers.base import TestContext
from backend.app.testers.live_stt import (
    AccuracyResultCache,
    LiveSTTTester,
    MockLiveSTTGRPCServer,
    RecordedWord,
    align_words_to_segments,
    asr_pb2,
    compute_cpwer,
    compute_der_and_speaker_error,
    compute_wer,
    discover_notsofar_meetings,
    extract_speaker_transcripts_from_segments,
    load_notsofar_meeting,
    load_notsofar_meetings,
    make_pcm_wav_bytes,
    normalize_live_stt_suites,
    normalize_text_for_scoring,
    parse_meeting_selection,
    resolve_notsofar_root,
    resolve_notsofar_split_dir,
    tokenize_for_scoring,
)


def _create_synthetic_notsofar_split(root_dir: Path) -> Path:
    """Create a hermetic miniature NOTSOFAR-1 dataset with 2 meetings for fast unit/integration tests."""
    split_dir = root_dir / "eval_set" / "240629.1_eval_small_with_GT" / "MTG"
    meetings_spec = [
        {
            "meeting_id": "MTG_90001",
            "duration_sec": 3.0,
            "participants": ["Alice", "Bob"],
            "segments": [
                {
                    "speaker_id": "Alice",
                    "start_time": 0.2,
                    "end_time": 1.4,
                    "text": "<ST/> Hello <PName>Bob</PName>, let's start the quarterly review.",
                    "word_timing": [
                        ["Hello", 0.2, 0.4],
                        ["Bob", 0.4, 0.6],
                        ["let's", 0.6, 0.8],
                        ["start", 0.8, 1.0],
                        ["the", 1.0, 1.1],
                        ["quarterly", 1.1, 1.25],
                        ["review", 1.25, 1.4],
                    ],
                },
                {
                    "speaker_id": "Bob",
                    "start_time": 1.6,
                    "end_time": 2.8,
                    "text": "Sounds good <FILL/> I have the metrics ready.",
                    "word_timing": [
                        ["Sounds", 1.6, 1.8],
                        ["good", 1.8, 2.0],
                        ["I", 2.0, 2.1],
                        ["have", 2.1, 2.3],
                        ["the", 2.3, 2.4],
                        ["metrics", 2.4, 2.6],
                        ["ready", 2.6, 2.8],
                    ],
                },
            ],
        },
        {
            "meeting_id": "MTG_90002",
            "duration_sec": 3.0,
            "participants": ["Carol", "Dave", "Eve"],
            "segments": [
                {
                    "speaker_id": "Carol",
                    "start_time": 0.1,
                    "end_time": 1.0,
                    "text": "Deployment succeeded on staging.",
                    "word_timing": [
                        ["Deployment", 0.1, 0.4],
                        ["succeeded", 0.4, 0.7],
                        ["on", 0.7, 0.8],
                        ["staging", 0.8, 1.0],
                    ],
                },
                {
                    "speaker_id": "Dave",
                    "start_time": 1.1,
                    "end_time": 2.0,
                    "text": "Great, latency is within budget.",
                    "word_timing": [
                        ["Great", 1.1, 1.3],
                        ["latency", 1.3, 1.5],
                        ["is", 1.5, 1.6],
                        ["within", 1.6, 1.8],
                        ["budget", 1.8, 2.0],
                    ],
                },
                {
                    "speaker_id": "Eve",
                    "start_time": 2.1,
                    "end_time": 2.9,
                    "text": "Let us ship to production.",
                    "word_timing": [
                        ["Let", 2.1, 2.25],
                        ["us", 2.25, 2.4],
                        ["ship", 2.4, 2.55],
                        ["to", 2.55, 2.7],
                        ["production", 2.7, 2.9],
                    ],
                },
            ],
        },
    ]

    for spec in meetings_spec:
        m_dir = split_dir / spec["meeting_id"]
        sc_dir = m_dir / "sc_meetup_0"
        ct_dir = m_dir / "close_talk"
        sc_dir.mkdir(parents=True, exist_ok=True)
        ct_dir.mkdir(parents=True, exist_ok=True)

        wav_bytes = make_pcm_wav_bytes(duration_sec=spec["duration_sec"], sample_rate=16000)
        (sc_dir / "ch0.wav").write_bytes(wav_bytes)
        (ct_dir / "CT_0.wav").write_bytes(wav_bytes)

        metadata = {
            "meeting_id": spec["meeting_id"],
            "Topic": "Engineering Sync",
            "MeetingDurationSec": spec["duration_sec"],
            "NumParticipants": len(spec["participants"]),
            "ParticipantAliases": spec["participants"],
        }
        (m_dir / "gt_meeting_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (m_dir / "gt_transcription.json").write_text(json.dumps(spec["segments"]), encoding="utf-8")
        devices = [
            {
                "device_name": "mc_ularge_0",
                "is_mc": True,
                "is_close_talk": False,
                "channels_num": 7,
                "wav_file_names": "mc_ularge_0",
            },
            {
                "device_name": "sc_meetup_0",
                "is_mc": False,
                "is_close_talk": False,
                "channels_num": 1,
                "wav_file_names": "sc_meetup_0/ch0.wav",
            },
            {
                "device_name": "CT_0",
                "is_mc": False,
                "is_close_talk": True,
                "channels_num": 1,
                "wav_file_names": "close_talk/CT_0.wav",
            },
        ]
        (m_dir / "devices.json").write_text(json.dumps(devices), encoding="utf-8")

    return root_dir


LONG_MEETING_WORDS = (
    "good morning everyone thanks for joining the weekly planning call today we will review "
    "the release schedule the open bugs and the staffing plan for next quarter so let us start "
    "with the release schedule because that is the most urgent item on the agenda this week"
).split()


def _create_long_synthetic_notsofar_split(root_dir: Path) -> Path:
    """Hermetic NOTSOFAR-1 split with one 20s, two-speaker meeting (MTG_91001) whose speech starts at
    1.0s: long enough for the 6s/15s ASR speech excerpts and the 30s-capped dialog excerpt."""
    split_dir = root_dir / "eval_set" / "240629.1_eval_small_with_GT" / "MTG"
    m_dir = split_dir / "MTG_91001"
    sc_dir = m_dir / "sc_meetup_0"
    sc_dir.mkdir(parents=True, exist_ok=True)
    (sc_dir / "ch0.wav").write_bytes(make_pcm_wav_bytes(duration_sec=20.0, sample_rate=16000))

    segments = []
    t = 1.0
    words = list(LONG_MEETING_WORDS)
    speakers = ["Alice", "Bob"]
    seg_idx = 0
    while words and t < 19.0:
        chunk, words = words[:9], words[9:]
        timing = []
        for w in chunk:
            timing.append([w, round(t, 2), round(t + 0.35, 2)])
            t += 0.4
        segments.append(
            {
                "speaker_id": speakers[seg_idx % 2],
                "start_time": timing[0][1],
                "end_time": timing[-1][2],
                "text": " ".join(chunk),
                "word_timing": timing,
            }
        )
        seg_idx += 1
        t += 0.3

    metadata = {
        "meeting_id": "MTG_91001",
        "Topic": "Planning",
        "MeetingDurationSec": 20.0,
        "NumParticipants": 2,
        "ParticipantAliases": speakers,
    }
    (m_dir / "gt_meeting_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (m_dir / "gt_transcription.json").write_text(json.dumps(segments), encoding="utf-8")
    devices = [
        {
            "device_name": "sc_meetup_0",
            "is_mc": False,
            "is_close_talk": False,
            "channels_num": 1,
            "wav_file_names": "sc_meetup_0/ch0.wav",
        }
    ]
    (m_dir / "devices.json").write_text(json.dumps(devices), encoding="utf-8")
    return root_dir


# ===========================================================================
# 1. NOTSOFAR-1 Dataset Loader & Text Normalization Tests
# ===========================================================================


def test_normalize_text_for_scoring_strips_notsofar_tags_and_punctuation():
    raw = (
        "<ST/> Okay <PName>Emily</PName>, <FILL/> don\u2019t forget: "
        "we're testing 100% of <FILLlaugh/> <BA/> <UNKNOWN/> cases!"
    )
    norm = normalize_text_for_scoring(raw)
    assert norm == "okay emily don't forget we're testing 100 of cases"
    assert tokenize_for_scoring(raw) == [
        "okay",
        "emily",
        "don't",
        "forget",
        "we're",
        "testing",
        "100",
        "of",
        "cases",
    ]
    assert normalize_text_for_scoring("") == ""
    assert normalize_text_for_scoring(None) == ""


def test_synthetic_notsofar_loader_and_duration_slicing(tmp_path: Path):
    ds_root = _create_synthetic_notsofar_split(tmp_path / "NOTSOFAR-1")

    ids = discover_notsofar_meetings(dataset_dir=ds_root, split="eval_small")
    assert ids == ["MTG_90001", "MTG_90002"]
    assert parse_meeting_selection("MTG_90002", ids) == ["MTG_90002"]
    assert parse_meeting_selection(None, ids, max_meetings=1) == ["MTG_90001"]

    m1 = load_notsofar_meeting("MTG_90001", dataset_dir=ds_root, split="eval_small")
    assert m1.meeting_id == "MTG_90001"
    assert m1.device_alias == "sc_meetup_0"
    assert m1.expected_speakers == 2
    assert m1.normalized_reference_text == (
        "hello bob let's start the quarterly review sounds good i have the metrics ready"
    )
    assert m1.speaker_reference_texts == {
        "Alice": "hello bob let's start the quarterly review",
        "Bob": "sounds good i have the metrics ready",
    }

    # Test prefer_close_talk option
    m1_ct = load_notsofar_meeting(
        "MTG_90001",
        dataset_dir=ds_root,
        split="eval_small",
        prefer_close_talk=True,
    )
    assert m1_ct.device_alias == "CT_0"

    # Test max_duration_sec clipping
    m1_clipped = load_notsofar_meeting(
        "MTG_90001",
        dataset_dir=ds_root,
        split="eval_small",
        max_duration_sec=1.5,
    )
    assert m1_clipped.duration_sec == 1.5
    assert len(m1_clipped.segments) == 1
    assert m1_clipped.segments[0].speaker_id == "Alice"
    assert m1_clipped.normalized_reference_text == "hello bob let's start the quarterly review"
    with wave.open(str(m1_clipped.audio_path), "rb") as wf:
        assert wf.getframerate() == 16000
    wav_bytes = m1_clipped.wav_bytes
    assert wav_bytes[:4] == b"RIFF"


def test_notsofar_split_aliases_selection_limits_and_wav_normalization(tmp_path: Path):
    """Verify split resolution aliases, meeting count/filter limits, mid-segment word clipping, and WAV resampling."""
    from backend.app.testers.live_stt.dataset import load_and_normalize_wav_bytes

    ds_root = _create_synthetic_notsofar_split(tmp_path / "NOTSOFAR-1")

    # Split aliases ("eval", "default") and direct split/MTG directory paths
    mtg_dir_eval, split_eval = resolve_notsofar_split_dir(dataset_dir=ds_root, split="eval")
    mtg_dir_def, split_def = resolve_notsofar_split_dir(dataset_dir=ds_root, split="default")
    assert mtg_dir_eval == mtg_dir_def
    assert split_eval == "eval_set/240629.1_eval_small_with_GT"

    direct_split_dir = ds_root / "eval_set" / "240629.1_eval_small_with_GT"
    mtg_from_split, _ = resolve_notsofar_split_dir(dataset_dir=direct_split_dir)
    mtg_from_mtg, _ = resolve_notsofar_split_dir(dataset_dir=mtg_dir_eval)
    assert mtg_from_split == mtg_dir_eval
    assert mtg_from_mtg == mtg_dir_eval

    with pytest.raises(FileNotFoundError):
        resolve_notsofar_split_dir(dataset_dir=ds_root, split="nonexistent_split")

    # Meeting selection: comma-separated string, deduplication, max_meetings limit, and unknown ID error
    avail = ["MTG_90001", "MTG_90002"]
    assert parse_meeting_selection("MTG_90002, MTG_90001, MTG_90002", avail) == ["MTG_90002", "MTG_90001"]
    assert parse_meeting_selection(["MTG_90001", "MTG_90002"], avail, max_meetings=1) == ["MTG_90001"]
    with pytest.raises(ValueError, match="Unknown NOTSOFAR-1 meeting ID"):
        parse_meeting_selection("MTG_99999", avail)

    # Mid-segment word-timing clipping and load_notsofar_meetings with max_meetings
    loaded = load_notsofar_meetings(
        dataset_dir=ds_root,
        split="eval_small",
        max_meetings=1,
        max_duration_sec=0.95,
    )
    assert len(loaded) == 1
    assert loaded[0].meeting_id == "MTG_90001"
    assert loaded[0].normalized_reference_text == "hello bob let's start"
    assert loaded[0].reference_speakers == ["Alice"]
    assert len(loaded[0].gt_segments_dicts) == 1
    assert loaded[0].to_dict()["reference_word_count"] == 4

    # Non-16kHz / stereo WAV resampling and downmixing to 16kHz mono 16-bit PCM
    import io

    raw_8k = make_pcm_wav_bytes(duration_sec=1.0, sample_rate=8000, channels=2)
    norm_wav, norm_dur = load_and_normalize_wav_bytes(raw_8k, max_duration_sec=0.5)
    assert norm_dur == pytest.approx(0.5, abs=0.02)
    with wave.open(io.BytesIO(norm_wav), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2


def test_real_notsofar_dataset_on_disk_discovery_and_loading():
    """Verify real NOTSOFAR-1 dataset under test-data/NOTSOFAR-1 if present on disk."""
    try:
        root = resolve_notsofar_root()
    except FileNotFoundError:
        pytest.skip("Real NOTSOFAR-1 dataset not mounted on this machine")
    if not root.is_dir():
        # resolve_notsofar_root() returns the expected path even when absent.
        pytest.skip("Real NOTSOFAR-1 dataset not mounted on this machine")

    assert root.is_dir()
    mtg_dir, split_key = resolve_notsofar_split_dir(dataset_dir=root, split="eval_small")
    assert split_key == "eval_set/240629.1_eval_small_with_GT"
    assert mtg_dir.is_dir()

    eval_small_ids = discover_notsofar_meetings(dataset_dir=root, split="eval_small")
    assert len(eval_small_ids) == 80
    assert "MTG_32000" in eval_small_ids

    eval_full_ids = discover_notsofar_meetings(dataset_dir=root, split="eval_full")
    assert len(eval_full_ids) == 129

    dev_ids = discover_notsofar_meetings(dataset_dir=root, split="dev")
    assert len(dev_ids) == 36

    train_ids = discover_notsofar_meetings(dataset_dir=root, split="train")
    assert len(train_ids) == 72

    meetings = load_notsofar_meetings(
        dataset_dir=root,
        split="eval_small",
        meetings=["MTG_32000", "MTG_32003"],
        max_duration_sec=15.0,
    )
    assert len(meetings) == 2
    assert meetings[0].meeting_id == "MTG_32000"
    assert meetings[0].duration_sec == 15.0
    assert meetings[0].device_alias in ("meetup_0", "sc_meetup_0", "rockfall_0", "sc_rockfall_0")
    assert len(meetings[0].wav_bytes) > 44
    assert len(meetings[0].reference_words) > 0


# ===========================================================================
# 2. WER, cpWER, DER, and Speaker-Count Error Unit Tests
# ===========================================================================


def test_compute_wer_exact_and_edits():
    res_exact = compute_wer(
        "<ST/> Hello <PName>Bob</PName>, how are you?",
        "hello bob how are you",
        include_ops=True,
    )
    assert res_exact.wer == 0.0
    assert res_exact.wer_pct == 0.0
    assert res_exact.substitutions == 0
    assert res_exact.deletions == 0
    assert res_exact.insertions == 0
    assert res_exact.hits == 5
    assert len(res_exact.ops) == 5

    # 2 substitutions on 4 ref words -> 2/4 = 0.50
    res_edits = compute_wer("hello world how today", "hello there how now", include_ops=True)
    assert res_edits.ref_word_count == 4
    assert res_edits.hyp_word_count == 4
    assert res_edits.edit_distance == 2
    assert res_edits.wer == 0.5


def test_compute_cpwer_permutation_invariance_and_speaker_mismatch():
    ref_speakers = {
        "spk_A": "the quick brown fox jumps",
        "spk_B": "over the lazy dog today",
    }
    # Hypothesis has permuted speaker IDs (SPEAKER_01 == spk_A, SPEAKER_00 == spk_B)
    hyp_permuted = {
        "SPEAKER_00": "over the lazy dog today",
        "SPEAKER_01": "the quick brown fox jumps",
    }
    cp_exact = compute_cpwer(ref_speakers, hyp_permuted)
    assert cp_exact.cpwer == 0.0
    assert cp_exact.edit_distance == 0
    assert cp_exact.ref_word_count == 10
    assert cp_exact.speaker_mapping == {"SPEAKER_01": "spk_A", "SPEAKER_00": "spk_B"}

    # Missing second speaker in hypothesis -> 5 deletions out of 10 reference words = 0.5 cpWER
    cp_missing = compute_cpwer(ref_speakers, {"SPEAKER_00": "the quick brown fox jumps"})
    assert cp_missing.cpwer == 0.5
    assert cp_missing.deletions == 5
    assert cp_missing.ref_speaker_count == 2
    assert cp_missing.hyp_speaker_count == 1


def test_align_words_to_segments_and_extract_speaker_transcripts():
    words = [
        {"word": "hello", "start": 0.1, "end": 0.4},
        {"word": "bob", "start": 0.5, "end": 0.9},
        {"word": "hi", "start": 1.2, "end": 1.5},
        {"word": "alice", "start": 1.6, "end": 1.9},
    ]
    segments = [
        {"speaker": "SPEAKER_00", "start_sec": 0.0, "end_sec": 1.0},
        {"speaker": "SPEAKER_01", "start_sec": 1.1, "end_sec": 2.0},
    ]
    aligned = align_words_to_segments(words, segments)
    assert aligned[0]["text"] == "hello bob"
    assert aligned[1]["text"] == "hi alice"

    spk_texts = extract_speaker_transcripts_from_segments(segments, words)
    assert spk_texts == {
        "SPEAKER_00": "hello bob",
        "SPEAKER_01": "hi alice",
    }


def test_compute_der_and_speaker_error():
    gt_segments = [
        {"speaker": "Alice", "start_sec": 0.0, "end_sec": 2.0},
        {"speaker": "Bob", "start_sec": 2.0, "end_sec": 4.0},
    ]
    # Permuted speaker labels with identical boundaries -> 0% DER, 0 speaker count error
    pred_exact = [
        {"speaker": "SPEAKER_01", "start_sec": 0.0, "end_sec": 2.0},
        {"speaker": "SPEAKER_00", "start_sec": 2.0, "end_sec": 4.0},
    ]
    der_exact = compute_der_and_speaker_error(
        gt_segments=gt_segments,
        pred_segments=pred_exact,
        duration_sec=4.0,
        expected_speakers=2,
        predicted_speakers=2,
    )
    assert der_exact.der == 0.0
    assert der_exact.missed_rate == 0.0
    assert der_exact.false_alarm_rate == 0.0
    assert der_exact.confusion_rate == 0.0
    assert der_exact.speaker_count_error == 0
    assert der_exact.speaker_count_exact is True
    assert der_exact.speaker_error_rate == 0.0

    # Imperfect prediction: 1.0s missed speech, +1 extra speaker predicted
    pred_imperfect = [
        {"speaker": "SPEAKER_00", "start_sec": 0.0, "end_sec": 1.0},
        {"speaker": "SPEAKER_01", "start_sec": 2.0, "end_sec": 4.0},
        {"speaker": "SPEAKER_02", "start_sec": 4.2, "end_sec": 4.5},
    ]
    der_imp = compute_der_and_speaker_error(
        gt_segments=gt_segments,
        pred_segments=pred_imperfect,
        duration_sec=5.0,
        expected_speakers=2,
        predicted_speakers=3,
    )
    assert der_imp.der > 0.0
    assert der_imp.missed_speech_sec == pytest.approx(1.0, abs=0.1)
    assert der_imp.false_alarm_sec == pytest.approx(0.3, abs=0.1)
    assert der_imp.speaker_count_error == 1
    assert der_imp.speaker_count_delta == 1
    assert der_imp.speaker_count_exact is False
    assert der_imp.speaker_error_rate == 0.5


# ===========================================================================
# 3. AccuracyResultCache Unit Tests
# ===========================================================================


def test_accuracy_result_cache_hit_miss_and_skip(tmp_path: Path):
    cache_file = tmp_path / "cache" / "eval_cache.sqlite3"
    cache = AccuracyResultCache(cache_file, enabled=True, skip_read=False)

    key1 = {"suite": "accuracy_wer", "model_id": "parakeet", "meeting_id": "MTG_90001"}
    assert cache.get(key1) is None
    assert cache.misses == 1
    assert cache.hits == 0

    cache.put(key1, {"status_code": 200, "body": {"text": "hello world"}})
    assert cache.writes == 1

    hit_val = cache.get(key1)
    assert hit_val is not None
    assert hit_val["body"]["text"] == "hello world"
    assert cache.hits == 1

    # Different config -> miss
    key2 = {"suite": "accuracy_wer", "model_id": "whisper", "meeting_id": "MTG_90001"}
    assert cache.get(key2) is None
    assert cache.misses == 2

    # skip_read=True bypasses reading from cache
    cache_skip = AccuracyResultCache(cache_file, enabled=True, skip_read=True)
    assert cache_skip.get(key1) is None
    assert cache_skip.misses == 1

    removed = cache.clear()
    assert removed == 1
    assert cache.get(key1) is None


# ===========================================================================
# 4. End-to-End Accuracy Suites (Batch HTTP, Streaming gRPC/SSE, Diarization, Cache)
# ===========================================================================


def test_live_stt_accuracy_suites_end_to_end_with_cache_and_grpc(tmp_path: Path):
    """Verify accuracy_wer, accuracy_streaming (gRPC + SSE), accuracy_diarization, cpWER, and SQLite result cache."""
    from backend.app.testers.live_stt.fixtures import WordTimestamp

    ds_root = _create_synthetic_notsofar_split(tmp_path / "NOTSOFAR-1")
    cache_file = tmp_path / "notsofar_cache.sqlite3"

    request_counts = {"transcriptions_json": 0, "transcriptions_sse": 0, "diarization": 0}

    # Meeting responses matching MTG_90001 and MTG_90002 ground truth
    mtg_responses: dict[str, dict[str, Any]] = {
        "MTG_90001.wav": {
            "text": "Hello Bob let's start the quarterly review Sounds good I have the metrics ready",
            "words": [
                {"word": "Hello", "start": 0.2, "end": 0.4},
                {"word": "Bob", "start": 0.4, "end": 0.6},
                {"word": "let's", "start": 0.6, "end": 0.8},
                {"word": "start", "start": 0.8, "end": 1.0},
                {"word": "the", "start": 1.0, "end": 1.1},
                {"word": "quarterly", "start": 1.1, "end": 1.25},
                {"word": "review", "start": 1.25, "end": 1.4},
                {"word": "Sounds", "start": 1.6, "end": 1.8},
                {"word": "good", "start": 1.8, "end": 2.0},
                {"word": "I", "start": 2.0, "end": 2.1},
                {"word": "have", "start": 2.1, "end": 2.3},
                {"word": "the", "start": 2.3, "end": 2.4},
                {"word": "metrics", "start": 2.4, "end": 2.6},
                {"word": "ready", "start": 2.6, "end": 2.8},
            ],
            "diar_segments": [
                {"speaker": "SPEAKER_00", "start_sec": 0.2, "end_sec": 1.4},
                {"speaker": "SPEAKER_01", "start_sec": 1.6, "end_sec": 2.8},
            ],
            "num_speakers": 2,
        },
        "MTG_90002.wav": {
            "text": "Deployment succeeded on staging Great latency is within budget Let us ship to production",
            "words": [
                {"word": "Deployment", "start": 0.1, "end": 0.4},
                {"word": "succeeded", "start": 0.4, "end": 0.7},
                {"word": "on", "start": 0.7, "end": 0.8},
                {"word": "staging", "start": 0.8, "end": 1.0},
                {"word": "Great", "start": 1.1, "end": 1.3},
                {"word": "latency", "start": 1.3, "end": 1.5},
                {"word": "is", "start": 1.5, "end": 1.6},
                {"word": "within", "start": 1.6, "end": 1.8},
                {"word": "budget", "start": 1.8, "end": 2.0},
                {"word": "Let", "start": 2.1, "end": 2.25},
                {"word": "us", "start": 2.25, "end": 2.4},
                {"word": "ship", "start": 2.4, "end": 2.55},
                {"word": "to", "start": 2.55, "end": 2.7},
                {"word": "production", "start": 2.7, "end": 2.9},
            ],
            "diar_segments": [
                {"speaker": "SPEAKER_00", "start_sec": 0.1, "end_sec": 1.0},
                {"speaker": "SPEAKER_01", "start_sec": 1.1, "end_sec": 2.0},
                {"speaker": "SPEAKER_02", "start_sec": 2.1, "end_sec": 2.9},
            ],
            "num_speakers": 3,
        },
    }

    def mock_http_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body_bytes = request.read()
        body_text = body_bytes.decode("latin-1", errors="ignore")
        mtg_key = "MTG_90002.wav" if "MTG_90002.wav" in body_text else "MTG_90001.wav"
        spec = mtg_responses[mtg_key]

        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "parakeet-tdt-0.6b-v2", "object": "model", "type": "asr", "engine": "parakeet"},
                        {"id": "pyannote/speaker-diarization-3.1", "object": "model", "type": "diarization", "engine": "pyannote"},
                    ],
                },
            )
        if path == "/v1/audio/transcriptions":
            if 'name="stream"\r\n\r\ntrue' in body_text:
                request_counts["transcriptions_sse"] += 1
                sse_payload = (
                    f"data: {json.dumps({'text': spec['text'], 'words': spec['words']})}\n\n"
                    "data: [DONE]\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_payload.encode("utf-8"),
                )
            request_counts["transcriptions_json"] += 1
            return httpx.Response(
                200,
                json={
                    "text": spec["text"],
                    "language": "en",
                    "duration": 3.0,
                    "words": spec["words"],
                },
            )
        if path == "/v1/audio/diarization":
            request_counts["diarization"] += 1
            return httpx.Response(
                200,
                json={
                    "num_speakers": spec["num_speakers"],
                    "duration": 3.0,
                    "segments": spec["diar_segments"],
                    "speakers": [
                        {"speaker": f"SPEAKER_{i:02d}", "total_sec": 1.0, "segment_count": 1}
                        for i in range(spec["num_speakers"])
                    ],
                },
            )
        return httpx.Response(404, json={"error": {"message": "Not found"}})

    transport = httpx.MockTransport(mock_http_handler)

    # Start in-process MockLiveSTTGRPCServer for gRPC streaming path
    grpc_words = [
        WordTimestamp(text=w["word"], start_sec=w["start"], end_sec=w["end"])
        for w in mtg_responses["MTG_90001.wav"]["words"]
    ]
    with MockLiveSTTGRPCServer(
        words_sequence=grpc_words,
        allowed_models={"parakeet-tdt-0.6b-v2", "realtime_eou_120m-v1"},
    ) as grpc_server:
        tester = LiveSTTTester()
        assert normalize_live_stt_suites(["accuracy"], tester.suites) == [
            "accuracy_wer",
            "accuracy_streaming",
            "accuracy_diarization",
        ]

        artifacts_dir_1 = tmp_path / "artifacts_run1"
        artifacts_dir_1.mkdir(parents=True, exist_ok=True)
        ctx1 = TestContext(run_id=1, artifacts_dir=artifacts_dir_1, emit_stdout=False)

        cfg = {
            "suites": ["accuracy"],
            "dataset_dir": str(ds_root),
            "dataset_split": "eval_small",
            "meetings": ["MTG_90001"],
            "streaming_protocol": "both",
            "grpc_target": grpc_server.target,
            "cache_path": str(cache_file),
            "use_cache": True,
            "_transport": transport,
        }
        target = {"name": "mock-live-stt", "base_url": "http://mockserver/v1"}

        # First run: cache misses -> hits mock HTTP & gRPC servers
        res1 = tester.run(cfg, target, ctx1)
        assert res1.status == "pass"
        # 1 HTTP WER case + 2 streaming cases (grpc + sse) + 1 diarization case = 4 cases
        assert len(res1.cases) == 4
        assert request_counts["transcriptions_json"] == 1
        assert request_counts["transcriptions_sse"] == 1
        assert request_counts["diarization"] == 1

        # Verify aggregate accuracy metrics
        assert res1.metrics["wer"]["value"] == 0.0
        assert res1.metrics["http_wer"]["value"] == 0.0
        assert res1.metrics["streaming_wer"]["value"] == 0.0
        assert res1.metrics["cpwer"]["value"] == 0.0
        assert res1.metrics["http_cpwer"]["value"] == 0.0
        assert res1.metrics["streaming_cpwer"]["value"] == 0.0
        assert res1.metrics["der"]["value"] == 0.0
        assert res1.metrics["speaker_count_error"]["value"] == 0.0
        assert res1.metrics["speaker_count_exact_rate"]["value"] == 100.0
        assert res1.metrics["accuracy_cache_hits"]["value"] == 0
        assert res1.metrics["accuracy_cache_misses"]["value"] == 4

        # Verify HTML report artifact was generated
        html_report_path = artifacts_dir_1 / "notsofar_accuracy_report.html"
        assert html_report_path.is_file()
        html_content = html_report_path.read_text(encoding="utf-8")
        assert "NOTSOFAR-1 Accuracy Evaluation Report" in html_content
        assert "MTG_90001" in html_content

        # Verify Compare API model-agnostic case alignment on accuracy case IDs
        wer_case = next(c for c in res1.cases if c.suite == "accuracy_wer")
        assert _extract_model_agnostic_case_id(wer_case) == (
            "accuracy_wer:MTG_90001"
        )

        # Second run with unchanged config: 100% cache hits, zero new transcription/diarization calls!
        artifacts_dir_2 = tmp_path / "artifacts_run2"
        artifacts_dir_2.mkdir(parents=True, exist_ok=True)
        ctx2 = TestContext(run_id=2, artifacts_dir=artifacts_dir_2, emit_stdout=False)
        res2 = tester.run(cfg, target, ctx2)
        assert res2.status == "pass"
        assert res2.metrics["accuracy_cache_hits"]["value"] == 4
        assert res2.metrics["accuracy_cache_misses"]["value"] == 0
        assert request_counts["transcriptions_json"] == 1
        assert request_counts["transcriptions_sse"] == 1
        assert request_counts["diarization"] == 1
        assert all(c.detail.get("cache_hit") is True for c in res2.cases)

        # Third run with skip_cache=True: forces fresh evaluation
        artifacts_dir_3 = tmp_path / "artifacts_run3"
        artifacts_dir_3.mkdir(parents=True, exist_ok=True)
        ctx3 = TestContext(run_id=3, artifacts_dir=artifacts_dir_3, emit_stdout=False)
        cfg_skip = {**cfg, "skip_cache": True}
        res3 = tester.run(cfg_skip, target, ctx3)
        assert res3.status == "pass"
        assert res3.metrics["accuracy_cache_hits"]["value"] == 0
        assert res3.metrics["accuracy_cache_misses"]["value"] == 4
        assert request_counts["transcriptions_json"] == 2
        assert request_counts["transcriptions_sse"] == 2
        assert request_counts["diarization"] == 2


def test_real_notsofar_meetings_accuracy_suites(tmp_path: Path):
    """Run accuracy_wer, accuracy_streaming, and accuracy_diarization on real NOTSOFAR-1 meetings (MTG_32000, MTG_32003)."""
    try:
        root = resolve_notsofar_root()
    except FileNotFoundError:
        pytest.skip("Real NOTSOFAR-1 dataset not mounted on this machine")
    if not root.is_dir():
        # resolve_notsofar_root() returns the expected path even when absent.
        pytest.skip("Real NOTSOFAR-1 dataset not mounted on this machine")

    meetings = load_notsofar_meetings(
        dataset_dir=root,
        split="eval_small",
        meetings=["MTG_32000", "MTG_32003"],
        max_duration_sec=15.0,
    )
    by_id = {m.meeting_id: m for m in meetings}

    def mock_real_mtg_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body_text = request.read().decode("latin-1", errors="ignore")
        mtg = by_id["MTG_32003"] if "MTG_32003.wav" in body_text else by_id["MTG_32000"]

        words_payload = [
            {"word": wt.word, "start": wt.start_sec, "end": wt.end_sec}
            for seg in mtg.segments
            for wt in seg.word_timings
        ]
        diar_segs = [
            {"speaker": seg.speaker_id, "start_sec": seg.start_time, "end_sec": seg.end_time}
            for seg in mtg.segments
        ]

        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "parakeet-tdt-0.6b-v2", "object": "model", "type": "asr", "engine": "parakeet"},
                        {"id": "pyannote/speaker-diarization-3.1", "object": "model", "type": "diarization", "engine": "pyannote"},
                    ],
                },
            )
        if path == "/v1/audio/transcriptions":
            if 'name="stream"\r\n\r\ntrue' in body_text:
                sse = (
                    f"data: {json.dumps({'text': mtg.reference_text, 'words': words_payload})}\n\n"
                    "data: [DONE]\n\n"
                )
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode("utf-8"))
            return httpx.Response(
                200,
                json={"text": mtg.reference_text, "language": "en", "duration": mtg.duration_sec, "words": words_payload},
            )
        if path == "/v1/audio/diarization":
            return httpx.Response(
                200,
                json={
                    "num_speakers": mtg.expected_speakers,
                    "duration": mtg.duration_sec,
                    "segments": diar_segs,
                    "speakers": [
                        {"speaker": spk, "total_sec": 1.0, "segment_count": 1}
                        for spk in mtg.reference_speakers
                    ],
                },
            )
        return httpx.Response(404, json={"error": {"message": "Not found"}})

    transport = httpx.MockTransport(mock_real_mtg_handler)
    tester = LiveSTTTester()
    artifacts_dir = tmp_path / "real_notsofar_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ctx = TestContext(run_id=10, artifacts_dir=artifacts_dir, emit_stdout=False)

    res = tester.run(
        {
            "suites": ["accuracy_wer", "accuracy_streaming", "accuracy_diarization"],
            "dataset_split": "eval_small",
            "meetings": "MTG_32000,MTG_32003",
            "max_audio_duration_sec": 15.0,
            "streaming_protocol": "sse",
            "cache_path": str(tmp_path / "real_cache.sqlite3"),
            "_transport": transport,
        },
        {"name": "real-notsofar-target", "base_url": "http://mockserver/v1"},
        ctx,
    )
    assert res.status == "pass"
    assert len(res.cases) == 6  # 2 meetings x 3 suites
    assert res.metrics["wer"]["value"] == 0.0
    assert res.metrics["http_wer"]["value"] == 0.0
    assert res.metrics["streaming_wer"]["value"] == 0.0
    assert res.metrics["der"]["value"] == 0.0
    assert res.metrics["speaker_count_error"]["value"] == 0.0
    assert res.metrics["accuracy_meetings_evaluated"]["value"] == 2


def test_accuracy_wer_server_error_excluded_from_aggregates_and_not_cached(tmp_path: Path):
    """Verify HTTP 500 in accuracy_wer is recorded as status='error', excluded from aggregates, and not cached."""
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    cache_file = tmp_path / "wer_err_cache.sqlite3"

    request_counts = {"MTG_90001": 0, "MTG_90002": 0}
    # MTG_90001 has 14 reference words; change 1 word ("ready" -> "prepared") so WER = 1/14 = 0.0714
    mtg1_words = [
        {"word": "Hello", "start": 0.2, "end": 0.4},
        {"word": "Bob", "start": 0.4, "end": 0.6},
        {"word": "let's", "start": 0.6, "end": 0.8},
        {"word": "start", "start": 0.8, "end": 1.0},
        {"word": "the", "start": 1.0, "end": 1.1},
        {"word": "quarterly", "start": 1.1, "end": 1.25},
        {"word": "review", "start": 1.25, "end": 1.4},
        {"word": "Sounds", "start": 1.6, "end": 1.8},
        {"word": "good", "start": 1.8, "end": 2.0},
        {"word": "I", "start": 2.0, "end": 2.1},
        {"word": "have", "start": 2.1, "end": 2.3},
        {"word": "the", "start": 2.3, "end": 2.4},
        {"word": "metrics", "start": 2.4, "end": 2.6},
        {"word": "prepared", "start": 2.6, "end": 2.8},
    ]
    mtg1_text = " ".join(w["word"] for w in mtg1_words)

    def mock_http_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body_text = request.read().decode("latin-1", errors="ignore")
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "parakeet-tdt-0.6b-v3", "object": "model", "type": "asr", "engine": "parakeet"},
                    ],
                },
            )
        if path == "/v1/audio/transcriptions":
            if "MTG_90002.wav" in body_text:
                request_counts["MTG_90002"] += 1
                return httpx.Response(
                    500,
                    json={
                        "error": {
                            "message": "engine error during transcription: CUDA out of memory",
                            "type": "server_error",
                        }
                    },
                )
            request_counts["MTG_90001"] += 1
            return httpx.Response(
                200,
                json={
                    "text": mtg1_text,
                    "language": "en",
                    "duration": 3.0,
                    "words": mtg1_words,
                },
            )
        return httpx.Response(404, json={"error": {"message": "Not found"}})

    transport = httpx.MockTransport(mock_http_handler)
    tester = LiveSTTTester()

    artifacts_dir_1 = tmp_path / "wer_err_run1"
    artifacts_dir_1.mkdir(parents=True, exist_ok=True)
    ctx1 = TestContext(run_id=101, artifacts_dir=artifacts_dir_1, emit_stdout=False)

    cfg = {
        "suites": ["accuracy_wer"],
        "dataset_dir": str(ds_root),
        "dataset_split": "eval_small",
        "meetings": ["MTG_90001", "MTG_90002"],
        "cache_path": str(cache_file),
        "use_cache": True,
        "_transport": transport,
    }
    target = {"name": "mock-live-stt", "base_url": "http://mockserver/v1"}

    res1 = tester.run(cfg, target, ctx1)
    assert len(res1.cases) == 2

    case_ok = next(c for c in res1.cases if "MTG_90001" in c.case_id)
    case_err = next(c for c in res1.cases if "MTG_90002" in c.case_id)

    assert case_ok.status == "pass"
    assert case_err.status == "error"
    assert case_err.detail["http_status"] == 500
    assert "HTTP 500" in case_err.detail["eval_message"]
    assert "engine error during transcription: CUDA out of memory" in case_err.detail["eval_message"]
    assert case_err.detail["wer"] is None
    assert "wer" not in case_err.metrics
    assert "latency_ms" not in case_err.metrics
    assert "rtf" not in case_err.metrics

    log_text = ctx1.log_file.read_text(encoding="utf-8")
    assert "-> ERROR | HTTP 500: engine error during transcription: CUDA out of memory" in log_text

    # Aggregates must equal MTG_90001 only (1/14 = 0.0714), NOT include MTG_90002 as 100% WER
    expected_wer = round(1 / 14.0, 4)
    assert res1.metrics["wer"]["value"] == expected_wer
    assert res1.metrics["http_wer"]["value"] == expected_wer
    assert res1.metrics["http_wer_mean"]["value"] == expected_wer
    assert res1.metrics["wer_evaluated_cases"]["value"] == 1
    assert res1.metrics["accuracy_wer_evaluated_cases"]["value"] == 1
    assert res1.metrics["accuracy_wer_errored_cases"]["value"] == 1
    assert res1.metrics["errored_cases"]["value"] == 1
    assert res1.metrics["cases_errored"]["value"] == 1
    assert res1.metrics["cases_failed"]["value"] == 0

    # Latency and RTF aggregates must equal case_ok's latency and RTF only
    ok_lat = case_ok.metrics["latency_ms"]["value"]
    ok_rtf = case_ok.metrics["rtf"]["value"]
    assert res1.metrics["latency_mean_ms"]["value"] == ok_lat
    assert res1.metrics["accuracy_wer_latency_mean_ms"]["value"] == ok_lat
    assert res1.metrics["rtf_mean"]["value"] == ok_rtf
    assert res1.metrics["accuracy_wer_rtf_mean"]["value"] == ok_rtf
    assert "1 cases errored; aggregates exclude them" in (res1.summary or "")

    # Second run: MTG_90001 must hit cache, while MTG_90002 (500 error) must NOT be cached
    assert request_counts == {"MTG_90001": 1, "MTG_90002": 1}
    artifacts_dir_2 = tmp_path / "wer_err_run2"
    artifacts_dir_2.mkdir(parents=True, exist_ok=True)
    ctx2 = TestContext(run_id=102, artifacts_dir=artifacts_dir_2, emit_stdout=False)
    res2 = tester.run(cfg, target, ctx2)
    assert request_counts == {"MTG_90001": 1, "MTG_90002": 2}
    case2_ok = next(c for c in res2.cases if "MTG_90001" in c.case_id)
    case2_err = next(c for c in res2.cases if "MTG_90002" in c.case_id)
    assert case2_ok.detail["cache_hit"] is True
    assert case2_err.detail["cache_hit"] is False
    assert res2.metrics["accuracy_cache_hits"]["value"] == 1
    assert res2.metrics["accuracy_cache_misses"]["value"] == 1


def test_accuracy_streaming_and_diarization_server_errors_excluded_and_not_cached(tmp_path: Path):
    """Verify gRPC error, SSE error, and diarization HTTP 500 are recorded as status='error', excluded from totals, and not cached."""
    import grpc
    from backend.app.testers.live_stt.grpc_client import MockStreamingASRServicer, WordTimestamp

    ds_root = _create_synthetic_notsofar_split(tmp_path)
    cache_file = tmp_path / "stream_diar_err_cache.sqlite3"

    http_counts = {"sse_1": 0, "sse_2": 0, "diar_1": 0, "diar_2": 0}
    mtg1_words = [
        {"word": "Hello", "start": 0.2, "end": 0.4},
        {"word": "Bob", "start": 0.4, "end": 0.6},
        {"word": "let's", "start": 0.6, "end": 0.8},
        {"word": "start", "start": 0.8, "end": 1.0},
        {"word": "the", "start": 1.0, "end": 1.1},
        {"word": "quarterly", "start": 1.1, "end": 1.25},
        {"word": "review", "start": 1.25, "end": 1.4},
        {"word": "Sounds", "start": 1.6, "end": 1.8},
        {"word": "good", "start": 1.8, "end": 2.0},
        {"word": "I", "start": 2.0, "end": 2.1},
        {"word": "have", "start": 2.1, "end": 2.3},
        {"word": "the", "start": 2.3, "end": 2.4},
        {"word": "metrics", "start": 2.4, "end": 2.6},
        {"word": "ready", "start": 2.6, "end": 2.8},
    ]
    mtg1_text = " ".join(w["word"] for w in mtg1_words)
    mtg1_diar_segments = [
        {"speaker": "SPEAKER_00", "start_sec": 0.2, "end_sec": 1.4},
        {"speaker": "SPEAKER_01", "start_sec": 1.6, "end_sec": 2.8},
    ]

    def mock_http_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body_text = request.read().decode("latin-1", errors="ignore")
        is_mtg2 = "MTG_90002.wav" in body_text

        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "parakeet-tdt-0.6b-v2", "object": "model", "type": "asr", "engine": "parakeet"},
                        {"id": "pyannote/speaker-diarization-3.1", "object": "model", "type": "diarization", "engine": "pyannote"},
                    ],
                },
            )
        if path == "/v1/audio/transcriptions":
            if is_mtg2:
                http_counts["sse_2"] += 1
                err_sse = (
                    f"data: {json.dumps({'type': 'error', 'error': {'message': 'SSE stream CUDA OOM crash'}})}\n\n"
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=err_sse.encode("utf-8"),
                )
            http_counts["sse_1"] += 1
            ok_sse = (
                f"data: {json.dumps({'text': mtg1_text, 'words': mtg1_words})}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=ok_sse.encode("utf-8"),
            )
        if path == "/v1/audio/diarization":
            if is_mtg2:
                http_counts["diar_2"] += 1
                return httpx.Response(
                    500,
                    json={"error": {"message": "diarization worker crashed: CUDA OOM"}},
                )
            http_counts["diar_1"] += 1
            return httpx.Response(
                200,
                json={
                    "num_speakers": 2,
                    "duration": 3.0,
                    "segments": mtg1_diar_segments,
                    "speakers": [
                        {"speaker": "SPEAKER_00", "total_sec": 1.2, "segment_count": 1},
                        {"speaker": "SPEAKER_01", "total_sec": 1.2, "segment_count": 1},
                    ],
                },
            )
        return httpx.Response(404, json={"error": {"message": "Not found"}})

    class FailEverySecondCallServicer(MockStreamingASRServicer):
        async def Transcribe(self, request_iterator: Any, context: grpc.aio.ServicerContext) -> Any:  # noqa: N802
            # Call 1 is MTG_90001 (succeeds); subsequent calls are MTG_90002 (abort with INTERNAL)
            if self.total_calls >= 1:
                self.total_calls += 1
                try:
                    await request_iterator.__anext__()
                except Exception:
                    pass
                await context.abort(grpc.StatusCode.INTERNAL, "gRPC worker crash: CUDA out of memory")
                return
            async for ev in super().Transcribe(request_iterator, context):
                yield ev

    grpc_words = [
        WordTimestamp(text=w["word"], start_sec=w["start"], end_sec=w["end"])
        for w in mtg1_words
    ]
    servicer = FailEverySecondCallServicer(
        words_sequence=grpc_words,
        allowed_models={"parakeet-tdt-0.6b-v2"},
    )
    transport = httpx.MockTransport(mock_http_handler)

    with MockLiveSTTGRPCServer(servicer=servicer) as grpc_server:
        tester = LiveSTTTester()
        artifacts_dir_1 = tmp_path / "stream_diar_run1"
        artifacts_dir_1.mkdir(parents=True, exist_ok=True)
        ctx1 = TestContext(run_id=201, artifacts_dir=artifacts_dir_1, emit_stdout=False)

        cfg = {
            "suites": ["accuracy_streaming", "accuracy_diarization"],
            "dataset_dir": str(ds_root),
            "dataset_split": "eval_small",
            "meetings": ["MTG_90001", "MTG_90002"],
            "streaming_protocol": "both",
            "grpc_target": grpc_server.target,
            "cache_path": str(cache_file),
            "use_cache": True,
            "_transport": transport,
        }
        target = {"name": "mock-live-stt", "base_url": "http://mockserver/v1"}

        res1 = tester.run(cfg, target, ctx1)
        # 2 streaming protocols x 2 meetings + 1 diarization x 2 meetings = 6 cases
        assert len(res1.cases) == 6

        by_cid = {c.case_id: c for c in res1.cases}
        grpc_ok = by_cid["accuracy_streaming:parakeet-tdt-0.6b-v2:grpc:MTG_90001"]
        grpc_err = by_cid["accuracy_streaming:parakeet-tdt-0.6b-v2:grpc:MTG_90002"]
        sse_ok = by_cid["accuracy_streaming:parakeet-tdt-0.6b-v2:sse:MTG_90001"]
        sse_err = by_cid["accuracy_streaming:parakeet-tdt-0.6b-v2:sse:MTG_90002"]
        diar_ok = by_cid["accuracy_diarization:pyannote/speaker-diarization-3.1:MTG_90001"]
        diar_err = by_cid["accuracy_diarization:pyannote/speaker-diarization-3.1:MTG_90002"]

        assert grpc_ok.status == "pass"
        assert sse_ok.status == "pass"
        assert diar_ok.status == "pass"

        assert grpc_err.status == "error"
        assert "gRPC INTERNAL" in grpc_err.detail["eval_message"]
        assert "gRPC worker crash: CUDA out of memory" in grpc_err.detail["eval_message"]

        assert sse_err.status == "error"
        assert "SSE stream CUDA OOM crash" in sse_err.detail["eval_message"]

        assert diar_err.status == "error"
        assert "HTTP 500" in diar_err.detail["eval_message"]
        assert "diarization worker crashed: CUDA OOM" in diar_err.detail["eval_message"]

        # Aggregates cover only MTG_90001 (WER=0.0, DER=0.0)
        assert res1.metrics["streaming_wer"]["value"] == 0.0
        assert res1.metrics["der"]["value"] == 0.0
        assert res1.metrics["accuracy_streaming_errored_cases"]["value"] == 2
        assert res1.metrics["accuracy_streaming_evaluated_cases"]["value"] == 2
        assert res1.metrics["accuracy_diarization_errored_cases"]["value"] == 1
        assert res1.metrics["accuracy_diarization_evaluated_cases"]["value"] == 1
        assert res1.metrics["errored_cases"]["value"] == 3

        # Second run: MTG_90001 hits cache (3 hits), MTG_90002 is NOT cached (3 misses)
        artifacts_dir_2 = tmp_path / "stream_diar_run2"
        artifacts_dir_2.mkdir(parents=True, exist_ok=True)
        ctx2 = TestContext(run_id=202, artifacts_dir=artifacts_dir_2, emit_stdout=False)
        res2 = tester.run(cfg, target, ctx2)
        assert res2.metrics["accuracy_cache_hits"]["value"] == 3
        assert res2.metrics["accuracy_cache_misses"]["value"] == 3
        assert http_counts["sse_1"] == 1
        assert http_counts["sse_2"] == 2
        assert http_counts["diar_1"] == 1
        assert http_counts["diar_2"] == 2
        assert servicer.total_calls == 3  # 1 ok + 1 fail on run1, + 1 fail on run2 (ok was cached)




def _diarization_progress_transport(events: list[tuple[str, str]]) -> httpx.MockTransport:
    """Mock live-stt for diarization progress tests; records each request into ``events``."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body_text = request.read().decode("latin-1", errors="ignore")
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "parakeet-tdt-0.6b-v2", "object": "model", "type": "asr", "engine": "parakeet"},
                        {"id": "pyannote/speaker-diarization-3.1", "object": "model", "type": "diarization", "engine": "pyannote"},
                    ],
                },
            )
        meeting = "MTG_90002" if "MTG_90002.wav" in body_text else "MTG_90001"
        if path == "/v1/audio/diarization":
            events.append(("request", f"diarization:{meeting}"))
            return httpx.Response(
                200,
                json={
                    "num_speakers": 2,
                    "duration": 3.0,
                    "segments": [
                        {"speaker": "SPEAKER_00", "start_sec": 0.1, "end_sec": 1.4},
                        {"speaker": "SPEAKER_01", "start_sec": 1.6, "end_sec": 2.8},
                    ],
                    "speakers": [
                        {"speaker": "SPEAKER_00", "total_sec": 1.3, "segment_count": 1},
                        {"speaker": "SPEAKER_01", "total_sec": 1.2, "segment_count": 1},
                    ],
                },
            )
        if path == "/v1/audio/transcriptions":
            events.append(("request", f"transcription:{meeting}"))
            return httpx.Response(
                200,
                json={
                    "text": "hello bob lets start",
                    "language": "en",
                    "duration": 3.0,
                    "words": [{"word": "hello", "start": 0.2, "end": 0.4}],
                    "segments": [],
                },
            )
        return httpx.Response(404, json={"error": {"message": "Not found"}})

    return httpx.MockTransport(handler)


def test_accuracy_diarization_logs_progress_per_meeting(tmp_path: Path):
    """accuracy_diarization alone must diarize meeting by meeting, logging each one, instead of
    silently diarizing every meeting up front (looked stuck on 80 meetings)."""
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    events: list[tuple[str, str]] = []

    def on_event(ev: dict[str, Any]) -> None:
        if ev.get("type") == "log":
            events.append(("log", str(ev.get("message", ""))))

    ctx = TestContext(run_id=210, artifacts_dir=tmp_path / "diar_progress", emit_stdout=False, on_event=on_event)
    res = LiveSTTTester().run(
        {
            "suites": ["accuracy_diarization"],
            "dataset_dir": str(ds_root),
            "meetings": ["MTG_90001", "MTG_90002"],
            "use_cache": False,
            "_transport": _diarization_progress_transport(events),
        },
        {"name": "mock-live-stt", "base_url": "http://mockserver/v1"},
        ctx,
    )
    assert [c.case_id for c in res.cases] == [
        "accuracy_diarization:pyannote/speaker-diarization-3.1:MTG_90001",
        "accuracy_diarization:pyannote/speaker-diarization-3.1:MTG_90002",
    ]
    assert not any(kind == "log" and "Pre-computing diarization" in msg for kind, msg in events)

    # Order per meeting: "Running [i/N] ..." log, then its diarization request, then the "-> STATUS" log
    def idx(pred) -> int:
        return next(i for i, e in enumerate(events) if pred(e))

    for n, mtg in enumerate(("MTG_90001", "MTG_90002"), start=1):
        running = idx(lambda e: e[0] == "log" and e[1].startswith(f"Running [{n}/2] accuracy_diarization:") and mtg in e[1])
        request = idx(lambda e: e == ("request", f"diarization:{mtg}"))
        done = idx(lambda e: e[0] == "log" and f"accuracy_diarization:pyannote/speaker-diarization-3.1:{mtg} -> " in e[1])
        assert running < request < done
    assert sum(1 for e in events if e[0] == "request") == 2


def test_accuracy_wer_cpwer_precompute_logs_each_meeting_and_is_reused(tmp_path: Path):
    """With accuracy_wer + accuracy_diarization, the cpWER pre-compute logs every meeting and the
    diarization suite reuses those results instead of diarizing again."""
    ds_root = _create_synthetic_notsofar_split(tmp_path)
    events: list[tuple[str, str]] = []

    def on_event(ev: dict[str, Any]) -> None:
        if ev.get("type") == "log":
            events.append(("log", str(ev.get("message", ""))))

    ctx = TestContext(run_id=211, artifacts_dir=tmp_path / "cpwer_progress", emit_stdout=False, on_event=on_event)
    res = LiveSTTTester().run(
        {
            "suites": ["accuracy_wer", "accuracy_diarization"],
            "dataset_dir": str(ds_root),
            "meetings": ["MTG_90001", "MTG_90002"],
            "use_cache": False,
            "_transport": _diarization_progress_transport(events),
        },
        {"name": "mock-live-stt", "base_url": "http://mockserver/v1"},
        ctx,
    )
    logs = [msg for kind, msg in events if kind == "log"]
    assert any(m.startswith("Pre-computing diarization for cpWER") for m in logs)
    for n, mtg in enumerate(("MTG_90001", "MTG_90002"), start=1):
        assert any(f"[diarize {n}/2] pyannote/speaker-diarization-3.1:{mtg} -> HTTP 200, speakers=2, segments=2" in m for m in logs)
    # Each meeting diarized exactly once across pre-compute and the accuracy_diarization suite
    assert sorted(msg for kind, msg in events if kind == "request" and msg.startswith("diarization:")) == [
        "diarization:MTG_90001",
        "diarization:MTG_90002",
    ]
    assert len([c for c in res.cases if c.suite == "accuracy_diarization"]) == 2
