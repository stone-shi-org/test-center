"""Small built-in audio and text fixtures for the live-stt HTTP tester.

Generates deterministic 16kHz mono 16-bit PCM WAV fixtures with realistic
syllabic speech envelopes, formants, and word-level timestamps so tests
can run completely offline without external audio files while satisfying
live-stt's strict 16kHz mono 16-bit PCM WAV contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import wave
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class WordTimestamp:
    """Reference word timestamp entry matching live-stt's words JSON contract."""
    text: str
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_sec": round(float(self.start_sec), 3),
            "end_sec": round(float(self.end_sec), 3),
        }


@dataclass(frozen=True)
class AudioFixture:
    """Built-in 16kHz mono 16-bit PCM WAV audio fixture."""
    id: str
    name: str
    description: str
    duration_sec: float
    sample_rate: int
    channels: int
    sample_width: int
    wav_bytes: bytes
    expected_speakers: int
    reference_text: str
    words: tuple[WordTimestamp, ...]

    def words_payload(self) -> list[dict[str, Any]]:
        """Return list of word timestamp dicts for /v1/audio/diarization 'words' field."""
        return [w.to_dict() for w in self.words]


@dataclass(frozen=True)
class TTSFixture:
    """Built-in text prompt fixture for TTS /v1/audio/speech tests."""
    id: str
    name: str
    text: str
    stream: bool
    description: str


def _pack_pcm16_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Pack float32 [-1.0, 1.0] samples into a 16-bit mono PCM RIFF WAV byte buffer."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def synthesize_speech_like_wav(
    duration_sec: float,
    words: Sequence[WordTimestamp] | tuple[WordTimestamp, ...],
    sample_rate: int = 16000,
    speaker_f0_by_index: Sequence[float] | None = None,
) -> bytes:
    """Synthesize a deterministic 16kHz mono 16-bit PCM WAV with speech-like formants.

    Each word interval receives a voiced harmonic + formant burst with smooth attack/release
    envelopes, separated by natural low-energy silence gaps.
    """
    n_samples = max(1, int(round(duration_sec * sample_rate)))
    signal = np.zeros(n_samples, dtype=np.float32)
    t_full = np.arange(n_samples, dtype=np.float32) / float(sample_rate)

    for idx, w in enumerate(words):
        start_idx = max(0, min(n_samples, int(round(w.start_sec * sample_rate))))
        end_idx = max(start_idx, min(n_samples, int(round(w.end_sec * sample_rate))))
        seg_len = end_idx - start_idx
        if seg_len <= 0:
            continue

        f0 = (
            float(speaker_f0_by_index[idx])
            if speaker_f0_by_index and idx < len(speaker_f0_by_index)
            else (140.0 + (idx % 3) * 18.0)
        )
        t_seg = t_full[start_idx:end_idx]

        # Fundamental + vowel-like formant harmonics (F1 ~500Hz, F2 ~1500Hz, F3 ~2500Hz)
        carrier = (
            0.45 * np.sin(2.0 * math.pi * f0 * t_seg)
            + 0.25 * np.sin(2.0 * math.pi * (f0 * 3.0) * t_seg)
            + 0.18 * np.sin(2.0 * math.pi * (f0 * 7.0) * t_seg)
            + 0.12 * np.sin(2.0 * math.pi * (f0 * 11.0) * t_seg)
        )

        # Smooth raised-cosine window envelope for syllable cadence
        local_phase = np.linspace(0.0, math.pi, seg_len, dtype=np.float32)
        envelope = np.sqrt(np.maximum(0.0, np.sin(local_phase)))
        signal[start_idx:end_idx] = 0.4 * carrier * envelope

    return _pack_pcm16_wav(signal, sample_rate=sample_rate)


def _build_default_audio_fixtures() -> dict[str, AudioFixture]:
    short_words = (
        WordTimestamp("hello", 0.10, 0.45),
        WordTimestamp("world", 0.55, 0.95),
        WordTimestamp("testing", 1.05, 1.40),
    )
    short_wav = synthesize_speech_like_wav(
        duration_sec=1.5,
        words=short_words,
        sample_rate=16000,
    )

    medium_words = (
        WordTimestamp("the", 0.12, 0.30),
        WordTimestamp("quick", 0.35, 0.62),
        WordTimestamp("brown", 0.68, 0.98),
        WordTimestamp("fox", 1.04, 1.32),
        WordTimestamp("jumps", 1.38, 1.70),
        WordTimestamp("over", 1.76, 2.04),
        WordTimestamp("the", 2.10, 2.28),
        WordTimestamp("lazy", 2.34, 2.60),
        WordTimestamp("dog", 2.65, 2.90),
    )
    medium_wav = synthesize_speech_like_wav(
        duration_sec=3.0,
        words=medium_words,
        sample_rate=16000,
    )

    dialog_words = (
        WordTimestamp("good", 0.15, 0.45),
        WordTimestamp("morning", 0.50, 0.95),
        WordTimestamp("everyone", 1.00, 1.55),
        WordTimestamp("hello", 2.10, 2.55),
        WordTimestamp("thanks", 2.65, 3.10),
        WordTimestamp("joining", 3.20, 3.75),
    )
    dialog_f0s = [125.0, 130.0, 128.0, 215.0, 220.0, 212.0]
    dialog_wav = synthesize_speech_like_wav(
        duration_sec=4.0,
        words=dialog_words,
        sample_rate=16000,
        speaker_f0_by_index=dialog_f0s,
    )

    return {
        "short_speech": AudioFixture(
            id="short_speech",
            name="Short Utterance (1.5s)",
            description="1.5s 16kHz mono 16-bit PCM WAV with 3 spoken words",
            duration_sec=1.5,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            wav_bytes=short_wav,
            expected_speakers=1,
            reference_text="hello world testing",
            words=short_words,
        ),
        "medium_speech": AudioFixture(
            id="medium_speech",
            name="Medium Phrase (3.0s)",
            description="3.0s 16kHz mono 16-bit PCM WAV with 9 spoken words",
            duration_sec=3.0,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            wav_bytes=medium_wav,
            expected_speakers=1,
            reference_text="the quick brown fox jumps over the lazy dog",
            words=medium_words,
        ),
        "two_speaker_dialog": AudioFixture(
            id="two_speaker_dialog",
            name="Two-Speaker Turn Dialog (4.0s)",
            description="4.0s 16kHz mono 16-bit PCM WAV with 2 distinct speaker pitch turns and word timestamps",
            duration_sec=4.0,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            wav_bytes=dialog_wav,
            expected_speakers=2,
            reference_text="good morning everyone hello thanks joining",
            words=dialog_words,
        ),
    }


BUILTIN_AUDIO_FIXTURES: dict[str, AudioFixture] = _build_default_audio_fixtures()

BUILTIN_TTS_FIXTURES: dict[str, TTSFixture] = {
    "tts_batch": TTSFixture(
        id="tts_batch",
        name="Single Sentence TTS Synthesis",
        text="Hello, this is a functional and latency test of the text to speech endpoint.",
        stream=False,
        description="Non-streaming speech synthesis of a single clear sentence",
    ),
    "tts_stream": TTSFixture(
        id="tts_stream",
        name="Multi-Sentence Streaming TTS Synthesis",
        text="Welcome to the audio test center. This second sentence verifies chunked streaming synthesis.",
        stream=True,
        description="Sentence-chunked streaming speech synthesis",
    ),
}


def get_audio_fixture(fixture_id: str = "short_speech") -> AudioFixture:
    """Retrieve a built-in audio fixture by ID, falling back to 'short_speech'."""
    return BUILTIN_AUDIO_FIXTURES.get(fixture_id, BUILTIN_AUDIO_FIXTURES["short_speech"])


def get_builtin_audio_fixtures() -> list[AudioFixture]:
    """Return all built-in audio fixtures."""
    return list(BUILTIN_AUDIO_FIXTURES.values())


def get_builtin_tts_fixtures() -> list[TTSFixture]:
    """Return all built-in TTS text fixtures."""
    return list(BUILTIN_TTS_FIXTURES.values())
