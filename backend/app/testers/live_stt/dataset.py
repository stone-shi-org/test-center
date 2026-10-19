"""NOTSOFAR-1 dataset loader, subset/split selector, audio reader, and text normalizer for scoring.

Layout under ``test-data/NOTSOFAR-1`` (mounted read-only in Docker, gitignored):
- ``eval_set/240629.1_eval_small_with_GT/MTG/MTG_<id>/`` (80 meetings; default ``eval_small`` split)
- ``eval_set/240825.1_eval_full_with_GT/MTG/MTG_<id>/`` (129 meetings; ``eval_full`` split)
- ``dev_set/240825.1_dev1/MTG/MTG_<id>/`` (36 meetings; ``dev`` split)
- ``train_set/240825.1_train/MTG/MTG_<id>/`` (72 meetings; ``train`` split)

Each ``MTG_<id>/`` directory contains:
- ``gt_meeting_metadata.json``: ``meeting_id``, ``Topic``, ``MeetingDurationSec``, ``NumParticipants``, ``ParticipantAliases``
- ``gt_transcription.json``: list of ``{speaker_id, start_time, end_time, text, word_timing, ct_wav_file_name}``
- ``devices.json``: list of recording devices; single-channel far-field ("meeting mix") devices have
  ``is_mc: false, is_close_talk: false`` (e.g. ``sc_meetup_0/ch0.wav``, ``sc_rockfall_0/ch0.wav``).
"""

from __future__ import annotations

import array
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import wave

from backend.app.config import BASE_DIR, get_settings


DEFAULT_NOTSOFAR_SPLIT = "eval_small"

NOTSOFAR_SPLITS: dict[str, str] = {
    "eval_small": "eval_set/240629.1_eval_small_with_GT",
    "eval": "eval_set/240629.1_eval_small_with_GT",
    "default": "eval_set/240629.1_eval_small_with_GT",
    "240629.1_eval_small_with_GT": "eval_set/240629.1_eval_small_with_GT",
    "eval_full": "eval_set/240825.1_eval_full_with_GT",
    "240825.1_eval_full_with_GT": "eval_set/240825.1_eval_full_with_GT",
    "dev": "dev_set/240825.1_dev1",
    "dev1": "dev_set/240825.1_dev1",
    "dev_set": "dev_set/240825.1_dev1",
    "240825.1_dev1": "dev_set/240825.1_dev1",
    "train": "train_set/240825.1_train",
    "train_set": "train_set/240825.1_train",
    "240825.1_train": "train_set/240825.1_train",
}

TARGET_SAMPLE_RATE_HZ = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

# NOTSOFAR-1 ground-truth annotation tags: <ST/>, <PName>, </PName>, <FILL/>, <FILLlaugh/>, <BA/>, <FL/>, <ISSUE/>, <UNKNOWN/>
_ANNOTATION_TAG_RE = re.compile(r"<[^>]*>")
_APOSTROPHE_CHARS_RE = re.compile(r"[\u2018\u2019\u02bc\u2032`]")
_NON_WORD_RE = re.compile(r"[^\w']+|_")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text_for_scoring(text: str | None) -> str:
    """Normalize reference or hypothesis transcript text for WER / cpWER scoring.

    1. Strips NOTSOFAR-1 XML/angle-bracket annotation tags (e.g. ``<ST/>``, ``<PName>``, ``<FILL/>``).
    2. Normalizes curly apostrophes to ASCII ``'`` and lowercases text.
    3. Replaces punctuation and underscores with spaces while preserving intra-word contractions (e.g. ``don't``).
    4. Strips leading/trailing apostrophes on tokens and collapses whitespace.
    """
    if not text:
        return ""
    cleaned = _ANNOTATION_TAG_RE.sub(" ", str(text))
    cleaned = _APOSTROPHE_CHARS_RE.sub("'", cleaned.lower())
    cleaned = _NON_WORD_RE.sub(" ", cleaned)
    tokens: list[str] = []
    for raw_tok in cleaned.split():
        tok = raw_tok.strip("'")
        if tok:
            tokens.append(tok)
    return " ".join(tokens)


def tokenize_for_scoring(text: str | None) -> list[str]:
    """Tokenize transcript text into normalized lowercase words for WER / cpWER scoring."""
    norm = normalize_text_for_scoring(text)
    return norm.split() if norm else []


@dataclass(frozen=True)
class NotsofarWordTiming:
    """Single ground-truth word timestamp entry from ``gt_transcription.json``."""

    word: str
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
        }


@dataclass(frozen=True)
class NotsofarSegment:
    """Ground-truth speaker segment from ``gt_transcription.json``."""

    speaker_id: str
    start_time: float
    end_time: float
    text: str
    normalized_text: str
    words: tuple[str, ...]
    word_timings: tuple[NotsofarWordTiming, ...] = ()
    ct_wav_file_name: str | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "text": self.text,
            "normalized_text": self.normalized_text,
            "word_count": len(self.words),
        }


@dataclass(frozen=True)
class NotsofarDevice:
    """Recording device entry from ``devices.json``."""

    device_name: str
    is_mc: bool
    is_close_talk: bool
    channels_num: int
    wav_file_names: str

    @property
    def is_single_channel_far_field(self) -> bool:
        return (not self.is_mc) and (not self.is_close_talk) and self.channels_num == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_name": self.device_name,
            "is_mc": self.is_mc,
            "is_close_talk": self.is_close_talk,
            "channels_num": self.channels_num,
            "wav_file_names": self.wav_file_names,
        }


@dataclass(frozen=True)
class NotsofarMeeting:
    """Parsed NOTSOFAR-1 meeting with audio path, ground-truth transcript, and speaker labels."""

    meeting_id: str
    split: str
    meeting_dir: Path
    audio_path: Path
    device_name: str
    topic: str
    duration_sec: float
    num_speakers: int
    participant_aliases: tuple[str, ...]
    room: str = ""
    hashtags: str = ""
    segments: tuple[NotsofarSegment, ...] = ()
    devices: tuple[NotsofarDevice, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_text(self) -> str:
        """Raw ground-truth transcript joined across segments in chronological order."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    @property
    def normalized_reference_text(self) -> str:
        """Normalized ground-truth transcript for WER scoring."""
        return " ".join(self.reference_words)

    @property
    def reference_words(self) -> list[str]:
        """Normalized ground-truth word sequence across the meeting."""
        words: list[str] = []
        for seg in self.segments:
            words.extend(seg.words)
        return words

    @property
    def device_alias(self) -> str:
        """Alias for device_name."""
        return self.device_name

    @property
    def expected_speakers(self) -> int:
        """Expected speaker count for the meeting."""
        return self.num_speakers

    @property
    def speaker_reference_words(self) -> dict[str, list[str]]:
        """Map each reference speaker_id to their chronologically concatenated normalized words (for cpWER)."""
        by_spk: dict[str, list[str]] = defaultdict(list)
        for seg in self.segments:
            if seg.words:
                by_spk[seg.speaker_id].extend(seg.words)
        return dict(by_spk)

    @property
    def speaker_reference_texts(self) -> dict[str, str]:
        """Map each reference speaker_id to their normalized reference text (for cpWER)."""
        return {spk: " ".join(words) for spk, words in self.speaker_reference_words.items()}

    @property
    def reference_speakers(self) -> list[str]:
        """Sorted list of distinct speaker IDs present in the reference segments (or participant_aliases)."""
        from_segs = sorted({seg.speaker_id for seg in self.segments if seg.speaker_id})
        if from_segs:
            return from_segs
        return sorted(set(self.participant_aliases))

    @property
    def gt_segments_dicts(self) -> list[dict[str, Any]]:
        """Ground-truth segments as dicts compatible with DER / speaker timeline scorers."""
        return [
            {
                "speaker": seg.speaker_id,
                "speaker_id": seg.speaker_id,
                "start_sec": seg.start_time,
                "end_sec": seg.end_time,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
            }
            for seg in self.segments
        ]

    @property
    def wav_bytes(self) -> bytes:
        """Normalized 16kHz mono 16-bit PCM WAV bytes clipped to self.duration_sec."""
        wav_data, _ = self.read_wav_bytes(max_duration_sec=self.duration_sec)
        return wav_data

    def read_wav_bytes(self, max_duration_sec: float | None = None) -> tuple[bytes, float]:
        """Load meeting WAV audio, normalizing to 16kHz mono 16-bit PCM and optionally trimming to max_duration_sec."""
        return load_and_normalize_wav_bytes(
            self.audio_path,
            max_duration_sec=max_duration_sec if max_duration_sec is not None else self.duration_sec,
        )

    def slice_to_duration(self, max_duration_sec: float) -> NotsofarMeeting:
        """Return a copy of this meeting with ground-truth segments clipped to ``[0, max_duration_sec]``."""
        if max_duration_sec <= 0 or max_duration_sec >= self.duration_sec:
            return self
        clipped_dur = min(self.duration_sec, float(max_duration_sec))
        clipped_segments: list[NotsofarSegment] = []
        for seg in self.segments:
            if seg.start_time >= clipped_dur:
                continue
            new_end = min(seg.end_time, clipped_dur)
            if new_end <= seg.start_time:
                continue
            if seg.end_time <= clipped_dur or not seg.word_timings:
                clipped_segments.append(
                    NotsofarSegment(
                        speaker_id=seg.speaker_id,
                        start_time=seg.start_time,
                        end_time=new_end,
                        text=seg.text,
                        normalized_text=seg.normalized_text,
                        words=seg.words,
                        word_timings=seg.word_timings,
                        ct_wav_file_name=seg.ct_wav_file_name,
                    )
                )
            else:
                kept_timings = tuple(
                    wt for wt in seg.word_timings if wt.start_sec < clipped_dur
                )
                kept_raw_text = " ".join(wt.word for wt in kept_timings)
                kept_words = tuple(tokenize_for_scoring(kept_raw_text))
                clipped_segments.append(
                    NotsofarSegment(
                        speaker_id=seg.speaker_id,
                        start_time=seg.start_time,
                        end_time=new_end,
                        text=kept_raw_text,
                        normalized_text=" ".join(kept_words),
                        words=kept_words,
                        word_timings=kept_timings,
                        ct_wav_file_name=seg.ct_wav_file_name,
                    )
                )

        active_speakers = {s.speaker_id for s in clipped_segments if s.speaker_id}
        num_spk = len(active_speakers) if active_speakers else self.num_speakers
        return NotsofarMeeting(
            meeting_id=self.meeting_id,
            split=self.split,
            meeting_dir=self.meeting_dir,
            audio_path=self.audio_path,
            device_name=self.device_name,
            topic=self.topic,
            duration_sec=round(clipped_dur, 4),
            num_speakers=num_spk,
            participant_aliases=self.participant_aliases,
            room=self.room,
            hashtags=self.hashtags,
            segments=tuple(clipped_segments),
            devices=self.devices,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "split": self.split,
            "meeting_dir": str(self.meeting_dir),
            "audio_path": str(self.audio_path),
            "device_name": self.device_name,
            "topic": self.topic,
            "duration_sec": round(self.duration_sec, 4),
            "num_speakers": self.num_speakers,
            "participant_aliases": list(self.participant_aliases),
            "room": self.room,
            "segment_count": len(self.segments),
            "reference_word_count": len(self.reference_words),
        }


def _decode_pcm16(raw_frames: bytes) -> array.array[int]:
    samples = array.array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _encode_pcm16(samples: array.array[int]) -> bytes:
    if sys.byteorder != "little":
        copied = array.array("h", samples)
        copied.byteswap()
        return copied.tobytes()
    return samples.tobytes()


def _mix_to_mono(samples: array.array[int], nchannels: int) -> array.array[int]:
    if nchannels <= 1:
        return samples
    n = len(samples) // nchannels
    mono = array.array("h", [0]) * n
    for i in range(n):
        total = 0
        base = i * nchannels
        for c in range(nchannels):
            total += samples[base + c]
        mono[i] = int(total // nchannels)
    return mono


def _resample_linear(samples: array.array[int], src_rate: int, dst_rate: int) -> array.array[int]:
    if src_rate == dst_rate or not samples:
        return samples
    n = len(samples)
    n_out = max(1, int(round(n * dst_rate / float(src_rate))))
    out = array.array("h", [0]) * n_out
    for i in range(n_out):
        pos = i * src_rate / float(dst_rate)
        j = int(pos)
        frac = pos - j
        if j + 1 < n:
            out[i] = int(round(samples[j] + (samples[j + 1] - samples[j]) * frac))
        else:
            out[i] = samples[min(j, n - 1)]
    return out


def load_and_normalize_wav_bytes(
    wav_source: str | Path | bytes,
    *,
    max_duration_sec: float | None = None,
) -> tuple[bytes, float]:
    """Load WAV audio from disk or bytes, ensuring 16kHz mono 16-bit PCM format and optional duration clipping."""
    if isinstance(wav_source, (str, Path)):
        raw_bytes = Path(wav_source).read_bytes()
    else:
        raw_bytes = bytes(wav_source)

    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sw = int(wf.getsampwidth())
        total_frames = int(wf.getnframes())
        if max_duration_sec is not None and max_duration_sec > 0:
            frames_to_read = min(total_frames, max(1, int(round(max_duration_sec * sr))))
        else:
            frames_to_read = total_frames
        frames = wf.readframes(frames_to_read)

    # Fast path: already 16kHz mono 16-bit PCM and no trimming needed
    if (
        sr == TARGET_SAMPLE_RATE_HZ
        and ch == TARGET_CHANNELS
        and sw == TARGET_SAMPLE_WIDTH
        and frames_to_read == total_frames
    ):
        duration_sec = round(total_frames / float(sr), 4) if sr > 0 else 0.0
        return raw_bytes, duration_sec

    if sw != TARGET_SAMPLE_WIDTH:
        raise ValueError(f"Unsupported WAV sample width {sw * 8}-bit (expected 16-bit PCM)")

    if sr == TARGET_SAMPLE_RATE_HZ and ch == TARGET_CHANNELS:
        pcm_frames = frames
    else:
        samples = _decode_pcm16(frames)
        samples = _mix_to_mono(samples, ch)
        samples = _resample_linear(samples, sr, TARGET_SAMPLE_RATE_HZ)
        pcm_frames = _encode_pcm16(samples)

    n_out_samples = len(pcm_frames) // (TARGET_CHANNELS * TARGET_SAMPLE_WIDTH)
    duration_sec = round(n_out_samples / float(TARGET_SAMPLE_RATE_HZ), 4)

    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as dst:
        dst.setnchannels(TARGET_CHANNELS)
        dst.setsampwidth(TARGET_SAMPLE_WIDTH)
        dst.setframerate(TARGET_SAMPLE_RATE_HZ)
        dst.writeframes(pcm_frames)
    return out_buf.getvalue(), duration_sec


def resolve_notsofar_root(dataset_dir: str | Path | None = None) -> Path:
    """Locate the NOTSOFAR-1 root directory on disk.

    Precedence:
    1. Explicit ``dataset_dir`` argument
    2. ``NOTSOFAR_DIR`` environment variable
    3. ``get_settings().TEST_DATA_DIR / "NOTSOFAR-1"``
    4. ``BASE_DIR / "test-data" / "NOTSOFAR-1"``
    5. Parent repository ``test-data/NOTSOFAR-1`` when running inside ``.worktrees/<branch>``
    """
    if dataset_dir is not None and str(dataset_dir).strip():
        p = Path(str(dataset_dir).strip())
        if p.exists():
            return p.resolve()
        # Allow relative to TEST_DATA_DIR
        rel = get_settings().TEST_DATA_DIR / str(dataset_dir).strip()
        if rel.exists():
            return rel.resolve()
        return p

    env_dir = os.environ.get("NOTSOFAR_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p.resolve()

    settings = get_settings()
    candidates = [
        settings.TEST_DATA_DIR / "NOTSOFAR-1",
        BASE_DIR / "test-data" / "NOTSOFAR-1",
        BASE_DIR.parent.parent / "test-data" / "NOTSOFAR-1",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    return (settings.TEST_DATA_DIR / "NOTSOFAR-1").resolve()


def resolve_notsofar_split_dir(
    dataset_dir: str | Path | None = None,
    split: str | None = None,
) -> tuple[Path, str]:
    """Resolve the ``MTG`` container directory and canonical split name for NOTSOFAR-1.

    Supports:
    - Split aliases: ``"eval_small"`` (default), ``"eval_full"``, ``"dev"``, ``"train"``
    - Relative split paths: ``"eval_set/240629.1_eval_small_with_GT"``
    - Direct paths where ``dataset_dir`` itself is a split dir (contains ``MTG/``) or ``MTG`` dir (contains ``MTG_*``).
    """
    root = resolve_notsofar_root(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"NOTSOFAR-1 dataset directory not found at '{root}'. "
            "Mount or copy NOTSOFAR-1 under test-data/NOTSOFAR-1 or set dataset_dir."
        )

    raw_split = (split or DEFAULT_NOTSOFAR_SPLIT).strip()
    mapped_rel = NOTSOFAR_SPLITS.get(raw_split.lower(), NOTSOFAR_SPLITS.get(raw_split, raw_split))

    # Case 1: root/<mapped_rel>/MTG exists
    cand_split_dir = root / mapped_rel
    if (cand_split_dir / "MTG").is_dir():
        return (cand_split_dir / "MTG").resolve(), mapped_rel
    if cand_split_dir.is_dir() and any(
        d.is_dir() and d.name.startswith("MTG_") for d in cand_split_dir.iterdir()
    ):
        return cand_split_dir.resolve(), mapped_rel

    # Case 2: root itself has MTG/ (user pointed dataset_dir directly at a split folder)
    if (root / "MTG").is_dir():
        return (root / "MTG").resolve(), root.name

    # Case 3: root itself contains MTG_* directories
    if root.is_dir() and any(d.is_dir() and d.name.startswith("MTG_") for d in root.iterdir()):
        return root.resolve(), root.name

    raise FileNotFoundError(
        f"NOTSOFAR-1 split '{raw_split}' (resolved='{mapped_rel}') not found under '{root}'"
    )


def discover_notsofar_meetings(
    dataset_dir: str | Path | None = None,
    split: str | None = None,
) -> list[str]:
    """Return sorted list of ``MTG_*`` meeting IDs available in the specified NOTSOFAR-1 split."""
    mtg_root, _ = resolve_notsofar_split_dir(dataset_dir=dataset_dir, split=split)
    return sorted(
        entry.name
        for entry in mtg_root.iterdir()
        if entry.is_dir() and (entry / "gt_transcription.json").is_file()
    )


def select_notsofar_audio_device(
    meeting_dir: Path,
    devices: Sequence[dict[str, Any]],
    preferred_device: str | None = None,
) -> tuple[Path, str]:
    """Select a single-channel far-field ('meeting mix') WAV recording for a NOTSOFAR-1 meeting.

    Single-channel non-close-talk devices have ``is_mc=False, is_close_talk=False``
    (e.g. ``meetup_0 -> sc_meetup_0/ch0.wav``, ``rockfall_0 -> sc_rockfall_0/ch0.wav``).
    Sorted deterministically by ``device_name`` unless ``preferred_device`` matches.
    """
    sc_devices = sorted(
        (
            d
            for d in devices
            if isinstance(d, dict)
            and not d.get("is_mc")
            and not d.get("is_close_talk")
            and d.get("wav_file_names")
        ),
        key=lambda d: str(d.get("device_name", "")),
    )

    if preferred_device and str(preferred_device).strip():
        pref = str(preferred_device).strip().lower()
        for d in devices:
            if not isinstance(d, dict) or not d.get("wav_file_names"):
                continue
            dev_name = str(d.get("device_name", "")).lower()
            wav_rel = str(d.get("wav_file_names", "")).replace("\\", "/").split(",")[0].strip()
            if pref in (dev_name, wav_rel.lower(), f"sc_{dev_name}"):
                wav_path = meeting_dir / wav_rel
                if wav_path.is_file():
                    return wav_path.resolve(), str(d.get("device_name", pref))

    for d in sc_devices:
        wav_rel = str(d["wav_file_names"]).replace("\\", "/").split(",")[0].strip()
        wav_path = meeting_dir / wav_rel
        if wav_path.is_file():
            return wav_path.resolve(), str(d.get("device_name", "sc_0"))

    # Fallback: search for any single-channel sc_*/ch0.wav or *.wav in meeting_dir
    for fallback_wav in sorted(meeting_dir.glob("sc_*/ch0.wav")) + sorted(meeting_dir.glob("*.wav")):
        if fallback_wav.is_file():
            return fallback_wav.resolve(), fallback_wav.parent.name

    raise FileNotFoundError(f"No single-channel far-field WAV file found in '{meeting_dir}'")


def parse_meeting_selection(
    meetings: Sequence[str] | str | None,
    available_ids: Sequence[str],
    max_meetings: int | None = None,
) -> list[str]:
    """Resolve user-configured meeting filter (list or comma-separated string) and meeting count limit."""
    avail_list = list(available_ids)
    avail_set = set(avail_list)

    selected: list[str]
    if meetings is None:
        selected = avail_list
    elif isinstance(meetings, str):
        parts = [m.strip() for m in meetings.split(",") if m.strip()]
        selected = parts if parts else avail_list
    else:
        parts = [str(m).strip() for m in meetings if str(m).strip()]
        selected = parts if parts else avail_list

    unknown = [m for m in selected if m not in avail_set]
    if unknown:
        raise ValueError(
            f"Unknown NOTSOFAR-1 meeting ID(s): {', '.join(unknown)}. "
            f"Available ({len(avail_list)}): {', '.join(avail_list[:10])}"
            + ("..." if len(avail_list) > 10 else "")
        )

    # Deduplicate preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for m in selected:
        if m not in seen:
            deduped.append(m)
            seen.add(m)

    if max_meetings is not None and int(max_meetings) > 0:
        deduped = deduped[: int(max_meetings)]

    return deduped


def load_notsofar_meeting(
    meeting_id: str,
    *,
    dataset_dir: str | Path | None = None,
    split: str | None = None,
    preferred_device: str | None = None,
    device_alias: str | None = None,
    prefer_close_talk: bool = False,
    max_duration_sec: float | None = None,
) -> NotsofarMeeting:
    """Load a single NOTSOFAR-1 meeting's metadata, ground-truth transcript, speaker labels, and audio path."""
    mtg_root, canonical_split = resolve_notsofar_split_dir(dataset_dir=dataset_dir, split=split)
    meeting_dir = mtg_root / meeting_id
    if not meeting_dir.is_dir():
        raise FileNotFoundError(f"Meeting directory '{meeting_dir}' does not exist")

    meta_path = meeting_dir / "gt_meeting_metadata.json"
    gt_path = meeting_dir / "gt_transcription.json"
    dev_path = meeting_dir / "devices.json"

    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    raw_segments: list[dict[str, Any]] = []
    if gt_path.is_file():
        raw_segments = json.loads(gt_path.read_text(encoding="utf-8"))

    raw_devices: list[dict[str, Any]] = []
    if dev_path.is_file():
        raw_devices = json.loads(dev_path.read_text(encoding="utf-8"))

    effective_device = preferred_device or device_alias
    if not effective_device and prefer_close_talk:
        for d in raw_devices:
            if isinstance(d, dict) and bool(d.get("is_close_talk", False)):
                effective_device = str(d.get("device_name") or d.get("wav_file_names") or "")
                break

    audio_path, device_name = select_notsofar_audio_device(
        meeting_dir=meeting_dir,
        devices=raw_devices,
        preferred_device=effective_device,
    )

    sorted_raw_segs = sorted(
        (s for s in raw_segments if isinstance(s, dict)),
        key=lambda s: (float(s.get("start_time", 0.0)), float(s.get("end_time", 0.0))),
    )

    parsed_segments: list[NotsofarSegment] = []
    for s in sorted_raw_segs:
        spk = str(s.get("speaker_id") or s.get("speaker") or "SPEAKER_00").strip()
        start_t = float(s.get("start_time", s.get("start", 0.0)) or 0.0)
        end_t = float(s.get("end_time", s.get("end", start_t)) or start_t)
        raw_txt = str(s.get("text", "") or "")
        norm_txt = normalize_text_for_scoring(raw_txt)
        words_tuple = tuple(norm_txt.split()) if norm_txt else ()

        timings: list[NotsofarWordTiming] = []
        raw_wt = s.get("word_timing")
        if isinstance(raw_wt, list):
            for entry in raw_wt:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    timings.append(
                        NotsofarWordTiming(
                            word=str(entry[0]),
                            start_sec=float(entry[1]),
                            end_sec=float(entry[2]),
                        )
                    )
                elif isinstance(entry, dict) and ("word" in entry or "text" in entry):
                    timings.append(
                        NotsofarWordTiming(
                            word=str(entry.get("word") or entry.get("text") or ""),
                            start_sec=float(entry.get("start_sec", entry.get("start", 0.0))),
                            end_sec=float(entry.get("end_sec", entry.get("end", 0.0))),
                        )
                    )

        parsed_segments.append(
            NotsofarSegment(
                speaker_id=spk,
                start_time=start_t,
                end_time=end_t,
                text=raw_txt,
                normalized_text=norm_txt,
                words=words_tuple,
                word_timings=tuple(timings),
                ct_wav_file_name=s.get("ct_wav_file_name"),
            )
        )

    parsed_devices = tuple(
        NotsofarDevice(
            device_name=str(d.get("device_name", "")),
            is_mc=bool(d.get("is_mc", False)),
            is_close_talk=bool(d.get("is_close_talk", False)),
            channels_num=int(d.get("channels_num", 1) or 1),
            wav_file_names=str(d.get("wav_file_names", "")),
        )
        for d in raw_devices
        if isinstance(d, dict)
    )

    aliases_raw = metadata.get("ParticipantAliases")
    aliases = (
        tuple(str(a) for a in aliases_raw if str(a).strip())
        if isinstance(aliases_raw, list)
        else tuple(sorted({seg.speaker_id for seg in parsed_segments if seg.speaker_id}))
    )

    num_speakers = int(metadata.get("NumParticipants") or len(aliases) or 1)
    duration_sec = float(metadata.get("MeetingDurationSec") or 0.0)
    if duration_sec <= 0.0:
        try:
            with wave.open(str(audio_path), "rb") as wf:
                sr = wf.getframerate()
                if sr > 0:
                    duration_sec = wf.getnframes() / float(sr)
        except Exception:
            if parsed_segments:
                duration_sec = max(seg.end_time for seg in parsed_segments)

    meeting = NotsofarMeeting(
        meeting_id=str(metadata.get("meeting_id") or meeting_id),
        split=canonical_split,
        meeting_dir=meeting_dir,
        audio_path=audio_path,
        device_name=device_name,
        topic=str(metadata.get("Topic") or ""),
        duration_sec=round(duration_sec, 4),
        num_speakers=num_speakers,
        participant_aliases=aliases,
        room=str(metadata.get("Room") or ""),
        hashtags=str(metadata.get("Hashtags") or ""),
        segments=tuple(parsed_segments),
        devices=parsed_devices,
        metadata=metadata,
    )

    if max_duration_sec is not None and max_duration_sec > 0:
        return meeting.slice_to_duration(float(max_duration_sec))
    return meeting


def load_notsofar_meetings(
    *,
    dataset_dir: str | Path | None = None,
    split: str | None = None,
    meetings: Sequence[str] | str | None = None,
    max_meetings: int | None = None,
    preferred_device: str | None = None,
    device_alias: str | None = None,
    prefer_close_talk: bool = False,
    max_duration_sec: float | None = None,
) -> list[NotsofarMeeting]:
    """Discover, filter, and load NOTSOFAR-1 meetings for evaluation."""
    available_ids = discover_notsofar_meetings(dataset_dir=dataset_dir, split=split)
    selected_ids = parse_meeting_selection(
        meetings=meetings,
        available_ids=available_ids,
        max_meetings=max_meetings,
    )
    return [
        load_notsofar_meeting(
            mid,
            dataset_dir=dataset_dir,
            split=split,
            preferred_device=preferred_device or device_alias,
            prefer_close_talk=prefer_close_talk,
            max_duration_sec=max_duration_sec,
        )
        for mid in selected_ids
    ]


NotsofarAudioDevice = NotsofarDevice
NotsofarTranscriptSegment = NotsofarSegment



# ---------------------------------------------------------------------------
# Real-speech diarization fixture (excerpt of a NOTSOFAR-1 meeting)
# ---------------------------------------------------------------------------

DEFAULT_DIARIZATION_CLIP_SEC = 30.0


def _words_for_clip(meeting: NotsofarMeeting) -> list[Any]:
    """Word timestamps for a clipped meeting: GT word timings when present, otherwise words
    spread evenly across their segment (good enough for the words_alignment contract check)."""
    from backend.app.testers.live_stt.fixtures import WordTimestamp

    words: list[WordTimestamp] = []
    for seg in meeting.segments:
        if seg.word_timings:
            for wt in seg.word_timings:
                if wt.word.strip() and wt.end_sec > wt.start_sec:
                    words.append(WordTimestamp(wt.word.strip(), wt.start_sec, wt.end_sec))
            continue
        raw = [w for w in seg.text.split() if w.strip()]
        if not raw or seg.end_time <= seg.start_time:
            continue
        step = (seg.end_time - seg.start_time) / len(raw)
        for i, w in enumerate(raw):
            start = seg.start_time + i * step
            words.append(WordTimestamp(w, start, start + step * 0.9))
    words.sort(key=lambda w: (w.start_sec, w.end_sec))
    return words


_DIALOG_FIXTURE_CACHE: dict[tuple[str, str, float, int], Any] = {}


def build_notsofar_dialog_fixture(
    *,
    dataset_dir: str | Path | None = None,
    split: str | None = None,
    clip_sec: float = DEFAULT_DIARIZATION_CLIP_SEC,
    min_speakers: int = 2,
    max_candidates: int = 20,
) -> Any | None:
    """Build a real-speech multi-speaker AudioFixture from the first ``clip_sec`` seconds of a
    NOTSOFAR-1 meeting with at least ``min_speakers`` speakers talking in that window.

    Meetings are tried in sorted ID order, so the choice is deterministic for a given dataset.
    Returns None if the dataset is not available or no meeting qualifies. Found fixtures are cached.
    """
    from backend.app.testers.live_stt.fixtures import AudioFixture

    cache_key = (str(dataset_dir or ""), str(split or DEFAULT_NOTSOFAR_SPLIT), float(clip_sec), int(min_speakers))
    if cache_key in _DIALOG_FIXTURE_CACHE:
        return _DIALOG_FIXTURE_CACHE[cache_key]

    fixture: AudioFixture | None = None
    try:
        meeting_ids = sorted(discover_notsofar_meetings(dataset_dir=dataset_dir, split=split))
    except (FileNotFoundError, ValueError, OSError):
        meeting_ids = []

    for mid in meeting_ids[: max(1, int(max_candidates))]:
        try:
            meeting = load_notsofar_meeting(
                mid, dataset_dir=dataset_dir, split=split, max_duration_sec=clip_sec
            )
        except (FileNotFoundError, ValueError, OSError):
            continue
        speakers = {s.speaker_id for s in meeting.segments if s.speaker_id}
        if len(speakers) < min_speakers:
            continue
        words = _words_for_clip(meeting)
        if not words:
            continue
        try:
            wav_bytes, duration = meeting.read_wav_bytes(max_duration_sec=meeting.duration_sec)
        except (FileNotFoundError, ValueError, OSError, wave.Error):
            continue
        fixture = AudioFixture(
            id=f"notsofar:{meeting.meeting_id}:0-{int(round(duration))}s",
            name=f"NOTSOFAR-1 {meeting.meeting_id} excerpt ({duration:.1f}s, {len(speakers)} speakers)",
            description=(
                f"First {duration:.1f}s of NOTSOFAR-1 meeting {meeting.meeting_id} "
                f"({meeting.split}, device {meeting.device_name}): real multi-speaker speech"
            ),
            duration_sec=round(float(duration), 3),
            sample_rate=16000,
            channels=1,
            sample_width=2,
            wav_bytes=wav_bytes,
            expected_speakers=len(speakers),
            reference_text=meeting.normalized_reference_text,
            words=tuple(words),
        )
        break

    if fixture is not None:
        # Only cache hits, so a dataset mounted later is still picked up.
        _DIALOG_FIXTURE_CACHE[cache_key] = fixture
    return fixture
