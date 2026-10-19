"""NOTSOFAR-1 accuracy evaluation metrics (WER, cpWER, DER, speaker-count error), SQLite result cache, and HTML report."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence

import numpy as np

from backend.app.testers.live_stt.dataset import (
    normalize_text_for_scoring,
    tokenize_for_scoring,
)

CACHE_SCHEMA_VERSION = "live_stt_notsofar_cache_v1"


@dataclass
class WERResult:
    """Word Error Rate (WER) evaluation result for a single transcript pair."""

    wer: float
    wer_pct: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    edit_distance: int
    ref_word_count: int
    hyp_word_count: int
    ref_normalized: str
    hyp_normalized: str
    ops: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_ops: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "wer": self.wer,
            "wer_pct": self.wer_pct,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "edit_distance": self.edit_distance,
            "ref_word_count": self.ref_word_count,
            "hyp_word_count": self.hyp_word_count,
            "ref_normalized": self.ref_normalized,
            "hyp_normalized": self.hyp_normalized,
        }
        if include_ops:
            payload["ops"] = self.ops
        return payload


@dataclass
class CPWERResult:
    """Concatenated minimum-permutation Word Error Rate (cpWER) result across speakers."""

    cpwer: float
    cpwer_pct: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    edit_distance: int
    ref_word_count: int
    hyp_word_count: int
    ref_speaker_count: int
    hyp_speaker_count: int
    speaker_mapping: dict[str, str | None] = field(default_factory=dict)
    per_speaker: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpwer": self.cpwer,
            "cpwer_pct": self.cpwer_pct,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hits": self.hits,
            "edit_distance": self.edit_distance,
            "ref_word_count": self.ref_word_count,
            "hyp_word_count": self.hyp_word_count,
            "ref_speaker_count": self.ref_speaker_count,
            "hyp_speaker_count": self.hyp_speaker_count,
            "speaker_mapping": self.speaker_mapping,
            "per_speaker": self.per_speaker,
        }


@dataclass
class DERResult:
    """Diarization Error Rate (DER) and speaker-count error result for a meeting."""

    der: float
    der_pct: float
    missed_speech_sec: float
    false_alarm_sec: float
    speaker_confusion_sec: float
    total_ref_speech_sec: float
    scored_duration_sec: float
    missed_rate: float
    false_alarm_rate: float
    confusion_rate: float
    speaker_error_rate: float
    speaker_count_error: int
    speaker_count_delta: int
    speaker_count_exact: bool
    expected_speakers: int
    predicted_speakers: int
    collar_sec: float = 0.0
    frame_sec: float = 0.05
    speaker_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "der": self.der,
            "der_pct": self.der_pct,
            "missed_speech_sec": self.missed_speech_sec,
            "false_alarm_sec": self.false_alarm_sec,
            "speaker_confusion_sec": self.speaker_confusion_sec,
            "total_ref_speech_sec": self.total_ref_speech_sec,
            "scored_duration_sec": self.scored_duration_sec,
            "missed_rate": self.missed_rate,
            "false_alarm_rate": self.false_alarm_rate,
            "confusion_rate": self.confusion_rate,
            "speaker_error_rate": self.speaker_error_rate,
            "speaker_count_error": self.speaker_count_error,
            "speaker_count_delta": self.speaker_count_delta,
            "speaker_count_exact": self.speaker_count_exact,
            "expected_speakers": self.expected_speakers,
            "predicted_speakers": self.predicted_speakers,
            "collar_sec": self.collar_sec,
            "frame_sec": self.frame_sec,
            "speaker_diagnostics": self.speaker_diagnostics,
        }


def _as_tokens(text_or_tokens: str | Sequence[str]) -> tuple[list[str], str]:
    if isinstance(text_or_tokens, str):
        norm = normalize_text_for_scoring(text_or_tokens)
        tokens = norm.split() if norm else []
        return tokens, norm
    raw_joined = " ".join(str(t) for t in text_or_tokens if str(t).strip())
    norm = normalize_text_for_scoring(raw_joined)
    tokens = norm.split() if norm else []
    return tokens, norm


def _word_edit_counts(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> tuple[int, int, int, int]:
    """Compute (substitutions, deletions, insertions, hits) between two token lists."""
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0, 0, m, 0
    if m == 0:
        return 0, n, 0, 0

    # DP table of edit distances
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        r_tok = ref_tokens[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            if r_tok == hyp_tokens[j - 1]:
                row[j] = prev_row[j - 1]
            else:
                sub = prev_row[j - 1] + 1
                dele = prev_row[j] + 1
                ins = row[j - 1] + 1
                row[j] = sub if (sub <= dele and sub <= ins) else (dele if dele <= ins else ins)

    # Backtrace to count S, D, I, H
    i, j = n, m
    subs = dels = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return subs, dels, ins, hits


def compute_wer(
    reference: str | Sequence[str],
    hypothesis: str | Sequence[str],
    include_ops: bool = False,
) -> WERResult:
    """Compute Word Error Rate (WER) using exact word-level Levenshtein alignment."""
    ref_tokens, ref_norm = _as_tokens(reference)
    hyp_tokens, hyp_norm = _as_tokens(hypothesis)

    n = len(ref_tokens)
    m = len(hyp_tokens)

    if n == 0 and m == 0:
        return WERResult(
            wer=0.0,
            wer_pct=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            hits=0,
            edit_distance=0,
            ref_word_count=0,
            hyp_word_count=0,
            ref_normalized=ref_norm,
            hyp_normalized=hyp_norm,
            ops=[],
        )
    if n == 0:
        ops = [{"op": "insert", "ref": "", "hyp": w} for w in hyp_tokens] if include_ops else []
        return WERResult(
            wer=1.0,
            wer_pct=100.0,
            substitutions=0,
            deletions=0,
            insertions=m,
            hits=0,
            edit_distance=m,
            ref_word_count=0,
            hyp_word_count=m,
            ref_normalized=ref_norm,
            hyp_normalized=hyp_norm,
            ops=ops,
        )

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        r_tok = ref_tokens[i - 1]
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            if r_tok == hyp_tokens[j - 1]:
                row[j] = prev_row[j - 1]
            else:
                sub = prev_row[j - 1] + 1
                dele = prev_row[j] + 1
                ins = row[j - 1] + 1
                row[j] = sub if (sub <= dele and sub <= ins) else (dele if dele <= ins else ins)

    i, j = n, m
    subs = dels = ins = hits = 0
    rev_ops: list[dict[str, Any]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            if include_ops:
                rev_ops.append({"op": "equal", "ref": ref_tokens[i - 1], "hyp": hyp_tokens[j - 1]})
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            if include_ops:
                rev_ops.append({"op": "replace", "ref": ref_tokens[i - 1], "hyp": hyp_tokens[j - 1]})
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            if include_ops:
                rev_ops.append({"op": "delete", "ref": ref_tokens[i - 1], "hyp": ""})
            i -= 1
        else:
            ins += 1
            if include_ops:
                rev_ops.append({"op": "insert", "ref": "", "hyp": hyp_tokens[j - 1]})
            j -= 1

    edit_dist = subs + dels + ins
    wer_val = round(edit_dist / float(n), 4)
    wer_pct = round(wer_val * 100.0, 2)
    ops = list(reversed(rev_ops)) if include_ops else []

    return WERResult(
        wer=wer_val,
        wer_pct=wer_pct,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        hits=hits,
        edit_distance=edit_dist,
        ref_word_count=n,
        hyp_word_count=m,
        ref_normalized=ref_norm,
        hyp_normalized=hyp_norm,
        ops=ops,
    )


def _solve_min_weight_bipartite_matching(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Solve exact minimum-weight bipartite assignment (Hungarian / Kuhn-Munkres algorithm).

    Accepts a 2D square or rectangular numpy array of non-negative costs.
    Returns list of (row_idx, col_idx) pairs.
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return []

    transposed = False
    if n_rows > n_cols:
        cost = cost_matrix.T.astype(np.float64)
        n_rows, n_cols = cost.shape
        transposed = True
    else:
        cost = cost_matrix.astype(np.float64)

    # 1-indexed potentials u, v and matching p, way
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            r = int(p[j] - 1)
            c = int(j - 1)
            pairs.append((c, r) if transposed else (r, c))
    pairs.sort(key=lambda rc: rc[0])
    return pairs


def align_words_to_segments(
    words: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align timestamped ASR words onto diarization speaker segments.

    Each word is assigned to the speaker segment with greatest temporal overlap,
    or to the nearest segment in time if there is no direct overlap.
    Returns a copy of `segments` with populated `"words"` and `"text"` fields.
    """
    enriched: list[dict[str, Any]] = []
    for seg in segments:
        s_start = float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)
        s_end = float(seg.get("end_sec", seg.get("end", s_start)) or s_start)
        enriched.append({
            "speaker": str(seg.get("speaker", "SPEAKER_00")),
            "start_sec": s_start,
            "end_sec": s_end,
            "text": str(seg.get("text", "") or ""),
            "words": list(seg.get("words", [])) if isinstance(seg.get("words"), list) else [],
        })

    if not enriched or not words:
        return enriched

    # Clear pre-existing words if we're explicitly aligning fresh ASR words
    has_existing_text = any(bool(s["text"].strip()) for s in enriched)
    if not has_existing_text:
        for s in enriched:
            s["words"] = []

        for w in words:
            w_text = str(w.get("word", w.get("text", "")) or "").strip()
            if not w_text:
                continue
            w_start = float(w.get("start_sec", w.get("start", 0.0)) or 0.0)
            w_end = float(w.get("end_sec", w.get("end", w_start)) or w_start)
            if w_end < w_start:
                w_end = w_start
            w_mid = 0.5 * (w_start + w_end)

            best_idx = 0
            best_overlap = -1.0
            best_dist = float("inf")

            for idx, seg in enumerate(enriched):
                s_start = seg["start_sec"]
                s_end = seg["end_sec"]
                overlap = max(0.0, min(w_end, s_end) - max(w_start, s_start))
                if overlap > best_overlap and overlap > 0.0:
                    best_overlap = overlap
                    best_idx = idx
                elif best_overlap <= 0.0:
                    if w_mid < s_start:
                        dist = s_start - w_mid
                    elif w_mid > s_end:
                        dist = w_mid - s_end
                    else:
                        dist = 0.0
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx

            enriched[best_idx]["words"].append({
                "text": w_text,
                "start_sec": w_start,
                "end_sec": w_end,
            })

        for seg in enriched:
            if seg["words"]:
                seg["text"] = " ".join(w["text"] for w in seg["words"])

    return enriched


def extract_speaker_transcripts_from_segments(
    segments: Sequence[Mapping[str, Any]],
    words: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Build per-speaker concatenated transcripts from diarization segments (and optional ASR words)."""
    aligned = align_words_to_segments(words or [], segments)
    by_speaker: dict[str, list[str]] = {}
    for seg in aligned:
        spk = str(seg.get("speaker", "SPEAKER_00") or "SPEAKER_00").strip()
        txt = str(seg.get("text", "") or "").strip()
        if not txt and isinstance(seg.get("words"), list):
            txt = " ".join(
                str(w.get("word", w.get("text", "")) or "").strip()
                for w in seg["words"]
                if isinstance(w, Mapping)
            ).strip()
        if txt:
            by_speaker.setdefault(spk, []).append(txt)
    return {spk: " ".join(parts) for spk, parts in sorted(by_speaker.items())}


def compute_cpwer(
    ref_speaker_texts: Mapping[str, str | Sequence[str]],
    hyp_speaker_texts: Mapping[str, str | Sequence[str]],
) -> CPWERResult:
    """Compute Concatenated Minimum-Permutation Word Error Rate (cpWER).

    Finds the 1-to-1 permutation mapping between reference speakers and hypothesis
    speakers that minimizes the total word edit distance across all speakers.
    Unmatched reference speakers incur deletion costs; unmatched hypothesis speakers
    incur insertion costs.
    """
    ref_items: list[tuple[str, list[str]]] = []
    for spk, text in sorted(ref_speaker_texts.items()):
        tokens, _ = _as_tokens(text)
        if tokens:
            ref_items.append((str(spk), tokens))

    hyp_items: list[tuple[str, list[str]]] = []
    for spk, text in sorted(hyp_speaker_texts.items()):
        tokens, _ = _as_tokens(text)
        if tokens:
            hyp_items.append((str(spk), tokens))

    m_ref = len(ref_items)
    n_hyp = len(hyp_items)
    total_ref_words = sum(len(toks) for _, toks in ref_items)
    total_hyp_words = sum(len(toks) for _, toks in hyp_items)

    if m_ref == 0 and n_hyp == 0:
        return CPWERResult(
            cpwer=0.0,
            cpwer_pct=0.0,
            substitutions=0,
            deletions=0,
            insertions=0,
            hits=0,
            edit_distance=0,
            ref_word_count=0,
            hyp_word_count=0,
            ref_speaker_count=0,
            hyp_speaker_count=0,
        )

    k = max(m_ref, n_hyp)
    cost_matrix = np.zeros((k, k), dtype=np.float64)
    counts_grid: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    for i in range(k):
        for j in range(k):
            if i < m_ref and j < n_hyp:
                subs, dels, ins, hits = _word_edit_counts(ref_items[i][1], hyp_items[j][1])
                counts_grid[(i, j)] = (subs, dels, ins, hits)
                cost_matrix[i, j] = float(subs + dels + ins)
            elif i < m_ref and j >= n_hyp:
                dels = len(ref_items[i][1])
                counts_grid[(i, j)] = (0, dels, 0, 0)
                cost_matrix[i, j] = float(dels)
            elif i >= m_ref and j < n_hyp:
                ins = len(hyp_items[j][1])
                counts_grid[(i, j)] = (0, 0, ins, 0)
                cost_matrix[i, j] = float(ins)
            else:
                counts_grid[(i, j)] = (0, 0, 0, 0)
                cost_matrix[i, j] = 0.0

    pairs = _solve_min_weight_bipartite_matching(cost_matrix)

    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_hits = 0
    speaker_mapping: dict[str, str | None] = {}
    per_speaker: list[dict[str, Any]] = []

    for r_idx, h_idx in pairs:
        subs, dels, ins, hits = counts_grid[(r_idx, h_idx)]
        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_hits += hits

        ref_spk = ref_items[r_idx][0] if r_idx < m_ref else None
        hyp_spk = hyp_items[h_idx][0] if h_idx < n_hyp else None
        ref_len = len(ref_items[r_idx][1]) if r_idx < m_ref else 0
        hyp_len = len(hyp_items[h_idx][1]) if h_idx < n_hyp else 0

        if hyp_spk is not None:
            speaker_mapping[hyp_spk] = ref_spk

        if ref_spk is not None or hyp_spk is not None:
            pair_ed = subs + dels + ins
            pair_wer = round(pair_ed / float(ref_len), 4) if ref_len > 0 else (1.0 if hyp_len > 0 else 0.0)
            per_speaker.append({
                "ref_speaker": ref_spk,
                "hyp_speaker": hyp_spk,
                "ref_words": ref_len,
                "hyp_words": hyp_len,
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "hits": hits,
                "edit_distance": pair_ed,
                "wer": pair_wer,
            })

    total_edit = total_subs + total_dels + total_ins
    cpwer_val = round(total_edit / float(total_ref_words), 4) if total_ref_words > 0 else (1.0 if total_hyp_words > 0 else 0.0)
    cpwer_pct = round(cpwer_val * 100.0, 2)

    return CPWERResult(
        cpwer=cpwer_val,
        cpwer_pct=cpwer_pct,
        substitutions=total_subs,
        deletions=total_dels,
        insertions=total_ins,
        hits=total_hits,
        edit_distance=total_edit,
        ref_word_count=total_ref_words,
        hyp_word_count=total_hyp_words,
        ref_speaker_count=m_ref,
        hyp_speaker_count=n_hyp,
        speaker_mapping=speaker_mapping,
        per_speaker=per_speaker,
    )


def _segments_to_timeline_runs(
    segments: Sequence[Mapping[str, Any]],
    duration_sec: float,
    frame_sec: float = 0.1,
) -> list[dict[str, Any]]:
    """Convert speaker segments into contiguous frame-discretized runs for timeline rendering."""
    n_frames = max(1, int(round(duration_sec / frame_sec)))
    labels = [""] * n_frames
    for seg in segments:
        spk = str(seg.get("speaker", "") or "").strip()
        if not spk:
            continue
        start_f = max(0, int(float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0) / frame_sec))
        end_f = min(n_frames, int(float(seg.get("end_sec", seg.get("end", 0.0)) or 0.0) / frame_sec))
        for idx in range(start_f, end_f):
            labels[idx] = spk

    runs: list[dict[str, Any]] = []
    if not labels:
        return runs
    cur = labels[0]
    run_start = 0
    for idx in range(1, len(labels)):
        if labels[idx] != cur:
            runs.append({
                "speaker": cur,
                "start": round(run_start * frame_sec, 2),
                "end": round(idx * frame_sec, 2),
            })
            cur = labels[idx]
            run_start = idx
    runs.append({
        "speaker": cur,
        "start": round(run_start * frame_sec, 2),
        "end": round(len(labels) * frame_sec, 2),
    })
    return runs


def compute_der_and_speaker_error(
    gt_segments: Sequence[Mapping[str, Any]],
    pred_segments: Sequence[Mapping[str, Any]],
    duration_sec: float,
    expected_speakers: int | None = None,
    predicted_speakers: int | None = None,
    frame_sec: float = 0.05,
    collar_sec: float = 0.0,
) -> DERResult:
    """Compute Diarization Error Rate (DER) and speaker-count error on a meeting.

    Uses standard NIST Rich Transcription frame-level multi-speaker scoring:
      - Optimal 1-to-1 assignment between reference speakers and hypothesis speakers
        maximizing total overlap duration (via Kuhn-Munkres / Hungarian algorithm).
      - Per-frame decomposition into Missed Speech, False Alarm Speech, and Speaker Confusion.
      - Optional forgiveness collar (`collar_sec`) around reference segment boundaries.
    """
    max_seg_end = 0.0
    for s in list(gt_segments) + list(pred_segments):
        end_v = float(s.get("end_sec", s.get("end", 0.0)) or 0.0)
        if end_v > max_seg_end:
            max_seg_end = end_v

    effective_duration = max(float(duration_sec or 0.0), max_seg_end, frame_sec)
    n_frames = max(1, int(round(effective_duration / frame_sec)))

    ref_speakers_sorted = sorted({
        str(s.get("speaker", "")).strip()
        for s in gt_segments
        if str(s.get("speaker", "")).strip()
    })
    hyp_speakers_sorted = sorted({
        str(s.get("speaker", "")).strip()
        for s in pred_segments
        if str(s.get("speaker", "")).strip()
    })

    exp_spk = int(expected_speakers) if expected_speakers is not None else len(ref_speakers_sorted)
    pred_spk = int(predicted_speakers) if predicted_speakers is not None else len(hyp_speakers_sorted)

    ref_spk_to_idx = {spk: i for i, spk in enumerate(ref_speakers_sorted)}
    hyp_spk_to_idx = {spk: i for i, spk in enumerate(hyp_speakers_sorted)}

    m_ref = len(ref_speakers_sorted)
    n_hyp = len(hyp_speakers_sorted)

    # Boolean activity matrices: [n_frames, num_speakers]
    ref_active = np.zeros((n_frames, m_ref), dtype=bool)
    hyp_active = np.zeros((n_frames, n_hyp), dtype=bool)

    collar_mask = np.zeros(n_frames, dtype=bool)
    collar_frames = int(round(collar_sec / frame_sec)) if collar_sec > 0 else 0

    for seg in gt_segments:
        spk = str(seg.get("speaker", "")).strip()
        if spk not in ref_spk_to_idx:
            continue
        r_idx = ref_spk_to_idx[spk]
        start_t = float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)
        end_t = float(seg.get("end_sec", seg.get("end", start_t)) or start_t)
        s_f = max(0, min(n_frames, int(round(start_t / frame_sec))))
        e_f = max(s_f, min(n_frames, int(round(end_t / frame_sec))))
        if e_f > s_f:
            ref_active[s_f:e_f, r_idx] = True
        if collar_frames > 0:
            c_s1 = max(0, s_f - collar_frames)
            c_e1 = min(n_frames, s_f + collar_frames)
            c_s2 = max(0, e_f - collar_frames)
            c_e2 = min(n_frames, e_f + collar_frames)
            collar_mask[c_s1:c_e1] = True
            collar_mask[c_s2:c_e2] = True

    for seg in pred_segments:
        spk = str(seg.get("speaker", "")).strip()
        if spk not in hyp_spk_to_idx:
            continue
        h_idx = hyp_spk_to_idx[spk]
        start_t = float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)
        end_t = float(seg.get("end_sec", seg.get("end", start_t)) or start_t)
        s_f = max(0, min(n_frames, int(round(start_t / frame_sec))))
        e_f = max(s_f, min(n_frames, int(round(end_t / frame_sec))))
        if e_f > s_f:
            hyp_active[s_f:e_f, h_idx] = True

    eval_mask = ~collar_mask
    ref_eval = ref_active[eval_mask]
    hyp_eval = hyp_active[eval_mask]

    # Overlap matrix [m_ref, n_hyp] in evaluated frames
    overlap_frames = np.zeros((m_ref, n_hyp), dtype=np.int64)
    if m_ref > 0 and n_hyp > 0 and ref_eval.shape[0] > 0:
        overlap_frames = ref_eval.astype(np.int64).T @ hyp_eval.astype(np.int64)

    # Optimal 1-to-1 speaker mapping via Hungarian algorithm on (-overlap)
    ref_to_hyp_idx: dict[int, int] = {}
    hyp_to_ref_map: dict[str, str | None] = {h_spk: None for h_spk in hyp_speakers_sorted}
    if m_ref > 0 and n_hyp > 0:
        k = max(m_ref, n_hyp)
        max_ov = int(overlap_frames.max()) if overlap_frames.size > 0 else 0
        cost_mat = np.full((k, k), float(max_ov + 1), dtype=np.float64)
        for i in range(m_ref):
            for j in range(n_hyp):
                cost_mat[i, j] = float(max_ov + 1 - int(overlap_frames[i, j]))
        pairs = _solve_min_weight_bipartite_matching(cost_mat)
        for r_i, h_j in pairs:
            if r_i < m_ref and h_j < n_hyp and overlap_frames[r_i, h_j] > 0:
                ref_to_hyp_idx[r_i] = h_j
                hyp_to_ref_map[hyp_speakers_sorted[h_j]] = ref_speakers_sorted[r_i]

    # Compute frame-by-frame N_ref, N_hyp, N_correct
    n_ref_per_frame = ref_eval.sum(axis=1) if m_ref > 0 else np.zeros(ref_eval.shape[0], dtype=np.int64)
    n_hyp_per_frame = hyp_eval.sum(axis=1) if n_hyp > 0 else np.zeros(hyp_eval.shape[0], dtype=np.int64)
    n_correct_per_frame = np.zeros(ref_eval.shape[0], dtype=np.int64)

    for r_i, h_j in ref_to_hyp_idx.items():
        both_active = ref_eval[:, r_i] & hyp_eval[:, h_j]
        n_correct_per_frame += both_active.astype(np.int64)

    missed_frames = int(np.maximum(0, n_ref_per_frame - n_hyp_per_frame).sum())
    fa_frames = int(np.maximum(0, n_hyp_per_frame - n_ref_per_frame).sum())
    conf_frames = int((np.minimum(n_ref_per_frame, n_hyp_per_frame) - n_correct_per_frame).sum())
    total_ref_frames = int(n_ref_per_frame.sum())

    missed_sec = round(missed_frames * frame_sec, 4)
    fa_sec = round(fa_frames * frame_sec, 4)
    conf_sec = round(conf_frames * frame_sec, 4)
    total_ref_sec = round(total_ref_frames * frame_sec, 4)
    scored_dur_sec = round(int(eval_mask.sum()) * frame_sec, 4)

    if total_ref_frames > 0:
        der_val = round((missed_frames + fa_frames + conf_frames) / float(total_ref_frames), 4)
        missed_rate = round(missed_frames / float(total_ref_frames), 4)
        fa_rate = round(fa_frames / float(total_ref_frames), 4)
        conf_rate = round(conf_frames / float(total_ref_frames), 4)
    else:
        der_val = 0.0 if fa_frames == 0 else 1.0
        missed_rate = 0.0
        fa_rate = 0.0 if fa_frames == 0 else 1.0
        conf_rate = 0.0

    spk_count_err = abs(pred_spk - exp_spk)
    spk_count_delta = pred_spk - exp_spk
    spk_count_exact = pred_spk == exp_spk
    spk_err_rate = round(spk_count_err / float(exp_spk), 4) if exp_spk > 0 else 0.0

    # Per-speaker precision / recall / F1 diagnostics
    per_speaker_metrics: list[dict[str, Any]] = []
    ref_to_hyp_name = {
        ref_speakers_sorted[r_i]: hyp_speakers_sorted[h_j]
        for r_i, h_j in ref_to_hyp_idx.items()
    }
    for r_i, r_spk in enumerate(ref_speakers_sorted):
        r_frames = int(ref_eval[:, r_i].sum())
        h_j = ref_to_hyp_idx.get(r_i)
        if h_j is not None:
            h_spk = hyp_speakers_sorted[h_j]
            h_frames = int(hyp_eval[:, h_j].sum())
            tp_frames = int(overlap_frames[r_i, h_j])
        else:
            h_spk = None
            h_frames = 0
            tp_frames = 0
        recall = round(tp_frames / float(r_frames), 4) if r_frames > 0 else 0.0
        precision = round(tp_frames / float(h_frames), 4) if h_frames > 0 else 0.0
        f1 = (
            round(2.0 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
        per_speaker_metrics.append({
            "ref_speaker": r_spk,
            "hyp_speaker": h_spk,
            "ref_sec": round(r_frames * frame_sec, 2),
            "hyp_sec": round(h_frames * frame_sec, 2),
            "overlap_sec": round(tp_frames * frame_sec, 2),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })

    timeline_frame_sec = max(frame_sec, 0.1)
    diagnostics = {
        "speaker_map": hyp_to_ref_map,
        "ref_to_hyp_map": ref_to_hyp_name,
        "per_speaker_metrics": per_speaker_metrics,
        "ref_runs": _segments_to_timeline_runs(gt_segments, effective_duration, frame_sec=timeline_frame_sec),
        "hyp_runs": _segments_to_timeline_runs(pred_segments, effective_duration, frame_sec=timeline_frame_sec),
        "duration": round(effective_duration, 2),
    }

    return DERResult(
        der=der_val,
        der_pct=round(der_val * 100.0, 2),
        missed_speech_sec=missed_sec,
        false_alarm_sec=fa_sec,
        speaker_confusion_sec=conf_sec,
        total_ref_speech_sec=total_ref_sec,
        scored_duration_sec=scored_dur_sec,
        missed_rate=missed_rate,
        false_alarm_rate=fa_rate,
        confusion_rate=conf_rate,
        speaker_error_rate=spk_err_rate,
        speaker_count_error=spk_count_err,
        speaker_count_delta=spk_count_delta,
        speaker_count_exact=spk_count_exact,
        expected_speakers=exp_spk,
        predicted_speakers=pred_spk,
        collar_sec=collar_sec,
        frame_sec=frame_sec,
        speaker_diagnostics=diagnostics,
    )


class AccuracyResultCache:
    """SQLite-backed result cache for NOTSOFAR-1 accuracy evaluations.

    Ensures repeated runs of unchanged target/model/meeting/config are fast and cheap.
    Errors and failed transport calls are never cached.
    """

    def __init__(
        self,
        cache_path: str | Path,
        enabled: bool = True,
        skip_read: bool = False,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.enabled = bool(enabled)
        self.skip_read = bool(skip_read)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.cache_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accuracy_eval_cache (
                    cache_key TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    suite TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    meeting_id TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def compute_key(key_payload: Mapping[str, Any]) -> tuple[str, str]:
        """Return (sha256_hex_key, canonical_json_str) for a cache key dictionary."""
        enriched = dict(key_payload)
        enriched.setdefault("schema_version", CACHE_SCHEMA_VERSION)
        canonical = json.dumps(enriched, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest, canonical

    def get(self, key_payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """Look up a cached evaluation result. Returns None on miss or when disabled/skipped."""
        if not self.enabled or self.skip_read:
            self.misses += 1
            return None

        cache_key, _ = self.compute_key(key_payload)
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT schema_version, value_json FROM accuracy_eval_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        except sqlite3.Error:
            self.misses += 1
            return None

        if not row:
            self.misses += 1
            return None

        schema_ver, value_json = row
        if schema_ver != CACHE_SCHEMA_VERSION:
            self.misses += 1
            return None

        try:
            parsed = json.loads(value_json)
            if isinstance(parsed, dict):
                self.hits += 1
                return parsed
        except (ValueError, TypeError):
            pass

        self.misses += 1
        return None

    def put(self, key_payload: Mapping[str, Any], value_payload: Mapping[str, Any]) -> str:
        """Persist a successful evaluation result in the SQLite cache."""
        cache_key, canonical_key = self.compute_key(key_payload)
        if not self.enabled:
            return cache_key

        suite = str(key_payload.get("suite", ""))
        model_id = str(key_payload.get("model_id", ""))
        meeting_id = str(key_payload.get("meeting_id", ""))
        value_json = json.dumps(dict(value_payload), sort_keys=True)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO accuracy_eval_cache (
                    cache_key, schema_version, suite, model_id, meeting_id,
                    key_json, value_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    value_json = excluded.value_json,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    CACHE_SCHEMA_VERSION,
                    suite,
                    model_id,
                    meeting_id,
                    canonical_key,
                    value_json,
                    time.time(),
                ),
            )
            conn.commit()
        self.writes += 1
        return cache_key

    def clear(self) -> int:
        """Delete all entries in the cache table and return number of rows removed."""
        if not self.cache_path.exists():
            return 0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM accuracy_eval_cache")
            conn.commit()
            return int(cur.rowcount or 0)

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "skip_read": self.skip_read,
            "cache_path": str(self.cache_path),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }


_SPEAKER_COLORS = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
]


def _render_timeline_bars_html(diagnostics: Mapping[str, Any]) -> str:
    """Render proportional Reference vs Hypothesis speaker timeline bars in HTML."""
    ref_runs = diagnostics.get("ref_runs", [])
    hyp_runs = diagnostics.get("hyp_runs", [])
    duration = float(diagnostics.get("duration", 0.0) or 0.0)
    speaker_map = diagnostics.get("speaker_map", {}) or {}
    if duration <= 0 or (not ref_runs and not hyp_runs):
        return ""

    ref_speakers = sorted({r.get("speaker", "") for r in ref_runs if r.get("speaker")})
    color_by_ref = {
        spk: _SPEAKER_COLORS[i % len(_SPEAKER_COLORS)]
        for i, spk in enumerate(ref_speakers)
    }

    def _bar_segments(runs: Sequence[Mapping[str, Any]], is_hyp: bool) -> str:
        segs_html: list[str] = []
        for r in runs:
            spk = str(r.get("speaker", "") or "")
            start = float(r.get("start", 0.0) or 0.0)
            end = float(r.get("end", start) or start)
            width_pct = max(0.0, min(100.0, (end - start) / duration * 100.0))
            if width_pct <= 0.01:
                continue
            if not spk:
                color = "#e5e7eb"
                title = f"silence ({start:.1f}s - {end:.1f}s)"
            else:
                mapped_ref = speaker_map.get(spk) if is_hyp else spk
                color = color_by_ref.get(mapped_ref or "", "#64748b")
                label = f"{spk} → {mapped_ref}" if (is_hyp and mapped_ref) else spk
                title = f"{label} ({start:.1f}s - {end:.1f}s)"
            segs_html.append(
                f'<span style="display:inline-block;width:{width_pct:.3f}%;height:14px;'
                f'background:{color};" title="{html.escape(title)}"></span>'
            )
        return "".join(segs_html)

    ref_bar = _bar_segments(ref_runs, is_hyp=False)
    hyp_bar = _bar_segments(hyp_runs, is_hyp=True)
    return (
        '<div style="margin-top:8px;font-size:12px;">'
        f'<div><strong>Reference Timeline ({duration:.1f}s):</strong>'
        f'<div style="display:flex;width:100%;border:1px solid #cbd5e1;border-radius:3px;overflow:hidden;margin:2px 0 6px 0;">{ref_bar}</div></div>'
        f'<div><strong>Hypothesis Timeline ({duration:.1f}s):</strong>'
        f'<div style="display:flex;width:100%;border:1px solid #cbd5e1;border-radius:3px;overflow:hidden;margin:2px 0;">{hyp_bar}</div></div>'
        "</div>"
    )


def render_notsofar_accuracy_html_report(
    base_url: str,
    split_name: str,
    metrics: Mapping[str, Any],
    accuracy_cases: Sequence[Mapping[str, Any]],
    cache_stats: Mapping[str, Any] | None = None,
) -> str:
    """Render a self-contained HTML accuracy report for NOTSOFAR-1 WER, cpWER, and DER suites."""

    def _mval(key: str, default: Any = None) -> Any:
        entry = metrics.get(key)
        if isinstance(entry, Mapping):
            return entry.get("value", default)
        return default

    wer_val = _mval("wer")
    cpwer_val = _mval("cpwer")
    http_wer = _mval("http_wer")
    streaming_wer = _mval("streaming_wer")
    der_val = _mval("der")
    spk_err = _mval("speaker_count_error")

    cache_hits = int((cache_stats or {}).get("hits", _mval("accuracy_cache_hits", 0)) or 0)
    cache_misses = int((cache_stats or {}).get("misses", _mval("accuracy_cache_misses", 0)) or 0)

    cards_html: list[str] = []
    if wer_val is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Aggregate WER</div>'
            f'<div class="val">{float(wer_val) * 100.0:.2f}%</div>'
            f'<div class="sub">ratio={float(wer_val):.4f}</div></div>'
        )
    if http_wer is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Batch HTTP WER</div>'
            f'<div class="val">{float(http_wer) * 100.0:.2f}%</div>'
            f'<div class="sub">ratio={float(http_wer):.4f}</div></div>'
        )
    if streaming_wer is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Streaming WER</div>'
            f'<div class="val">{float(streaming_wer) * 100.0:.2f}%</div>'
            f'<div class="sub">ratio={float(streaming_wer):.4f}</div></div>'
        )
    if cpwer_val is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Speaker-Attributed cpWER</div>'
            f'<div class="val">{float(cpwer_val) * 100.0:.2f}%</div>'
            f'<div class="sub">ratio={float(cpwer_val):.4f}</div></div>'
        )
    if der_val is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Diarization DER</div>'
            f'<div class="val">{float(der_val) * 100.0:.2f}%</div>'
            f'<div class="sub">ratio={float(der_val):.4f}</div></div>'
        )
    if spk_err is not None:
        cards_html.append(
            f'<div class="card"><div class="label">Speaker Count Error</div>'
            f'<div class="val">{float(spk_err):.2f}</div>'
            f'<div class="sub">|pred - ref| speakers</div></div>'
        )
    errored_cnt = sum(1 for c in accuracy_cases if str(c.get("status", "")).lower() == "error")
    if errored_cnt > 0:
        cards_html.append(
            f'<div class="card"><div class="label">Errored Cases</div>'
            f'<div class="val">{errored_cnt}</div>'
            f'<div class="sub">excluded from aggregates</div></div>'
        )
    cards_html.append(
        f'<div class="card"><div class="label">Result Cache</div>'
        f'<div class="val">{cache_hits}H / {cache_misses}M</div>'
        f'<div class="sub">SQLite eval cache</div></div>'
    )

    rows_html: list[str] = []
    for c in accuracy_cases:
        cid = html.escape(str(c.get("case_id", "")))
        suite = html.escape(str(c.get("suite", "")))
        status = str(c.get("status", "pass")).upper()
        cm = c.get("metrics", {}) if isinstance(c.get("metrics"), Mapping) else {}
        cd = c.get("detail", {}) if isinstance(c.get("detail"), Mapping) else {}

        meeting_id = html.escape(str(cd.get("meeting_id", "")))
        model_id = html.escape(str(cd.get("model", "")))
        cached = bool(cd.get("cache_hit", False))

        c_wer = cm.get("wer", {}).get("value") if isinstance(cm.get("wer"), Mapping) else None
        c_cpwer = cm.get("cpwer", {}).get("value") if isinstance(cm.get("cpwer"), Mapping) else None
        c_der = cm.get("der", {}).get("value") if isinstance(cm.get("der"), Mapping) else None
        c_spk_err = (
            cm.get("speaker_count_error", {}).get("value")
            if isinstance(cm.get("speaker_count_error"), Mapping)
            else None
        )
        c_rtf = cm.get("rtf", {}).get("value") if isinstance(cm.get("rtf"), Mapping) else None

        wer_str = f"{float(c_wer) * 100.0:.2f}%" if c_wer is not None else "—"
        cpwer_str = f"{float(c_cpwer) * 100.0:.2f}%" if c_cpwer is not None else "—"
        der_str = f"{float(c_der) * 100.0:.2f}%" if c_der is not None else "—"
        spk_str = (
            f"{int(c_spk_err)} (pred={cd.get('predicted_speakers')}, ref={cd.get('expected_speakers')})"
            if c_spk_err is not None
            else "—"
        )
        rtf_str = f"{float(c_rtf):.4f}x" if c_rtf is not None else "—"

        ref_preview = html.escape(str(cd.get("reference_preview", ""))[:320])
        hyp_preview = html.escape(str(cd.get("hypothesis_preview", ""))[:320])
        diag = cd.get("speaker_diagnostics") if isinstance(cd.get("speaker_diagnostics"), Mapping) else {}
        timeline_html = _render_timeline_bars_html(diag) if diag else ""
        err_msg = html.escape(str(cd.get("error_message") or cd.get("eval_message") or "")) if status == "ERROR" else ""

        detail_block = ""
        if status == "ERROR" and err_msg:
            detail_block = f'<span style="color:#dc2626;font-size:12px;">{err_msg}</span>'
        elif ref_preview or hyp_preview or timeline_html:
            detail_block = (
                f"<details><summary>Inspect {meeting_id}</summary>"
                f'<div style="padding:6px 0;font-size:12px;">'
                + (f"<div><strong>Ref:</strong> {ref_preview}</div>" if ref_preview else "")
                + (f"<div><strong>Hyp:</strong> {hyp_preview}</div>" if hyp_preview else "")
                + timeline_html
                + "</div></details>"
            )

        rows_html.append(
            f"<tr>"
            f"<td><code>{cid}</code></td>"
            f"<td>{suite}</td>"
            f"<td>{model_id}</td>"
            f"<td>{meeting_id}</td>"
            f"<td>{status}</td>"
            f"<td>{wer_str}</td>"
            f"<td>{cpwer_str}</td>"
            f"<td>{der_str}</td>"
            f"<td>{spk_str}</td>"
            f"<td>{rtf_str}</td>"
            f"<td>{'HIT' if cached else 'MISS'}</td>"
            f"<td>{detail_block}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>NOTSOFAR-1 Accuracy Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #0f172a; background: #f8fafc; }}
  h1 {{ margin-bottom: 4px; font-size: 22px; }}
  .meta {{ color: #475569; font-size: 13px; margin-bottom: 18px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  .card .label {{ font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.04em; }}
  .card .val {{ font-size: 20px; font-weight: 700; margin: 4px 0; color: #0f172a; }}
  .card .sub {{ font-size: 11px; color: #64748b; }}
  table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #e2e8f0; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
  code {{ font-size: 12px; color: #1e293b; }}
</style>
</head>
<body>
  <h1>NOTSOFAR-1 Accuracy Evaluation Report</h1>
  <div class="meta">Target: <code>{html.escape(base_url)}</code> | Split: <code>{html.escape(split_name)}</code> | Cases: {len(accuracy_cases)}</div>
  <div class="cards">{''.join(cards_html)}</div>
  <table>
    <thead>
      <tr>
        <th>Case ID</th>
        <th>Suite</th>
        <th>Model</th>
        <th>Meeting</th>
        <th>Status</th>
        <th>WER</th>
        <th>cpWER</th>
        <th>DER</th>
        <th>Speaker Err</th>
        <th>RTF</th>
        <th>Cache</th>
        <th>Diagnostics</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""


DEFAULT_SPEECH_WER_THRESHOLD = 0.75


def speech_accuracy_check(
    reference_text: str,
    hypothesis_text: str | None,
    threshold: float = DEFAULT_SPEECH_WER_THRESHOLD,
) -> tuple[bool, dict[str, Any], str | None, dict[str, Any], dict[str, Any]]:
    """Loose accuracy sanity check for real-speech functional fixtures.

    Catches a server that returns empty or garbage transcripts (WER near 100%) without failing on
    normal model-to-model accuracy differences. Returns
    ``(passed, check, error_or_None, extra_metrics, extra_detail)``.
    """
    wer_res = compute_wer(reference_text, hypothesis_text or "")
    passed = wer_res.wer <= float(threshold)
    detail_text = (
        f"WER={wer_res.wer_pct:.1f}% ({wer_res.edit_distance}/{wer_res.ref_word_count} words), "
        f"threshold={float(threshold) * 100.0:.0f}%"
    )
    check = {"name": "speech_wer_within_threshold", "passed": passed, "detail": detail_text}
    error = None if passed else f"speech_wer_within_threshold: {detail_text}"
    metrics = {
        "wer": {"value": wer_res.wer, "unit": "ratio", "better": "lower"},
        "wer_pct": {"value": wer_res.wer_pct, "unit": "%", "better": "lower"},
    }
    detail = {"wer": wer_res.to_dict(), "reference_text": reference_text[:500]}
    return passed, check, error, metrics, detail
