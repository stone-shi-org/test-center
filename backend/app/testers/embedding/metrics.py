"""Vector calculations, retrieval evaluation, and statistical metrics."""

from __future__ import annotations

import statistics
from typing import Sequence
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of matrix a and matrix b.

    Args:
        a: 2D array of shape [N, dim]
        b: 2D array of shape [M, dim]

    Returns:
        2D array of shape [N, M] with cosine similarities in [-1.0, 1.0].
    """
    if a.size == 0 or b.size == 0:
        return np.empty((a.shape[0], b.shape[0]), dtype=np.float32)

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.dot(a_norm, b_norm.T)


def compute_vector_norms(vectors: np.ndarray) -> dict[str, float]:
    """Compute vector L2 norm distribution statistics."""
    if vectors.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    norms = np.linalg.norm(vectors, axis=1)
    return {
        "mean": round(float(np.mean(norms)), 4),
        "std": round(float(np.std(norms)), 4),
        "min": round(float(np.min(norms)), 4),
        "max": round(float(np.max(norms)), 4),
    }


def compute_anisotropy(vectors: np.ndarray) -> float:
    """Compute mean pairwise cosine similarity among corpus vectors (off-diagonal).

    Lower values indicate more discriminative, isotropic embeddings.
    """
    n = vectors.shape[0]
    if n <= 1:
        return 1.0

    sim_matrix = cosine_similarity(vectors, vectors)
    off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
    mean_sim = off_diag_sum / (n * (n - 1))
    return round(float(mean_sim), 4)


def compute_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    return float(np.linalg.norm(a - b))


def compute_percentiles(values: Sequence[float], percentiles: Sequence[float] = (50, 95, 99)) -> dict[int, float]:
    """Compute percentile values for a sequence of numbers."""
    if not values:
        return {int(p): 0.0 for p in percentiles}
    return {int(p): round(float(np.percentile(values, p)), 2) for p in percentiles}


def compute_retrieval_metrics(ranks: Sequence[int], top_sims: Sequence[float]) -> dict[str, float]:
    """Compute Recall@k (1, 3, 5), MRR, mean rank, and top similarity.

    Args:
        ranks: 1-indexed ranks of expected target documents.
        top_sims: Top-1 similarity scores for each query.

    Returns:
        Dict containing recall_at_1, recall_at_3, recall_at_5, mrr, mean_rank, mean_top_similarity.
    """
    total = len(ranks)
    if total == 0:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "mean_top_similarity": 0.0,
        }

    hits_at_1 = sum(1 for r in ranks if r == 1)
    hits_at_3 = sum(1 for r in ranks if r <= 3)
    hits_at_5 = sum(1 for r in ranks if r <= 5)

    reciprocal_ranks = [1.0 / r for r in ranks]

    return {
        "recall_at_1": round(hits_at_1 / total * 100.0, 2),
        "recall_at_3": round(hits_at_3 / total * 100.0, 2),
        "recall_at_5": round(hits_at_5 / total * 100.0, 2),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "mean_rank": round(statistics.mean(ranks), 2),
        "mean_top_similarity": round(statistics.mean(top_sims), 4) if top_sims else 0.0,
    }
