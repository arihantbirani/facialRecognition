"""Vector math helpers."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""

    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def average_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Average a non-empty list of vectors."""

    if not vectors:
        raise ValueError("vectors must not be empty")
    stacked = np.stack(vectors, axis=0)
    return np.mean(stacked, axis=0, dtype=np.float32)
