from __future__ import annotations

import numpy as np

from app.utils.math_utils import average_vectors, cosine_similarity


def test_cosine_similarity_for_identical_vectors() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(vector, vector) == 1.0


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    vector_a = np.array([1.0, 0.0], dtype=np.float32)
    vector_b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(vector_a, vector_b) == 0.0


def test_average_vectors() -> None:
    vectors = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([3.0, 4.0, 5.0], dtype=np.float32),
    ]
    averaged = average_vectors(vectors)
    assert np.allclose(averaged, np.array([2.0, 3.0, 4.0], dtype=np.float32))
