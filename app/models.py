"""Lightweight internal models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PersonRecord:
    """Represents a person stored in the database."""

    name: str
    embedding: np.ndarray
    images_used: int
    created_at: str
    updated_at: str


@dataclass(slots=True)
class FaceRecord:
    """Represents one recorded face sample tied to a person."""

    person_name: str
    embedding: np.ndarray
    source_label: str
    confidence: float | None
    created_at: str
