"""SQLite access helpers."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from app.config import settings
from app.models import FaceRecord, PersonRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Return a SQLite connection with row access by column name."""

    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create database tables if they do not exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                embedding_json TEXT NOT NULL,
                images_used INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS face_records (
                id INTEGER PRIMARY KEY,
                person_name TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                source_label TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
            """
        )


def upsert_person(name: str, embedding: np.ndarray, images_used: int) -> None:
    """Insert or replace a person record by name."""

    now = _utc_now_iso()
    embedding_json = json.dumps(embedding.tolist())
    with get_connection() as connection:
        existing = connection.execute(
            "SELECT created_at FROM people WHERE name = ?",
            (name,),
        ).fetchone()
        created_at = existing["created_at"] if existing else now
        connection.execute(
            """
            INSERT INTO people (name, embedding_json, images_used, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                images_used = excluded.images_used,
                updated_at = excluded.updated_at
            """,
            (name, embedding_json, images_used, created_at, now),
        )


def get_all_people() -> list[PersonRecord]:
    """Fetch all people ordered alphabetically by name."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT name, embedding_json, images_used, created_at, updated_at
            FROM people
            ORDER BY name ASC
            """
        ).fetchall()

    return [_row_to_person(row) for row in rows]


def get_person_by_name(name: str) -> PersonRecord | None:
    """Fetch one person by name."""

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT name, embedding_json, images_used, created_at, updated_at
            FROM people
            WHERE name = ?
            """,
            (name,),
        ).fetchone()

    return _row_to_person(row) if row else None


def delete_person_by_name(name: str) -> bool:
    """Delete a person by name."""

    with get_connection() as connection:
        connection.execute("DELETE FROM face_records WHERE person_name = ?", (name,))
        cursor = connection.execute("DELETE FROM people WHERE name = ?", (name,))
        return cursor.rowcount > 0


def add_face_record(
    person_name: str,
    embedding: np.ndarray,
    source_label: str,
    confidence: float | None = None,
) -> None:
    """Store one labeled face sample."""

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO face_records (person_name, embedding_json, source_label, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                person_name,
                json.dumps(embedding.tolist()),
                source_label,
                confidence,
                _utc_now_iso(),
            ),
        )


def get_face_records_by_name(name: str) -> list[FaceRecord]:
    """Fetch all recorded face samples for one person."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT person_name, embedding_json, source_label, confidence, created_at
            FROM face_records
            WHERE person_name = ?
            ORDER BY created_at ASC
            """,
            (name,),
        ).fetchall()
    return [_row_to_face_record(row) for row in rows]


def _row_to_person(row: sqlite3.Row) -> PersonRecord:
    embedding = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
    return PersonRecord(
        name=row["name"],
        embedding=embedding,
        images_used=row["images_used"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_face_record(row: sqlite3.Row) -> FaceRecord:
    embedding = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
    return FaceRecord(
        person_name=row["person_name"],
        embedding=embedding,
        source_label=row["source_label"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )
