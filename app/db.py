"""Database access helpers supporting SQLite locally and Postgres in deployment."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

import numpy as np

from app.config import settings
from app.models import FaceRecord, PersonRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def get_connection() -> Iterator[Any]:
    """Return a database connection with dict-like row access."""

    if _uses_postgres():
        connection = psycopg.connect(settings.database_url, row_factory=dict_row)
    else:
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
        if _uses_postgres():
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id BIGSERIAL PRIMARY KEY,
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
                    id BIGSERIAL PRIMARY KEY,
                    person_name TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    created_at TEXT NOT NULL
                )
                """
            )
        else:
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
        existing = _fetchone(
            connection,
            "SELECT created_at FROM people WHERE name = ?",
            (name,),
        )
        created_at = existing["created_at"] if existing else now
        _execute(
            connection,
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
        rows = _fetchall(
            connection,
            """
            SELECT name, embedding_json, images_used, created_at, updated_at
            FROM people
            ORDER BY name ASC
            """
        )

    return [_row_to_person(row) for row in rows]


def get_person_by_name(name: str) -> PersonRecord | None:
    """Fetch one person by name."""

    with get_connection() as connection:
        row = _fetchone(
            connection,
            """
            SELECT name, embedding_json, images_used, created_at, updated_at
            FROM people
            WHERE name = ?
            """,
            (name,),
        )

    return _row_to_person(row) if row else None


def delete_person_by_name(name: str) -> bool:
    """Delete a person by name."""

    with get_connection() as connection:
        _execute(connection, "DELETE FROM face_records WHERE person_name = ?", (name,))
        return _execute(connection, "DELETE FROM people WHERE name = ?", (name,)) > 0


def add_face_record(
    person_name: str,
    embedding: np.ndarray,
    source_label: str,
    confidence: float | None = None,
) -> None:
    """Store one labeled face sample."""

    with get_connection() as connection:
        _execute(
            connection,
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
        rows = _fetchall(
            connection,
            """
            SELECT person_name, embedding_json, source_label, confidence, created_at
            FROM face_records
            WHERE person_name = ?
            ORDER BY created_at ASC
            """,
            (name,),
        )
    return [_row_to_face_record(row) for row in rows]


def _row_to_person(row: sqlite3.Row | dict[str, Any]) -> PersonRecord:
    embedding = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
    return PersonRecord(
        name=row["name"],
        embedding=embedding,
        images_used=row["images_used"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_face_record(row: sqlite3.Row | dict[str, Any]) -> FaceRecord:
    embedding = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
    return FaceRecord(
        person_name=row["person_name"],
        embedding=embedding,
        source_label=row["source_label"],
        confidence=row["confidence"],
        created_at=row["created_at"],
    )


def _uses_postgres() -> bool:
    return settings.database_url.startswith(("postgres://", "postgresql://"))


def _placeholder_query(query: str) -> str:
    if not _uses_postgres():
        return query
    return query.replace("?", "%s")


def _execute(connection: Any, query: str, params: tuple[Any, ...] = ()) -> int:
    cursor = connection.execute(_placeholder_query(query), params)
    return cursor.rowcount if cursor.rowcount is not None else 0


def _fetchone(connection: Any, query: str, params: tuple[Any, ...] = ()) -> Any:
    return connection.execute(_placeholder_query(query), params).fetchone()


def _fetchall(connection: Any, query: str, params: tuple[Any, ...] = ()) -> list[Any]:
    return list(connection.execute(_placeholder_query(query), params).fetchall())
