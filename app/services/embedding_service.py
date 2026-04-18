"""Registration pipeline."""

from __future__ import annotations

import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.db import add_face_record, get_face_records_by_name, upsert_person
from app.services.face_service import get_face_service
from app.utils.image_utils import bgr_to_rgb, decode_image, validate_file_size, validate_upload_file
from app.utils.math_utils import average_vectors


def register_person(name: str, files: list[UploadFile]) -> dict[str, int | str]:
    """Register or replace a person using one or more uploaded images."""

    normalized_name = name.strip()
    if not normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="name must not be empty",
        )
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="at least one file is required",
        )

    face_service = get_face_service()
    embeddings: list[np.ndarray] = []
    images_received = len(files)

    for upload_file in files:
        validate_upload_file(upload_file)
        file_bytes = upload_file.file.read()
        validate_file_size(file_bytes)
        image_rgb = bgr_to_rgb(decode_image(file_bytes))
        result = face_service.extract_largest_face(image_rgb)
        if result is None:
            continue
        face_tensor, _ = result
        embeddings.append(face_service.embed_face(face_tensor))

    if not embeddings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="no valid faces found in uploaded images",
        )

    _save_face_samples(normalized_name, embeddings, "registration_upload")
    total_images_used = _refresh_person_embedding(normalized_name)
    return {
        "name": normalized_name,
        "images_received": images_received,
        "images_used": total_images_used,
        "message": "person registered successfully",
    }


def identify_face(
    name: str,
    upload_file: UploadFile,
    confidence: float | None = None,
) -> dict[str, int | str]:
    """Register one manually labeled face crop from the frontend."""

    normalized_name = name.strip()
    if not normalized_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="name must not be empty",
        )

    validate_upload_file(upload_file)
    file_bytes = upload_file.file.read()
    validate_file_size(file_bytes)
    image_rgb = bgr_to_rgb(decode_image(file_bytes))

    face_service = get_face_service()
    result = face_service.extract_largest_face(image_rgb)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="no detectable face found in uploaded crop",
        )

    face_tensor, _ = result
    embedding = face_service.embed_face(face_tensor)
    _save_face_samples(normalized_name, [embedding], "manual_identification", confidence)
    total_images_used = _refresh_person_embedding(normalized_name)
    return {
        "name": normalized_name,
        "images_used": total_images_used,
        "message": "face recorded successfully",
    }


def _save_face_samples(
    name: str,
    embeddings: list[np.ndarray],
    source_label: str,
    confidence: float | None = None,
) -> None:
    for embedding in embeddings:
        add_face_record(name, embedding, source_label, confidence)


def _refresh_person_embedding(name: str) -> int:
    records = get_face_records_by_name(name)
    canonical_embedding = average_vectors([record.embedding for record in records])
    upsert_person(name, canonical_embedding, len(records))
    return len(records)
