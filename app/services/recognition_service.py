"""Face recognition pipeline."""

from __future__ import annotations

from fastapi import UploadFile

from app.config import settings
from app.db import get_all_people
from app.services.face_service import get_face_service
from app.utils.image_utils import bgr_to_rgb, decode_image, validate_file_size, validate_upload_file
from app.utils.math_utils import cosine_similarity


def recognize_faces(upload_file: UploadFile) -> list[dict[str, object]]:
    """Detect and recognize all faces in an uploaded image."""

    validate_upload_file(upload_file)
    file_bytes = upload_file.file.read()
    validate_file_size(file_bytes)
    image_rgb = bgr_to_rgb(decode_image(file_bytes))

    face_service = get_face_service()
    face_crops = face_service.extract_all_faces(image_rgb)
    if not face_crops:
        return []

    people = get_all_people()
    matches: list[dict[str, object]] = []

    for face_tensor, box in face_crops:
        embedding = face_service.embed_face(face_tensor)
        identity = "unknown"
        best_score = 0.0

        for person in people:
            score = cosine_similarity(embedding, person.embedding)
            if score > best_score:
                best_score = score
                identity = person.name

        is_known = best_score >= settings.similarity_threshold and bool(people)
        if not is_known:
            identity = "unknown"

        matches.append(
            {
                "box": box,
                "identity": identity,
                "confidence": round(best_score, 4),
                "is_known": is_known,
            }
        )

    matches.sort(key=lambda match: match["box"][0])  # type: ignore[index]
    return matches
