"""Image decoding and validation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import settings


def supported_extensions_message() -> str:
    """Return a stable human-readable list of supported extensions."""

    return ", ".join(sorted(settings.allowed_extensions))


def validate_upload_file(upload_file: UploadFile) -> None:
    """Validate file extension before processing image bytes."""

    extension = Path(upload_file.filename or "").suffix.lower()
    if not extension or extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported file type; allowed extensions: {supported_extensions_message()}",
        )


def validate_file_size(file_bytes: bytes) -> None:
    """Reject files larger than the configured upload limit."""

    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is too large",
        )


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR image."""

    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unable to decode image",
        )
    return image_bgr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to RGB."""

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def clamp_box(box: list[float] | tuple[float, float, float, float], width: int, height: int) -> list[int]:
    """Clamp floating point bounding box values to integer image bounds."""

    x1, y1, x2, y2 = box
    clamped = [
        max(0, min(int(round(x1)), width - 1)),
        max(0, min(int(round(y1)), height - 1)),
        max(0, min(int(round(x2)), width - 1)),
        max(0, min(int(round(y2)), height - 1)),
    ]
    if clamped[2] < clamped[0]:
        clamped[0], clamped[2] = clamped[2], clamped[0]
    if clamped[3] < clamped[1]:
        clamped[1], clamped[3] = clamped[3], clamped[1]
    return clamped
