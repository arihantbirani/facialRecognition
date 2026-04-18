"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class Settings:
    """Central application settings."""

    database_url: str = os.getenv("DATABASE_URL", "").strip()
    database_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATABASE_PATH", BASE_DIR / "face_recognition.db")
        )
    )
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
    face_detection_confidence_threshold: float = float(
        os.getenv("FACE_DETECTION_CONFIDENCE_THRESHOLD", "0.90")
    )
    min_face_box_size: int = int(os.getenv("MIN_FACE_BOX_SIZE", "40"))
    secondary_detection_scale: float = float(os.getenv("SECONDARY_DETECTION_SCALE", "1.6"))
    detection_merge_iou_threshold: float = float(
        os.getenv("DETECTION_MERGE_IOU_THRESHOLD", "0.35")
    )
    allowed_extensions: set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".bmp"}
    )
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(5 * 1024 * 1024)))
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )


settings = Settings()
