"""Face detection and embedding helpers."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from app.config import settings
from app.utils.image_utils import clamp_box


class FaceService:
    """Wraps face detection and embedding model access."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.detector = MTCNN(keep_all=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect_faces(self, image_rgb: np.ndarray) -> list[list[int]]:
        """Detect all faces and return clamped bounding boxes."""

        height, width = image_rgb.shape[:2]
        detections = self._detect_faces_at_scale(image_rgb, scale=1.0)
        detections.extend(
            self._detect_faces_at_scale(
                image_rgb,
                scale=settings.secondary_detection_scale,
            )
        )
        if not detections:
            return []

        merged_detections = _non_max_suppression(
            detections,
            settings.detection_merge_iou_threshold,
        )
        return [clamp_box(detection["box"], width, height) for detection in merged_detections]

    def extract_largest_face(
        self, image_rgb: np.ndarray
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Extract the largest detected face from an image."""

        faces = self.extract_all_faces(image_rgb)
        if not faces:
            return None
        return max(faces, key=lambda item: _box_area(item[1]))

    def extract_all_faces(self, image_rgb: np.ndarray) -> list[tuple[torch.Tensor, list[int]]]:
        """Extract face tensors for every detected face."""

        boxes = self.detect_faces(image_rgb)
        if not boxes:
            return []

        faces: list[tuple[torch.Tensor, list[int]]] = []
        for box in boxes:
            face_tensor = self.detector.extract(image_rgb, [box], save_path=None)
            if face_tensor is None or len(face_tensor) == 0:
                continue
            faces.append((face_tensor[0], box))
        return faces

    def embed_face(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Generate an embedding vector for one aligned face tensor."""

        with torch.no_grad():
            embedding = self.embedder(face_tensor.unsqueeze(0).to(self.device))
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def _detect_faces_at_scale(
        self,
        image_rgb: np.ndarray,
        scale: float,
    ) -> list[dict[str, object]]:
        """Run MTCNN at one scale and return scored detections."""

        height, width = image_rgb.shape[:2]
        scaled_image = image_rgb
        if scale != 1.0:
            scaled_width = max(1, int(round(width * scale)))
            scaled_height = max(1, int(round(height * scale)))
            scaled_image = cv2.resize(
                image_rgb,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_LINEAR,
            )

        boxes, probabilities = self.detector.detect(scaled_image)
        if boxes is None:
            return []

        detections: list[dict[str, object]] = []
        for index, box in enumerate(boxes):
            probability = float(probabilities[index]) if probabilities is not None else 1.0
            if probability < settings.face_detection_confidence_threshold:
                continue

            scaled_box = box.tolist()
            if scale != 1.0:
                scaled_box = [coordinate / scale for coordinate in scaled_box]

            clamped_box = clamp_box(scaled_box, width, height)
            if _box_width(clamped_box) < settings.min_face_box_size:
                continue
            if _box_height(clamped_box) < settings.min_face_box_size:
                continue

            detections.append({"box": clamped_box, "score": probability})

        return detections


@lru_cache(maxsize=1)
def get_face_service() -> FaceService:
    """Lazily initialize models once per process."""

    return FaceService()


def _box_area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_width(box: list[int]) -> int:
    return max(0, box[2] - box[0])


def _box_height(box: list[int]) -> int:
    return max(0, box[3] - box[1])


def _intersection_over_union(box_a: list[int], box_b: list[int]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    union_area = _box_area(box_a) + _box_area(box_b) - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area


def _non_max_suppression(
    detections: list[dict[str, object]],
    iou_threshold: float,
) -> list[dict[str, object]]:
    ordered = sorted(
        detections,
        key=lambda detection: float(detection["score"]),
        reverse=True,
    )
    kept: list[dict[str, object]] = []

    for detection in ordered:
        box = detection["box"]
        if any(
            _intersection_over_union(box, existing["box"]) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(detection)

    return kept
