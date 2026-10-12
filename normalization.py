"""Classic face-normalization preprocessing.

This implementation follows the virtual-camera warp introduced by Zhang,
Sugano, and Bulling in "Revisiting Data Normalization for Appearance-Based
Gaze Estimation" (ETRA 2018). It is included as a baseline for comparison and
reproducibility, rather than as a contribution of the CROp work.

Reference:
https://doi.org/10.1145/3204493.3204548
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class PreprocessResult:
    """Result returned by ``normalize_preprocess``.

    Attributes:
        image: Normalized image patch.
        rotation: 3x3 rotation from original camera coordinates to normalized coordinates.
        transform_matrix: 3x3 homography from original image pixels to normalized pixels.
        headpose_matrix: Head-pose rotation transformed to normalized coordinates.
        gaze_vector: Unit gaze vector transformed to normalized coordinates, if provided.
        gaze_pitchyaw: Pitch/yaw label derived from ``-gaze_vector``, if provided.
        landmarks: 2D landmarks transformed to normalized pixels, if provided.
        bbox: Always ``None`` for classic face normalization.
    """

    image: np.ndarray
    rotation: np.ndarray
    transform_matrix: np.ndarray
    headpose_matrix: Optional[np.ndarray] = None
    gaze_vector: Optional[np.ndarray] = None
    gaze_pitchyaw: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


def normalize_preprocess(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    face_center: ArrayLike,
    headpose: ArrayLike,
    gaze_vector: Optional[ArrayLike] = None,
    landmarks: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (448, 448),
    focal_norm: float = 960.0,
    distance_norm: float = 300.0,
) -> PreprocessResult:
    """Apply the classic virtual-camera face-normalization warp."""

    face_center = _as_vector3(face_center)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    headpose_matrix_original = _as_rotation_matrix(headpose)

    width, height = _ensure_size(output_size)
    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance

    camera_norm = np.array(
        [
            [focal_norm, 0.0, width / 2.0],
            [0.0, focal_norm, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ],
        dtype=np.float64,
    )

    rotation = get_normalization_rotation(face_center, headpose_matrix_original)
    transform_matrix = camera_norm @ scale_matrix @ rotation @ np.linalg.inv(camera_matrix)
    patch = cv2.warpPerspective(image, transform_matrix, (width, height))

    transformed_gaze, gaze_pitchyaw = _transform_gaze(rotation, gaze_vector=gaze_vector)
    transformed_landmarks = _transform_landmarks(landmarks, transform_matrix)

    return PreprocessResult(
        image=patch,
        rotation=rotation,
        transform_matrix=transform_matrix,
        headpose_matrix=rotation @ headpose_matrix_original,
        gaze_vector=transformed_gaze,
        gaze_pitchyaw=gaze_pitchyaw,
        landmarks=transformed_landmarks,
        bbox=None,
    )


def get_normalization_rotation(face_center: ArrayLike, headpose: ArrayLike) -> np.ndarray:
    """Compute the rotation used by face normalization."""

    face_center = _as_vector3(face_center)
    headpose = _as_rotation_matrix(headpose)

    head_x_axis = headpose[:, 0]
    forward = face_center / np.linalg.norm(face_center)
    down = np.cross(forward, head_x_axis)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.c_[right, down, forward].T


def vector_to_pitchyaw(vectors: np.ndarray) -> np.ndarray:
    """Convert 3D vectors to pitch/yaw angles in radians."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, 3)

    norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    vectors = vectors / np.clip(norms, 1e-7, None)

    pitchyaws = np.empty((vectors.shape[0], 2), dtype=np.float64)
    pitchyaws[:, 0] = np.arcsin(vectors[:, 1])
    pitchyaws[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])
    return pitchyaws


def _as_vector3(value: ArrayLike) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape[0] != 3:
        raise ValueError(f"Expected a 3D vector, got shape {np.asarray(value).shape}.")
    return vector


def _as_rotation_matrix(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        return array
    if array.size == 3:
        return cv2.Rodrigues(array.reshape(3, 1))[0]
    raise ValueError(f"Expected a 3x3 rotation matrix or 3D Rodrigues vector, got shape {array.shape}.")


def _ensure_size(output_size: Tuple[int, int]) -> Tuple[int, int]:
    if len(output_size) != 2:
        raise ValueError("output_size must be a (width, height) tuple.")
    width, height = int(output_size[0]), int(output_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("output_size dimensions must be positive.")
    return width, height


def _transform_gaze(
    rotation: np.ndarray,
    gaze_vector: Optional[ArrayLike] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if gaze_vector is None:
        return None, None

    gaze = _as_vector3(gaze_vector)
    gaze = gaze / np.linalg.norm(gaze)
    transformed = rotation @ gaze.reshape(3, 1)
    transformed = transformed.reshape(3)
    transformed = transformed / np.linalg.norm(transformed)
    pitchyaw = vector_to_pitchyaw(-transformed.reshape(1, 3))[0]
    return transformed, pitchyaw


def _transform_landmarks(landmarks: Optional[np.ndarray], transform_matrix: np.ndarray) -> Optional[np.ndarray]:
    if landmarks is None:
        return None

    landmarks = np.asarray(landmarks, dtype=np.float64)
    points = landmarks[:, :2].reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, transform_matrix).reshape(-1, 2)
    if landmarks.shape[1] > 2:
        transformed = np.concatenate([transformed, landmarks[:, 2:]], axis=1)
    return transformed
