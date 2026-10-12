"""CROp preprocessing.

This self-contained implementation is designed to be easy to
integrate into existing appearance-based gaze-estimation pipelines.

For details, see:
Santambrogio et al., "Constrained Rotation Optimization: Revisiting Crop-Based Gaze Estimation," ECCV 2026.
https://doi.org/10.1007/978-3-032-37232-1_20
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class PreprocessResult:
    """Result returned by ``crop_preprocess``.

    Attributes:
        image: Cropped and resized image patch.
        rotation: 3x3 rotation from original camera coordinates to crop coordinates.
        transform_matrix: 3x3 image transform from original pixels to crop pixels.
        headpose_matrix: Head-pose rotation transformed to crop coordinates, if provided.
        gaze_vector: Unit gaze vector transformed to crop coordinates, if provided.
        gaze_pitchyaw: Pitch/yaw label derived from ``-gaze_vector``, if provided.
        landmarks: 2D landmarks translated and resized to crop pixels, if provided.
        bbox: Source crop bounding box as ``(x1, y1, x2, y2)``.
    """

    image: np.ndarray
    rotation: np.ndarray
    transform_matrix: np.ndarray
    headpose_matrix: Optional[np.ndarray] = None
    gaze_vector: Optional[np.ndarray] = None
    gaze_pitchyaw: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


def crop_preprocess(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    bbox: Sequence[float],
    headpose: Optional[ArrayLike] = None,
    gaze_vector: Optional[ArrayLike] = None,
    landmarks: Optional[np.ndarray] = None,
    output_size: Tuple[int, int] = (448, 448),
) -> PreprocessResult:
    """Apply CROp preprocessing to an image and face bounding box.

    The required inputs are the image, camera intrinsics, and face bounding box.
    Optional labels are transformed to the crop coordinate system when provided.

    ``bbox`` accepts either ``(x1, y1, x2, y2)`` or four 2D corner points with
    shape ``(4, 2)``. ``gaze_vector`` should be a 3D gaze direction in camera
    coordinates.
    """

    bbox = tuple(int(np.rint(value)) for value in _bbox_to_xyxy(bbox))
    rotation, _ = get_crop_rotation(bbox, camera_matrix)
    patch, transform_matrix = _crop_and_resize(image, bbox, output_size)

    headpose_matrix = None
    if headpose is not None:
        headpose_matrix = rotation @ _as_rotation_matrix(headpose)

    transformed_gaze, gaze_pitchyaw = _transform_gaze(rotation, gaze_vector=gaze_vector)
    transformed_landmarks = _transform_landmarks(landmarks, transform_matrix)

    return PreprocessResult(
        image=patch,
        rotation=rotation,
        transform_matrix=transform_matrix,
        headpose_matrix=headpose_matrix,
        gaze_vector=transformed_gaze,
        gaze_pitchyaw=gaze_pitchyaw,
        landmarks=transformed_landmarks,
        bbox=bbox,
    )


def get_crop_rotation(bbox: Sequence[float], camera_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the CROp rotation for a face bounding box.

    The center ray is constrained exactly and the crop corners are aligned in a
    least-squares sense. The returned error is SciPy's root-sum-squared distance.
    """

    x1, y1, x2, y2 = _bbox_to_xyxy(bbox)
    crop_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    points = np.array(
        [
            [crop_center[0], crop_center[1], 1.0],
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x1, y2, 1.0],
            [x2, y2, 1.0],
        ],
        dtype=np.float64,
    ).T

    ray_camera = np.linalg.inv(camera_matrix) @ points
    ray_camera /= np.linalg.norm(ray_camera, axis=0)

    crop_camera_matrix = camera_matrix.copy()
    crop_camera_matrix[:2, 2] = crop_center
    ray_crop = np.linalg.inv(crop_camera_matrix) @ points
    ray_crop /= np.linalg.norm(ray_crop, axis=0)

    rotation, error = Rotation.align_vectors(
        ray_crop.T,
        ray_camera.T,
        weights=[np.inf, 1.0, 1.0, 1.0, 1.0],
    )
    return rotation.as_matrix(), float(error)


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


def _bbox_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    bbox_array = np.asarray(bbox, dtype=np.float64)
    if bbox_array.shape == (4,):
        x1, y1, x2, y2 = bbox_array
    elif bbox_array.shape == (4, 2):
        x1 = bbox_array[:, 0].min()
        y1 = bbox_array[:, 1].min()
        x2 = bbox_array[:, 0].max()
        y2 = bbox_array[:, 1].max()
    else:
        raise ValueError("bbox must be (x1, y1, x2, y2) or four corner points with shape (4, 2).")

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox {bbox}. Expected x2>x1 and y2>y1.")
    return float(x1), float(y1), float(x2), float(y2)


def _crop_and_resize(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox {bbox}. Expected x2>x1 and y2>y1.")

    crop_width = x2 - x1
    crop_height = y2 - y1
    crop = np.zeros((crop_height, crop_width, image.shape[2]), dtype=image.dtype)

    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    src_x2 = min(x2, image.shape[1])
    src_y2 = min(y2, image.shape[0])

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        crop[
            dst_y1 : dst_y1 + (src_y2 - src_y1),
            dst_x1 : dst_x1 + (src_x2 - src_x1),
        ] = image[src_y1:src_y2, src_x1:src_x2]

    square_size = max(crop_height, crop_width)
    pad_top = (square_size - crop_height) // 2
    pad_bottom = square_size - crop_height - pad_top
    pad_left = (square_size - crop_width) // 2
    pad_right = square_size - crop_width - pad_left
    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    width, height = _ensure_size(output_size)
    patch = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

    scale_x = width / square_size
    scale_y = height / square_size
    transform_matrix = np.array(
        [
            [scale_x, 0.0, scale_x * (pad_left - x1)],
            [0.0, scale_y, scale_y * (pad_top - y1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return patch, transform_matrix


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
