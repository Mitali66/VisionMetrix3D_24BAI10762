"""
VisionMetrix 3D - Corner & Keypoint Detectors
Module: corners.py

Theoretical Background:
1. Harris Corner Detector:
   Auto-correlation matrix (Structure Tensor) M over a local Gaussian window:
     M = [ sum w * Ix^2     sum w * Ix * Iy ]
         [ sum w * Ix * Iy  sum w * Iy^2    ]
   Corner Response Function:
     R = det(M) - k * (trace(M))^2 = (lambda1 * lambda2) - k * (lambda1 + lambda2)^2
   - R > 0: Corner (two large eigenvalues)
   - R < 0: Edge (one dominant eigenvalue)
   - |R| small: Flat homogeneous region
2. Hessian Blob Detector:
   Determinant of Hessian matrix H = [Ixx, Ixy; Ixy, Iyy]:
     det(H) = Ixx * Iyy - (Ixy)^2
"""

import cv2
import numpy as np
from typing import Tuple, List


def harris_corner_detector(image: np.ndarray,
                           k: float = 0.04,
                           threshold_rel: float = 0.01,
                           block_size: int = 3,
                           ksize: int = 3) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Computes Harris Corner Response R and performs local non-maximum suppression.
    
    Returns:
      response_map: Continuous Harris response R
      corner_coords: List of extracted (x, y) corner coordinates
      overlay: Visual canvas with detected corners marked as circles
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        overlay = image.copy()
    else:
        gray = image.copy()
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    gray_float = gray.astype(np.float32)

    # First derivatives
    Ix = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=ksize)

    # Structure tensor components
    Ixx = cv2.GaussianBlur(Ix**2, (block_size, block_size), 1.0)
    Iyy = cv2.GaussianBlur(Iy**2, (block_size, block_size), 1.0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (block_size, block_size), 1.0)

    # Harris Response R = det(M) - k * trace(M)^2
    det_M = (Ixx * Iyy) - (Ixy**2)
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M**2)

    # Non-maximum suppression over 3x3 local neighborhood
    thresh = threshold_rel * np.max(R) if np.max(R) > 0 else 1.0
    corner_mask = (R > thresh)

    # Dilate local neighborhood to find local maxima
    local_max = cv2.dilate(R, np.ones((5, 5), np.uint8))
    corner_mask = corner_mask & (R == local_max)

    y_indices, x_indices = np.where(corner_mask)
    corner_coords = list(zip(x_indices.tolist(), y_indices.tolist()))

    for (x, y) in corner_coords:
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), 2, cv2.LINE_AA)

    return R, corner_coords, overlay


def hessian_blob_detector(image: np.ndarray, threshold: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects blob features based on the Determinant of the Hessian matrix:
      det(H) = Ixx * Iyy - Ixy^2
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray_f = gray.astype(np.float32)

    # Second-order derivatives
    Ixx = cv2.Sobel(gray_f, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(gray_f, cv2.CV_32F, 1, 1, ksize=3)

    det_H = (Ixx * Iyy) - (Ixy**2)
    blobs = (det_H > threshold).astype(np.uint8) * 255

    det_norm = cv2.normalize(np.maximum(det_H, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return blobs, det_norm
