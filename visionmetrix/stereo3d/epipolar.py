"""
VisionMetrix 3D - Epipolar Geometry & Multi-View Geometry
Module: epipolar.py

Covers:
- Epipolar Constraint: x'^T * F * x = 0
- Normalized 8-Point Algorithm with rank-2 enforcement (det(F) = 0).
- Essential Matrix E = K'^T * F * K = [t]_x * R
- RANSAC Fundamental Matrix estimation with Sampson distance error.
- Epipolar line computation and visualization.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def normalize_points_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hartley's isotropic point normalization: transforms points so that
    the centroid is at (0, 0) and average distance from origin is sqrt(2).
    Crucial for numerical stability of the 8-point algorithm.
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-7)

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    norm_pts = (T @ pts_homo.T).T
    return norm_pts[:, :2], T


def estimate_fundamental_8point(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Normalized 8-point algorithm for Fundamental Matrix F.
    Enforces the algebraic epipolar constraint:
      u2*u1*f11 + u2*v1*f12 + u2*f13 + v2*u1*f21 + v2*v1*f22 + v2*f23 + u1*f31 + v1*f32 + f33 = 0
    Singular Value Decomposition enforces the rank-2 singularity constraint det(F) = 0.
    """
    n = pts1.shape[0]
    if n < 8:
        raise ValueError("8-point algorithm requires at least 8 point correspondences.")

    norm1, T1 = normalize_points_2d(pts1.astype(np.float64))
    norm2, T2 = normalize_points_2d(pts2.astype(np.float64))

    # Construct linear system Af = 0
    A = np.zeros((n, 9), dtype=np.float64)
    for i in range(n):
        x1, y1 = norm1[i]
        x2, y2 = norm2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1, :].reshape((3, 3))

    # Enforce rank-2 condition: project onto manifold of singular 3x3 matrices
    U, S, Vt_f = np.linalg.svd(F_norm)
    S[2] = 0.0
    F_rank2 = U @ np.diag(S) @ Vt_f

    # De-normalize: F = T2^T * F_norm * T1
    F = T2.T @ F_rank2 @ T1
    if abs(F[2, 2]) > 1e-9:
        F = F / F[2, 2]
    return F


def compute_sampson_distance(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Computes first-order geometric approximation (Sampson distance) for epipolar error:
      d_S = (x2^T * F * x1)^2 / ( (Fx1)_1^2 + (Fx1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2 )
    """
    n = pts1.shape[0]
    p1 = np.hstack([pts1, np.ones((n, 1))])
    p2 = np.hstack([pts2, np.ones((n, 1))])

    Fx1 = (F @ p1.T).T
    Ftx2 = (F.T @ p2.T).T

    numerator = np.sum(p2 * Fx1, axis=1)**2
    denominator = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    return numerator / np.maximum(denominator, 1e-8)


def estimate_fundamental_ransac(pts1: np.ndarray, pts2: np.ndarray,
                                threshold_px: float = 1.5,
                                max_iters: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust Fundamental Matrix estimation using RANSAC (RANdom SAmple Consensus).
    Rejects erroneous feature matches (outliers) caused by repetitive textures or glare.
    """
    n = pts1.shape[0]
    if n < 8:
        raise ValueError("Requires at least 8 correspondences for RANSAC.")

    best_inliers = np.zeros(n, dtype=bool)
    best_F = None
    max_inlier_count = 0

    thresh_sq = threshold_px**2

    for _ in range(max_iters):
        sample_indices = np.random.choice(n, 8, replace=False)
        try:
            F_candidate = estimate_fundamental_8point(pts1[sample_indices], pts2[sample_indices])
            sampson_dist = compute_sampson_distance(pts1, pts2, F_candidate)
            inliers = sampson_dist < thresh_sq
            inlier_count = np.sum(inliers)

            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                best_inliers = inliers
                best_F = F_candidate
        except Exception:
            continue

    # Re-estimate F using all consensus inliers for maximum accuracy
    if max_inlier_count >= 8:
        best_F = estimate_fundamental_8point(pts1[best_inliers], pts2[best_inliers])

    return best_F, best_inliers


def compute_essential_matrix(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """
    Computes Essential Matrix E = K2^T * F * K1.
    Encodes pure extrinsic geometric relationships between calibrated cameras.
    Enforces singular values [sigma, sigma, 0].
    """
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    sigma = (S[0] + S[1]) / 2.0
    E_calibrated = U @ np.diag([sigma, sigma, 0.0]) @ Vt
    return E_calibrated


def compute_epipolar_lines(points: np.ndarray, which_image: int, F: np.ndarray) -> np.ndarray:
    """
    Calculates corresponding epipolar lines:
      If which_image == 1 (points in img1), line in img2: l2 = F * x1
      If which_image == 2 (points in img2), line in img1: l1 = F^T * x2
    Line vector [a, b, c] represents line ax + by + c = 0.
    """
    n = points.shape[0]
    pts_homo = np.hstack([points, np.ones((n, 1), dtype=np.float64)])
    if which_image == 1:
        lines = (F @ pts_homo.T).T
    else:
        lines = (F.T @ pts_homo.T).T

    # Normalize line coefficients so a^2 + b^2 = 1
    scale = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
    scale[scale < 1e-8] = 1.0
    lines = lines / scale[:, np.newaxis]
    return lines
