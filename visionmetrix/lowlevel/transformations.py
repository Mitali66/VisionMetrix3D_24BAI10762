"""
VisionMetrix 3D - Low-Level Transformations
Module: transformations.py

Implements 2D coordinate transformations representing camera and surface motions:
1. Orthogonal: Rigid rotations by multiples of 90 degrees and mirror reflections.
2. Euclidean (Isometry): Preserves Euclidean distances and angles (rotation + translation).
3. Affine: Preserves parallelism of lines, lines remain lines (rotation, translation, scale, shear).
4. Projective (Homography): 8 degrees of freedom; models perspective projection between two planes.
Includes Direct Linear Transformation (DLT) using SVD to solve planar homography matrices.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def apply_orthogonal(image: np.ndarray, mode: str = "rot90") -> np.ndarray:
    """
    Applies an orthogonal transformation (det = +1 for rotation, det = -1 for reflection).
    Preserves all Euclidean distances and inner products.
    """
    if mode == "rot90":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif mode == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    elif mode == "rot270":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif mode == "flip_h":
        return cv2.flip(image, 1)
    elif mode == "flip_v":
        return cv2.flip(image, 0)
    else:
        raise ValueError(f"Unknown orthogonal mode: {mode}. Choose rot90, rot180, rot270, flip_h, or flip_v.")


def apply_euclidean(image: np.ndarray, angle_deg: float, tx: float, ty: float,
                    scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 2D Euclidean (Rigid-body) transformation:
    [x']   [cos(theta) -sin(theta)  tx] [x]
    [y'] = [sin(theta)  cos(theta)  ty] [y]
    [1 ]   [    0           0        1] [1]
    
    Preserves distances, angles, and object area (when scale=1.0).
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return warped, M


def apply_affine(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies an Affine transformation (6 degrees of freedom).
    Determined uniquely from 3 non-collinear corresponding points.
    Preserves line collinearity and parallelism of parallel lines.
    """
    h, w = image.shape[:2]
    M = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    return warped, M


def estimate_homography_dlt(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Computes the 3x3 Projective Homography matrix H using the Direct Linear Transformation (DLT).
    
    Given n >= 4 point pairs (x_i, y_i) <-> (u_i, v_i), each pair generates two linear constraints:
      [-x, -y, -1,  0,  0,  0,  u*x, u*y, u] [h] = 0
      [ 0,  0,  0, -x, -y, -1,  v*x, v*y, v] [h] = 0
      
    We assemble matrix A of shape (2n, 9) and compute the right nullspace via SVD:
      A = U * S * V^T, where h is the last row of V^T corresponding to the smallest singular value.
    """
    n = src_pts.shape[0]
    if n < 4:
        raise ValueError("DLT requires at least 4 point correspondences to solve for 8 DOFs.")

    # Data normalization for numerical stability (Hartley normalization)
    def normalize_points(pts):
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
        if mean_dist < 1e-7:
            scale = 1.0
        else:
            scale = np.sqrt(2.0) / mean_dist
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        norm_pts = (T @ pts_homo.T).T
        return norm_pts[:, :2], T

    norm_src, T_src = normalize_points(src_pts.astype(np.float64))
    norm_dst, T_dst = normalize_points(dst_pts.astype(np.float64))

    A = []
    for i in range(n):
        x, y = norm_src[i, 0], norm_src[i, 1]
        u, v = norm_dst[i, 0], norm_dst[i, 1]
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)

    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    h_norm = Vt[-1, :].reshape((3, 3))

    # De-normalize: H = T_dst^-1 * H_norm * T_src
    H = np.linalg.inv(T_dst) @ h_norm @ T_src
    if abs(H[2, 2]) > 1e-9:
        H = H / H[2, 2]
    return H


def apply_projective(image: np.ndarray, H: Optional[np.ndarray] = None,
                     src_quad: Optional[np.ndarray] = None,
                     dst_quad: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a Projective Transformation (Planar Homography).
    Maps lines to lines, but angles and parallelisms are generally not preserved.
    Crucial for camera perspective rectification in industrial metrology.
    """
    h, w = image.shape[:2]
    if H is None:
        if src_quad is None or dst_quad is None:
            raise ValueError("Must provide either a precomputed 3x3 H matrix or both src_quad and dst_quad.")
        H = estimate_homography_dlt(src_quad, dst_quad)

    warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
    return warped, H
