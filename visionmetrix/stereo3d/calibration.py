"""
VisionMetrix 3D - Camera Calibration & DLT
Module: calibration.py

Mathematical Formulation:
Direct Linear Transformation (DLT) solves the 3x4 Camera Projection Matrix P:
  s * [u, v, 1]^T = P * [X, Y, Z, 1]^T
Where P = K * [R | t].
Uses RQ decomposition to decouple intrinsic calibration matrix K from extrinsic
rotation matrix R and camera translation vector t.
"""

import numpy as np
import scipy.linalg
from typing import Tuple, Dict


def calibrate_camera_dlt(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """
    Estimates 3x4 Projection Matrix P from N >= 6 known (3D, 2D) correspondence pairs.
    
    Each 3D-2D correspondence (X, Y, Z) <-> (u, v) generates 2 linear equations:
      [X, Y, Z, 1,  0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u] [p] = 0
      [0, 0, 0, 0,  X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v] [p] = 0
    """
    n = points_3d.shape[0]
    if n < 6:
        raise ValueError("DLT camera calibration requires at least 6 non-coplanar 3D-2D correspondences.")

    A = []
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A.append([X, Y, Z, 1.0, 0.0, 0.0, 0.0, 0.0, -u * X, -u * Y, -u * Z, -u])
        A.append([0.0, 0.0, 0.0, 0.0, X, Y, Z, 1.0, -v * X, -v * Y, -v * Z, -v])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1, :].reshape((3, 4))
    P = P / P[2, 3]  # Normalize scale
    return P


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decomposes 3x4 Projection Matrix P = [M | p4] into:
      K: 3x3 Intrinsic matrix (Upper Triangular)
      R: 3x3 Rotation matrix (Orthogonal, det(R) = +1)
      t: 3x1 Translation vector
    Uses RQ decomposition of the first 3x3 submatrix M.
    """
    M = P[:, :3]
    p4 = P[:, 3]

    K, R = scipy.linalg.rq(M)

    # Ensure diagonal elements of intrinsic matrix K are positive
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Normalize K such that K[2, 2] == 1.0
    if abs(K[2, 2]) > 1e-8:
        scale = K[2, 2]
        K = K / scale
        p4 = p4 / scale

    # Check for right-handed coordinate system (det(R) == +1)
    if np.linalg.det(R) < 0:
        R = -R
        p4 = -p4

    t = np.linalg.inv(K) @ p4
    return K, R, t.reshape((3, 1))


def compute_reprojection_error(points_3d: np.ndarray, points_2d: np.ndarray,
                               P: np.ndarray) -> Dict[str, float]:
    """
    Calculates root-mean-square (RMSE) and maximum Euclidean reprojection error in pixels:
      e_i = || x_i - (P * X_i) / (P_3 * X_i) ||
    """
    n = points_3d.shape[0]
    points_3d_homo = np.hstack([points_3d, np.ones((n, 1), dtype=np.float64)])
    
    projected_homo = (P @ points_3d_homo.T).T
    z = projected_homo[:, 2:3]
    z[np.abs(z) < 1e-8] = 1e-8  # Prevent division by zero
    projected_2d = projected_homo[:, :2] / z

    errors = np.linalg.norm(points_2d - projected_2d, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))

    return {"rmse_pixels": rmse, "max_error_pixels": max_err, "mean_error_pixels": mean_err}
