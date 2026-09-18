"""
Unit Tests - Depth Estimation & Multi-Camera Views
"""
import pytest
import numpy as np
import os

from visionmetrix.stereo3d.calibration import (
    calibrate_camera_dlt,
    decompose_projection_matrix,
    compute_reprojection_error,
)
from visionmetrix.stereo3d.epipolar import (
    estimate_fundamental_8point,
    estimate_fundamental_ransac,
    compute_sampson_distance,
    compute_epipolar_lines,
)
from visionmetrix.stereo3d.disparity import (
    compute_disparity_sgbm,
    disparity_to_depth,
)
from visionmetrix.stereo3d.reconstruction import (
    triangulate_point_cloud,
    save_point_cloud_ply,
)
from visionmetrix.stereo3d.photometric_stereo import (
    solve_photometric_stereo,
    integrate_surface_frankot_chellappa,
)
from visionmetrix.data.sample_generator import generate_photometric_stereo_series


def test_camera_calibration_dlt():
    # 8 synthetic 3D world points
    pts_3d = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    # Known synthetic projection matrix
    K_true = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
    Rt_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2]], dtype=np.float64)
    P_true = K_true @ Rt_true

    # Project to 2D
    pts_3d_h = np.hstack([pts_3d, np.ones((8, 1))])
    proj = (P_true @ pts_3d_h.T).T
    pts_2d = proj[:, :2] / proj[:, 2:3]

    P_estimated = calibrate_camera_dlt(pts_3d, pts_2d)
    metrics = compute_reprojection_error(pts_3d, pts_2d, P_estimated)
    assert metrics["rmse_pixels"] < 1e-4

    K, R, t = decompose_projection_matrix(P_estimated)
    assert K.shape == (3, 3)
    assert R.shape == (3, 3)
    assert t.shape == (3, 1)


def test_epipolar_geometry():
    np.random.seed(42)
    pts1 = np.random.uniform(50, 400, (12, 2))
    # Synthetic horizontal disparity + slight vertical parallax
    pts2 = pts1.copy()
    pts2[:, 0] -= 25.0
    pts2[:, 1] += np.random.normal(0, 0.2, len(pts1))

    F = estimate_fundamental_8point(pts1, pts2)
    assert F.shape == (3, 3)
    # Check rank-2 condition: det(F) should be near 0
    assert abs(np.linalg.det(F)) < 1e-6

    F_ransac, inliers = estimate_fundamental_ransac(pts1, pts2, threshold_px=2.0)
    assert F_ransac is not None
    assert np.sum(inliers) >= 8

    lines1 = compute_epipolar_lines(pts2[:5], which_image=2, F=F)
    assert lines1.shape == (5, 3)


def test_disparity_and_reconstruction(tmp_path):
    left = np.zeros((120, 160), dtype=np.uint8)
    left[40:80, 50:110] = 180
    right = np.zeros((120, 160), dtype=np.uint8)
    right[40:80, 34:94] = 180  # 16px horizontal disparity

    disp = compute_disparity_sgbm(left, right, num_disparities=32, block_size=5)
    depth = disparity_to_depth(disp, focal_length_px=400.0, baseline_meters=0.1)
    assert depth.shape == left.shape

    pts, cols = triangulate_point_cloud(depth, left, focal_length=400.0, principal_point=(80.0, 60.0))
    assert len(pts) == len(cols)

    ply_path = os.path.join(tmp_path, "test.ply")
    save_point_cloud_ply(ply_path, pts[:10], cols[:10])
    assert os.path.exists(ply_path)


def test_photometric_stereo():
    images, light_vecs, _ = generate_photometric_stereo_series(size=64)
    normal_map, albedo_map = solve_photometric_stereo(images, light_vecs)

    assert normal_map.shape == (64, 64, 3)
    assert albedo_map.shape == (64, 64)

    # Reconstruct surface elevation via Frankot-Chellappa
    height_map = integrate_surface_frankot_chellappa(normal_map)
    assert height_map.shape == (64, 64)
