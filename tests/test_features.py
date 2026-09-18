"""
Unit Tests - Feature Extraction & Segmentation
"""
import pytest
import numpy as np
import cv2

from visionmetrix.features.edge_detectors import (
    canny_edge_detector_custom,
    apply_log_edge_detector,
    apply_dog_filter,
)
from visionmetrix.features.hough import (
    detect_hough_lines,
    detect_hough_circles,
)
from visionmetrix.features.corners import (
    harris_corner_detector,
    hessian_blob_detector,
)
from visionmetrix.features.descriptors import (
    compute_hog_features,
    generate_gabor_filter_bank,
    apply_gabor_texture_analysis,
)
from visionmetrix.features.scale_space import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian_pyramid,
)
from visionmetrix.features.segmentation import (
    region_growing_segmentation,
    otsu_thresholding,
)


@pytest.fixture
def test_canvas():
    canvas = np.zeros((160, 160), dtype=np.uint8)
    # Circle
    cv2.circle(canvas, (80, 80), 30, 220, -1)
    # Lines
    cv2.line(canvas, (20, 20), (140, 20), 255, 3)
    cv2.line(canvas, (20, 140), (140, 140), 255, 3)
    return canvas


def test_edge_detectors(test_canvas):
    edges, mag, ang = canny_edge_detector_custom(test_canvas, low_threshold=40, high_threshold=100)
    assert edges.shape == test_canvas.shape
    assert np.sum(edges > 0) > 0

    zc, lap = apply_log_edge_detector(test_canvas)
    assert zc.shape == test_canvas.shape

    dog = apply_dog_filter(test_canvas)
    assert dog.shape == test_canvas.shape


def test_hough_transforms(test_canvas):
    edges, _, _ = canny_edge_detector_custom(test_canvas, 40, 100)
    lines, overlay_lines = detect_hough_lines(edges, threshold=30, min_line_length=20)
    assert len(lines) > 0
    assert overlay_lines.shape[:2] == test_canvas.shape

    circles, overlay_circles = detect_hough_circles(test_canvas, min_radius=20, max_radius=40)
    assert len(circles) > 0
    assert overlay_circles.shape[:2] == test_canvas.shape


def test_corners_and_scale_space(test_canvas):
    R, corners, overlay = harris_corner_detector(test_canvas)
    assert R.shape == test_canvas.shape
    assert overlay.shape[:2] == test_canvas.shape

    # Pyramids & Exact Reconstruction
    gauss = build_gaussian_pyramid(test_canvas, levels=3)
    assert len(gauss) == 3
    assert gauss[1].shape == (80, 80)

    lap = build_laplacian_pyramid(test_canvas, levels=3)
    recon = reconstruct_from_laplacian_pyramid(lap)
    assert recon.shape == test_canvas.shape
    # Lossless or near-exact reconstruction check
    diff = np.mean(np.abs(test_canvas.astype(float) - recon.astype(float)))
    assert diff < 2.0


def test_segmentation(test_canvas):
    canvas_bimodal = np.where(test_canvas > 0, test_canvas, 40).astype(np.uint8)
    thresh_val, binary = otsu_thresholding(canvas_bimodal)
    assert 0 < thresh_val < 255
    assert binary.shape == canvas_bimodal.shape

    seg = region_growing_segmentation(test_canvas, seeds=[(80, 80)], threshold=20.0)
    assert seg.shape == test_canvas.shape
    assert np.sum(seg > 0) > 0
