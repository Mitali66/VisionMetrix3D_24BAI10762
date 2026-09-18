"""
Unit Tests - Low-Level Image Processing
"""
import pytest
import numpy as np
import cv2

from visionmetrix.lowlevel.transformations import (
    apply_orthogonal,
    apply_euclidean,
    apply_affine,
    estimate_homography_dlt,
    apply_projective,
)
from visionmetrix.lowlevel.filtering import (
    convolve2d_manual,
    apply_gaussian_blur,
    apply_sobel,
    apply_laplacian,
    apply_bilateral,
)
from visionmetrix.lowlevel.fourier import (
    compute_fft2d,
    apply_ideal_filter,
    apply_butterworth_filter,
    remove_periodic_noise,
)
from visionmetrix.lowlevel.enhancement import (
    global_histogram_equalization,
    clahe_enhancement,
    gamma_correction,
    histogram_specification,
)
from visionmetrix.lowlevel.restoration import (
    create_motion_blur_psf,
    apply_motion_blur,
    wiener_deconvolution,
    inverse_filtering,
)


@pytest.fixture
def sample_image():
    # 128x128 synthetic test pattern
    img = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (98, 98), 200, -1)
    cv2.circle(img, (64, 64), 20, 50, -1)
    return img


def test_orthogonal_transformations(sample_image):
    rot90 = apply_orthogonal(sample_image, "rot90")
    assert rot90.shape == sample_image.shape
    fliph = apply_orthogonal(sample_image, "flip_h")
    assert fliph.shape == sample_image.shape


def test_euclidean_and_affine(sample_image):
    warped_euc, M_euc = apply_euclidean(sample_image, angle_deg=30.0, tx=5.0, ty=-5.0)
    assert warped_euc.shape == sample_image.shape
    assert M_euc.shape == (2, 3)

    src_pts = np.array([[10, 10], [100, 10], [10, 100]], dtype=np.float32)
    dst_pts = np.array([[15, 12], [95, 20], [18, 90]], dtype=np.float32)
    warped_aff, M_aff = apply_affine(sample_image, src_pts, dst_pts)
    assert warped_aff.shape == sample_image.shape
    assert M_aff.shape == (2, 3)


def test_dlt_homography(sample_image):
    src_pts = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float64)
    # Apply synthetic perspective warp to points
    dst_pts = np.array([[15, 20], [90, 12], [110, 95], [5, 105]], dtype=np.float64)
    H = estimate_homography_dlt(src_pts, dst_pts)
    assert H.shape == (3, 3)
    warped, _ = apply_projective(sample_image, H)
    assert warped.shape == sample_image.shape


def test_spatial_convolution_and_filtering(sample_image):
    # Test manual convolution against simple identity / averaging kernel
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    manual_conv = convolve2d_manual(sample_image, kernel)
    assert manual_conv.shape == sample_image.shape

    # Sobel
    mag, gx, gy, ang = apply_sobel(sample_image)
    assert mag.shape == sample_image.shape
    assert np.max(mag) > 0

    # Laplacian
    lap = apply_laplacian(sample_image)
    assert lap.shape == sample_image.shape

    # Bilateral
    bilateral = apply_bilateral(sample_image, d=5, sigma_color=50, sigma_space=50)
    assert bilateral.shape == sample_image.shape


def test_fourier_filtering(sample_image):
    f_shift, mag_spec, phase_spec = compute_fft2d(sample_image)
    assert f_shift.shape == sample_image.shape
    assert mag_spec.shape == sample_image.shape

    # Ideal filter
    lp_ideal, _ = apply_ideal_filter(sample_image, cutoff_freq=25.0, filter_type="lowpass")
    assert lp_ideal.shape == sample_image.shape

    # Butterworth filter
    lp_butter, _ = apply_butterworth_filter(sample_image, cutoff_freq=25.0, order=2, filter_type="lowpass")
    assert lp_butter.shape == sample_image.shape

    # Periodic notch noise removal
    restored_notch, _ = remove_periodic_noise(sample_image, [(15, 15)])
    assert restored_notch.shape == sample_image.shape


def test_enhancement_and_restoration(sample_image):
    eq = global_histogram_equalization(sample_image)
    assert eq.shape == sample_image.shape

    clahe = clahe_enhancement(sample_image)
    assert clahe.shape == sample_image.shape

    gamma = gamma_correction(sample_image, gamma=1.8)
    assert gamma.shape == sample_image.shape

    # Degradation and Restoration
    degraded, psf = apply_motion_blur(sample_image, length=11, angle_deg=0.0, noise_std=0.005)
    assert degraded.shape == sample_image.shape

    restored_wiener = wiener_deconvolution(degraded, psf, snr=50.0)
    assert restored_wiener.shape == sample_image.shape

    restored_inv = inverse_filtering(degraded, psf, threshold=0.1)
    assert restored_inv.shape == sample_image.shape
