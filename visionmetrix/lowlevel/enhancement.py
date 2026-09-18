"""
VisionMetrix 3D - Radiometric Image Enhancement
Module: enhancement.py

Covers global histogram equalization, Contrast Limited Adaptive Histogram Equalization
(CLAHE), power-law / gamma radiometric corrections, and histogram matching/specification.
"""

import cv2
import numpy as np
from typing import Tuple


def global_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Flattens the probability density function (PDF) and spreads dynamic range
    using the Cumulative Distribution Function (CDF):
      s_k = T(r_k) = (L - 1) * sum_{j=0}^k p_r(r_j)
    For color images, performs equalization on luminance (Y or L) to preserve chromaticity.
    """
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    
    # Color image: transform to YCrCb space and equalize Y channel
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Divides the image into contextual tiles, computes localized histograms,
    clips the histogram to prevent over-amplification of noise, and interpolates
    bilinearly across tile boundaries to eliminate artificial seams.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(image.shape) == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Power-Law (Gamma) radiometric transformation:
      s = c * r^gamma, where r in [0, 1]
    - gamma < 1: Expands dark regions, brightens shadows (under-exposed inspection)
    - gamma > 1: Compresses high intensities, suppresses glare from specular surfaces
    """
    inv_gamma = 1.0 / max(gamma, 1e-4)
    # Precompute lookup table for high-throughput real-time performance
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def histogram_specification(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Histogram Specification (Matching).
    Transforms the source image so that its histogram matches the target reference profile.
    Critical for standardizing images captured across multiple camera stations with varying exposures.
    """
    if len(source.shape) == 3:
        src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = source.copy()

    if len(reference.shape) == 3:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference.copy()

    # Compute CDF of source
    src_hist, _ = np.histogram(src_gray.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum()
    src_cdf_normalized = src_cdf / (src_cdf[-1] + 1e-8)

    # Compute CDF of reference
    ref_hist, _ = np.histogram(ref_gray.flatten(), 256, [0, 256])
    ref_cdf = ref_hist.cumsum()
    ref_cdf_normalized = ref_cdf / (ref_cdf[-1] + 1e-8)

    # Build mapping table: for each source gray level, find closest match in ref CDF
    mapping = np.zeros(256, dtype=np.uint8)
    for src_val in range(256):
        diff = np.abs(ref_cdf_normalized - src_cdf_normalized[src_val])
        mapping[src_val] = np.argmin(diff)

    if len(source.shape) == 2:
        return cv2.LUT(source, mapping)
    else:
        ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], mapping)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
