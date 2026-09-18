"""
VisionMetrix 3D - Edge Detectors
Module: edge_detectors.py

Covers:
1. Custom Canny Edge Detector:
   - Stage 1: Gaussian Smoothing (Noise suppression)
   - Stage 2: Sobel Gradient Magnitude and Direction Calculation
   - Stage 3: Non-Maximum Suppression (Thinning edges to 1-pixel ridges)
   - Stage 4: Double Thresholding (Strong, Weak, Non-edge classification)
   - Stage 5: Edge Tracking by Hysteresis (Connectivity analysis)
2. Laplacian of Gaussian (LoG) & Zero-Crossing Detection
3. Difference of Gaussians (DoG) Scale-Space Approximation
"""

import cv2
import numpy as np
from typing import Tuple


def canny_edge_detector_custom(image: np.ndarray,
                               low_threshold: float = 50.0,
                               high_threshold: float = 120.0,
                               sigma: float = 1.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handcrafted 5-Stage Canny Edge Detection Pipeline.
    
    Returns:
      edges: Binary edge map (uint8, 0 or 255)
      gradient_magnitude: Continuous gradient magnitude
      gradient_angle: Quantized gradient direction (0, 45, 90, 135 deg)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Stage 1: Gaussian Smoothing
    ksize = int(2 * round(3 * sigma) + 1)
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma).astype(np.float64)

    # Stage 2: Spatial Gradients using Sobel
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

    mag = np.hypot(gx, gy)
    angle_rad = np.arctan2(gy, gx)
    # Convert angle to degrees [0, 180)
    angle_deg = np.rad2deg(angle_rad) % 180.0

    # Stage 3: Non-Maximum Suppression (NMS)
    h, w = gray.shape
    nms = np.zeros((h, w), dtype=np.float64)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            theta = angle_deg[i, j]
            # Quantize gradient direction into 4 primary directions
            # 0 deg (East-West)
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                p1, p2 = mag[i, j + 1], mag[i, j - 1]
            # 45 deg (North-East / South-West)
            elif 22.5 <= theta < 67.5:
                p1, p2 = mag[i + 1, j - 1], mag[i - 1, j + 1]
            # 90 deg (North-South)
            elif 67.5 <= theta < 112.5:
                p1, p2 = mag[i + 1, j], mag[i - 1, j]
            # 135 deg (North-West / South-East)
            else:
                p1, p2 = mag[i - 1, j - 1], mag[i + 1, j + 1]

            if (mag[i, j] >= p1) and (mag[i, j] >= p2):
                nms[i, j] = mag[i, j]
            else:
                nms[i, j] = 0.0

    # Stage 4: Double Thresholding
    res = np.zeros((h, w), dtype=np.uint8)
    weak_val = 75
    strong_val = 255

    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))

    res[strong_i, strong_j] = strong_val
    res[weak_i, weak_j] = weak_val

    # Stage 5: Edge Tracking by Hysteresis
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if res[i, j] == weak_val:
                # If connected to any strong edge pixel in 8-neighborhood, retain as edge
                if (res[i - 1:i + 2, j - 1:j + 2] == strong_val).any():
                    res[i, j] = strong_val
                else:
                    res[i, j] = 0

    return res, mag, angle_deg


def apply_log_edge_detector(image: np.ndarray, sigma: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Laplacian of Gaussian (LoG) operator:
      LoG(x, y) = - (1 / (pi*sigma^4)) * [1 - (x^2 + y^2)/(2*sigma^2)] * exp(-(x^2 + y^2)/(2*sigma^2))
    Detects edges at zero-crossings of the second derivative.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    ksize = int(2 * round(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)

    # Detect zero crossings
    h, w = gray.shape
    zero_crossings = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            patch = lap[i - 1:i + 2, j - 1:j + 2]
            min_val = np.min(patch)
            max_val = np.max(patch)
            if min_val < 0 and max_val > 0 and (max_val - min_val) > 10:
                zero_crossings[i, j] = 255

    lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return zero_crossings, lap_norm


def apply_dog_filter(image: np.ndarray, sigma1: float = 1.0,
                     sigma2: float = 1.6) -> np.ndarray:
    """
    Difference of Gaussians (DoG):
      DoG(x, y) = G(x, y, sigma1) - G(x, y, sigma2)
    Serves as an efficient band-pass filter and approximation of LoG.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

    dog = cv2.subtract(g1, g2)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog_norm
