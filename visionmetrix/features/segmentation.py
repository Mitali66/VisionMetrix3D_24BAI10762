"""
VisionMetrix 3D - Image Segmentation & Surface Defect Extraction
Module: segmentation.py

Covers:
1. Region Growing Segmentation:
   Seeds expand based on statistical homogeneity criteria |I(p) - mean| < threshold.
2. Mean-Shift Clustering Segmentation:
   Non-parametric mode seeking in joint spatial-color domain (x, y, L, a, b).
3. Otsu's Global Adaptive Thresholding:
   Maximizes between-class variance sigma_B^2 = w0 * w1 * (mu0 - mu1)^2.
4. Graph-Cut / GrabCut Segmentation:
   Energy minimization on a pixel adjacency graph for precise defect boundary delineation.
"""

import cv2
import numpy as np
from typing import List, Tuple


def region_growing_segmentation(image: np.ndarray,
                                seeds: List[Tuple[int, int]],
                                threshold: float = 15.0) -> np.ndarray:
    """
    Segment connected surface defect regions using 4-connected Region Growing.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    segmented = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    # 4-connected neighbor offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for seed in seeds:
        sx, sy = seed
        if not (0 <= sx < w and 0 <= sy < h) or visited[sy, sx]:
            continue

        seed_value = float(gray[sy, sx])
        queue = [(sx, sy)]
        visited[sy, sx] = True
        segmented[sy, sx] = 255

        while queue:
            cx, cy = queue.pop(0)
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if abs(float(gray[ny, nx]) - seed_value) <= threshold:
                        segmented[ny, nx] = 255
                        queue.append((nx, ny))

    return segmented


def mean_shift_segmentation(image: np.ndarray,
                            spatial_radius: int = 15,
                            color_radius: int = 25) -> np.ndarray:
    """
    Applies Mean-Shift color-spatial segmentation (cv2.pyrMeanShiftFiltering).
    Locates local modes of density distribution to group homogeneous material facets.
    """
    if len(image.shape) == 2:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()

    segmented = cv2.pyrMeanShiftFiltering(img_color, sp=spatial_radius, sr=color_radius)
    return segmented


def otsu_thresholding(image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Otsu's Global Adaptive Thresholding from first principles.
    Finds the optimal threshold t* that maximizes between-class variance:
      sigma_b^2(t) = w0(t) * w1(t) * [mu0(t) - mu1(t)]^2
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    total = gray.size
    p = hist / max(total, 1)

    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 1e-6
    sigma_b_sq = np.zeros(256)
    sigma_b_sq[valid] = ((mu_total * omega[valid] - mu[valid])**2) / denom[valid]

    best_thresh = float(np.argmax(sigma_b_sq))
    binary = (gray > best_thresh).astype(np.uint8) * 255

    return best_thresh, binary


def grabcut_defect_segmentation(image: np.ndarray,
                                bounding_box: Tuple[int, int, int, int],
                                iter_count: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Graph-Cut / GrabCut Defect Segmentation.
    Constructs a Markov Random Field (MRF) graph and computes the minimum cut:
      E(alpha, k, theta, z) = U(alpha, k, theta, z) + V(alpha, z)
    Where U represents regional data likelihood (GMMs) and V represents boundary smoothness.
    
    bounding_box: (x, y, width, height) enclosing the candidate defect area.
    """
    if len(image.shape) == 2:
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_color = image.copy()

    h, w = img_color.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_color, mask, bounding_box, bgd_model, fgd_model,
                iter_count, cv2.GC_INIT_WITH_RECT)

    # Defect pixels are marked as GC_FGD (1) or GC_PR_FGD (3)
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    segmented_defect = cv2.bitwise_and(img_color, img_color, mask=binary_mask)

    return binary_mask, segmented_defect
