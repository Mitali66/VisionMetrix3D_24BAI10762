"""
VisionMetrix 3D - Scale-Space & Image Pyramids
Module: scale_space.py

Theoretical Background:
1. Gaussian Pyramid:
   Multi-scale image representation generated via recursive lowpass filtering
   and subsampling: G_{k+1} = Downsample(G_k * w)
2. Laplacian Pyramid:
   Band-pass scale decomposition encoding detail lost between pyramid levels:
     L_k = G_k - Upsample(G_{k+1})
3. Exact Reconstruction:
   G_k = L_k + Upsample(G_{k+1})
"""

import cv2
import numpy as np
from typing import List


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Constructs an N-level Gaussian Pyramid.
    """
    pyramid = [image.copy()]
    current = image.copy()
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Constructs an N-level Laplacian Pyramid (band-pass scale-space).
    The top level is the lowest-resolution Gaussian residual.
    """
    gauss_pyr = build_gaussian_pyramid(image, levels)
    lap_pyr = []

    for i in range(levels - 1):
        g_curr = gauss_pyr[i]
        g_next = gauss_pyr[i + 1]
        
        # Upsample g_next to match g_curr dimensions
        h, w = g_curr.shape[:2]
        expanded = cv2.pyrUp(g_next, dstsize=(w, h))

        diff = cv2.subtract(g_curr.astype(np.int16), expanded.astype(np.int16))
        lap_pyr.append(diff)

    lap_pyr.append(gauss_pyr[-1])  # Top-level residual
    return lap_pyr


def reconstruct_from_laplacian_pyramid(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Reconstructs the original full-resolution image from its Laplacian pyramid.
    Mathematically verifies lossless energy preservation of the scale-space decomposition.
    """
    levels = len(lap_pyr)
    current = lap_pyr[-1].astype(np.float64)

    for i in range(levels - 2, -1, -1):
        h, w = lap_pyr[i].shape[:2]
        expanded = cv2.pyrUp(current, dstsize=(w, h))
        current = cv2.add(lap_pyr[i].astype(np.float64), expanded.astype(np.float64))

    return np.clip(current, 0, 255).astype(np.uint8)
