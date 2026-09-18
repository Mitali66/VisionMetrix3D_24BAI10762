"""
VisionMetrix 3D - Low-Level Spatial Filtering & Convolution
Module: filtering.py

Implements spatial convolution from fundamental mathematical principles and provides
standard spatial operators for industrial noise reduction and structural edge extraction.
"""

import cv2
import numpy as np
from typing import Tuple


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray, padding_mode: str = "reflect") -> np.ndarray:
    """
    Applies 2D spatial convolution from first principles:
      (I * K)(x, y) = sum_i sum_j I(x - i, y - j) * K(i, j)
    Flips the kernel by 180 degrees (correlation vs convolution distinction),
    pads the input image, and slides the kernel over all spatial coordinates.
    """
    is_color = (len(image.shape) == 3)
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Kernel flipping for true mathematical convolution
    kernel_flipped = np.flip(np.flip(kernel, axis=0), axis=1)

    if not is_color:
        channels = [image]
    else:
        channels = [image[:, :, c] for c in range(image.shape[2])]

    filtered_channels = []
    for ch in channels:
        if padding_mode == "reflect":
            padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        else:
            padded = np.pad(ch, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)

        h, w = ch.shape
        out = np.zeros((h, w), dtype=np.float64)

        # Efficient sliding window convolution using strided views or vectorized dot products
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k_h, j:j + k_w]
                out[i, j] = np.sum(region * kernel_flipped)

        filtered_channels.append(out)

    if not is_color:
        result = filtered_channels[0]
    else:
        result = np.stack(filtered_channels, axis=2)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                        sigma: float = 1.0) -> np.ndarray:
    """
    Isotropic 2D Gaussian smoothing filter:
      G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    Suppresses high-frequency sensor noise while preserving low-frequency structural details.
    """
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)


def apply_sobel(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes first-order spatial derivatives using 3x3 Sobel operators:
      S_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
      S_y = [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]]
      
    Returns:
      magnitude: sqrt(G_x^2 + G_y^2) normalized to [0, 255]
      grad_x: Raw horizontal gradient (float32)
      grad_y: Raw vertical gradient (float32)
      angle: Gradient orientation in radians [-pi, pi]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)

    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_norm, grad_x, grad_y, angle


def apply_laplacian(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Computes the second-order derivative (Laplacian operator):
      del^2 I = (d^2 I / dx^2) + (d^2 I / dy^2)
    Highlights rapid intensity discontinuities and zero-crossings corresponding to edge centers.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel_size)
    lap_abs = np.absolute(lap)
    lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return lap_norm


def apply_bilateral(image: np.ndarray, d: int = 9,
                    sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Non-linear bilateral filter for edge-preserving smoothing.
    Combines spatial Gaussian domain kernel with radiometric range kernel:
      W(p, q) = G_sigma_s(||p - q||) * G_sigma_r(|I(p) - I(q)|)
    Smooths micro-textures while preserving sharp defect boundaries.
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
