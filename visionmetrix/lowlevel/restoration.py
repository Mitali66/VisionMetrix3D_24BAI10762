"""
VisionMetrix 3D - Image Restoration & Deconvolution
Module: restoration.py

Degradation model: g(x, y) = f(x, y) * h(x, y) + n(x, y)
Where h(x, y) is the Point Spread Function (PSF) of motion blur and n is additive Gaussian noise.
Implements:
1. Motion blur PSF synthesis based on motion length and direction.
2. Wiener Filter deconvolution minimizing expected mean squared error:
     F_hat(u, v) = [ H*(u, v) / (|H(u, v)|^2 + K) ] * G(u, v)
3. Inverse Filtering with low-pass / threshold regularization.
"""

import cv2
import numpy as np
from typing import Tuple


def create_motion_blur_psf(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """
    Generates a 2D Point Spread Function (PSF) modeling linear camera motion blur.
    """
    psf = np.zeros((length, length), dtype=np.float64)
    center = (length - 1) / 2.0
    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    for i in range(length):
        offset = i - center
        x = int(round(center + offset * cos_a))
        y = int(round(center + offset * sin_a))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1.0

    total = np.sum(psf)
    if total > 0:
        psf /= total
    else:
        psf[int(center), int(center)] = 1.0
    return psf


def apply_motion_blur(image: np.ndarray, length: int = 15, angle_deg: float = 0.0,
                      noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Degrades an image by convolving with a motion blur PSF and adding Gaussian sensor noise.
    """
    psf = create_motion_blur_psf(length, angle_deg)
    blurred = cv2.filter2D(image, -1, psf)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std * 255.0, image.shape).astype(np.float64)
        degraded = np.clip(blurred.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    else:
        degraded = blurred

    return degraded, psf


def wiener_deconvolution(blurred_image: np.ndarray, psf: np.ndarray, snr: float = 100.0) -> np.ndarray:
    """
    Wiener Filter Deconvolution.
    Minimizes the expected mean square error between the estimated image and the ground truth:
      W(u, v) = H*(u, v) / (|H(u, v)|^2 + 1 / SNR)
    Where SNR is the Signal-to-Noise Ratio (P_signal / P_noise).
    """
    is_color = (len(blurred_image.shape) == 3)
    if not is_color:
        channels = [blurred_image.astype(np.float64)]
    else:
        channels = [blurred_image[:, :, c].astype(np.float64) for c in range(3)]

    h, w = blurred_image.shape[:2]

    # Pad PSF to full image dimensions for DFT
    psf_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf

    # Circular shift PSF to align origin at (0, 0)
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H)**2

    k_factor = 1.0 / max(snr, 1e-6)
    W = H_conj / (H_mag_sq + k_factor)

    restored_channels = []
    for ch in channels:
        G = np.fft.fft2(ch)
        F_hat = G * W
        f_restored = np.fft.ifft2(F_hat)
        restored = np.real(f_restored)
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        restored_channels.append(restored)

    if not is_color:
        return restored_channels[0]
    else:
        return np.stack(restored_channels, axis=2)


def inverse_filtering(blurred_image: np.ndarray, psf: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Direct Inverse Filtering with low-frequency regularization threshold:
      F_hat(u, v) = G(u, v) / H(u, v) if |H(u, v)| >= threshold else G(u, v)
    Prevents catastrophic noise amplification when H(u, v) approaches zero.
    """
    is_color = (len(blurred_image.shape) == 3)
    if not is_color:
        channels = [blurred_image.astype(np.float64)]
    else:
        channels = [blurred_image[:, :, c].astype(np.float64) for c in range(3)]

    h, w = blurred_image.shape[:2]
    psf_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    H = np.fft.fft2(psf_padded)
    H_mag = np.abs(H)

    inv_H = np.zeros_like(H, dtype=np.complex128)
    valid = H_mag >= threshold
    inv_H[valid] = 1.0 / H[valid]
    inv_H[~valid] = 0.0

    restored_channels = []
    for ch in channels:
        G = np.fft.fft2(ch)
        F_hat = G * inv_H
        f_restored = np.fft.ifft2(F_hat)
        restored = np.real(f_restored)
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        restored_channels.append(restored)

    if not is_color:
        return restored_channels[0]
    else:
        return np.stack(restored_channels, axis=2)
