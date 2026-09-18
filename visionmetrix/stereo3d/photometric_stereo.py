"""
VisionMetrix 3D - Photometric Stereo & Surface Normal Estimation
Module: photometric_stereo.py

Implements Woodham's Classical Photometric Stereo algorithm (Shape from Shading):
Given m >= 3 images captured from a single fixed camera viewpoint under varying
calibrated light source directions L_1, L_2, ..., L_m:
  I_k(x, y) = rho(x, y) * (L_k . N(x, y))
Solving the linear least-squares system:
  I = L * g, where g = rho * N
Yields surface albedo rho(x, y) = ||g|| and surface normal vector N(x, y) = g / ||g||.

Surface Height Integration:
Reconstructs the continuous 3D elevation profile Z(x, y) from gradients p = -Nx/Nz, q = -Ny/Nz
using the Frankot-Chellappa Fourier domain integration method.
"""

import cv2
import numpy as np
from typing import List, Tuple


def solve_photometric_stereo(images: List[np.ndarray],
                             light_sources: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates surface normal map N(x, y) and albedo map rho(x, y).
    
    images: List of m grayscale images (all same dimensions)
    light_sources: (m, 3) matrix of normalized light vector directions [Lx, Ly, Lz]
    
    Returns:
      normal_map: (H, W, 3) normalized surface normals [-1, 1]
      albedo_map: (H, W) diffuse surface reflectance [0, 255]
    """
    m = len(images)
    if m < 3:
        raise ValueError("Photometric Stereo requires at least 3 distinct illumination directions.")

    h, w = images[0].shape[:2]
    I = np.zeros((m, h * w), dtype=np.float64)

    for k in range(m):
        if len(images[k].shape) == 3:
            gray = cv2.cvtColor(images[k], cv2.COLOR_BGR2GRAY)
        else:
            gray = images[k]
        I[k, :] = gray.reshape(-1).astype(np.float64) / 255.0

    L = light_sources.astype(np.float64)  # shape (m, 3)
    # Normalize light direction vectors
    L_norms = np.linalg.norm(L, axis=1, keepdims=True)
    L = L / np.maximum(L_norms, 1e-8)

    # Solve L * g = I via Moore-Penrose pseudo-inverse: g = (L^T L)^-1 L^T I
    L_pinv = np.linalg.pinv(L)  # shape (3, m)
    g = L_pinv @ I              # shape (3, H*W)

    # Albedo rho = ||g||
    rho = np.linalg.norm(g, axis=0)  # shape (H*W,)
    
    # Surface normal N = g / ||g||
    valid_mask = rho > 1e-6
    N = np.zeros_like(g)
    N[:, valid_mask] = g[:, valid_mask] / rho[valid_mask]
    # Default upward normal [0, 0, 1] for shadowed/dark points
    N[2, ~valid_mask] = 1.0

    normal_map = N.T.reshape((h, w, 3))
    albedo_map = (np.clip(rho, 0.0, 1.0) * 255.0).reshape((h, w)).astype(np.uint8)

    return normal_map, albedo_map


def integrate_surface_frankot_chellappa(normal_map: np.ndarray) -> np.ndarray:
    """
    Frankot-Chellappa Fourier-based Surface Height Integration.
    Enforces surface integrability (curl == 0) in the frequency domain:
      del^2 Z = (dp/dx) + (dq/dy)
    Solves Poisson's equation for surface elevation Z(x, y) from surface slopes:
      p = -N_x / N_z, q = -N_y / N_z
    """
    h, w = normal_map.shape[:2]
    nx = normal_map[:, :, 0]
    ny = normal_map[:, :, 1]
    nz = normal_map[:, :, 2]

    # Prevent division by zero near grazing angles
    nz_safe = np.where(np.abs(nz) < 1e-3, 1e-3 * np.sign(nz + 1e-8), nz)
    p = -nx / nz_safe
    q = -ny / nz_safe

    # Frequency domain coordinates
    u = np.fft.fftfreq(w)
    v = np.fft.fftfreq(h)
    U, V = np.meshgrid(u, v)

    # Frequency gradients
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    # Frankot-Chellappa projection
    denom = (2.0 * np.pi * 1j * U)**2 + (2.0 * np.pi * 1j * V)**2
    # Avoid zero division at DC frequency (0, 0)
    denom[0, 0] = 1.0

    numer = (2.0 * np.pi * 1j * U) * P + (2.0 * np.pi * 1j * V) * Q
    Z_freq = numer / denom
    Z_freq[0, 0] = 0.0  # Set DC baseline height to 0

    height_map = np.real(np.fft.ifft2(Z_freq))
    # Normalize height map for visualization
    height_norm = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return height_norm
