"""
VisionMetrix 3D - Stereo Disparity & Depth Estimation
Module: disparity.py

Computes dense disparity maps from rectified stereo pairs:
1. Block Matching (Sum of Absolute Differences - SAD)
2. Semi-Global Block Matching (SGBM) using directional path optimization
3. Triangulation to Metric Depth: Z = (f * B) / d
"""

import cv2
import numpy as np
from typing import Tuple


def compute_disparity_block_matching(left_gray: np.ndarray, right_gray: np.ndarray,
                                     num_disparities: int = 64,
                                     block_size: int = 15) -> np.ndarray:
    """
    Computes disparity using classical Block Matching (StereoBM).
    Compares pixel intensity patches along the rectified horizontal epipolar scanline.
    """
    # OpenCV requires num_disparities to be divisible by 16 and block_size to be odd
    num_disp = max(16, (num_disparities // 16) * 16)
    bs = max(5, block_size if block_size % 2 == 1 else block_size + 1)

    matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=bs)
    matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    matcher.setPreFilterCap(31)
    matcher.setTextureThreshold(10)
    matcher.setUniquenessRatio(15)

    disparity_raw = matcher.compute(left_gray, right_gray)
    # StereoBM outputs fixed-point values with 4 fractional bits (divided by 16)
    disparity = disparity_raw.astype(np.float32) / 16.0
    disparity[disparity < 0] = 0.0
    return disparity


def compute_disparity_sgbm(left: np.ndarray, right: np.ndarray,
                           min_disparity: int = 0,
                           num_disparities: int = 64,
                           block_size: int = 5) -> np.ndarray:
    """
    Semi-Global Block Matching (SGBM) based on Hirschm?ller's algorithm.
    Optimizes a 1D energy functional along multiple 1D paths (8 directions),
    combining pixel matching costs with smoothness penalty terms P1 and P2:
      E(D) = sum_p C(p, D_p) + sum_{q in N_p} P1 * [|D_p - D_q| = 1] + P2 * [|D_p - D_q| > 1]
    Eliminates aperture problems and provides crisp defect depth boundaries.
    """
    num_disp = max(16, (num_disparities // 16) * 16)
    bs = max(3, block_size if block_size % 2 == 1 else block_size + 1)

    # Calculate optimal smoothness penalties
    p1 = 8 * 3 * bs**2
    p2 = 32 * 3 * bs**2

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disp,
        blockSize=bs,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity_raw = matcher.compute(left, right)
    disparity = disparity_raw.astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.0
    return disparity


def disparity_to_depth(disparity: np.ndarray, focal_length_px: float,
                       baseline_meters: float,
                       min_depth: float = 0.1,
                       max_depth: float = 10.0) -> np.ndarray:
    """
    Converts disparity map d(x, y) to metric depth Z(x, y) via triangulation:
      Z = (focal_length * baseline) / disparity
      
    Zero or negative disparities (unmatched pixels / occlusions) are mapped to 0 (invalid).
    """
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid_mask = disparity > 1e-4

    depth[valid_mask] = (focal_length_px * baseline_meters) / disparity[valid_mask]
    depth[depth < min_depth] = 0.0
    depth[depth > max_depth] = 0.0
    return depth
