"""
VisionMetrix 3D - Feature Descriptors & Texture Analysis
Module: descriptors.py

Covers:
1. SIFT / ORB Keypoint Extraction and Feature Matching with Lowe's Ratio Test
2. Histogram of Oriented Gradients (HOG) Feature Representation
3. 2D Gabor Filter Bank for Multi-Orientation Texture and Scratch Metrology
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


def extract_sift_features(image: np.ndarray, n_features: int = 500) -> Tuple[List[Any], np.ndarray]:
    """
    Extracts Scale-Invariant Feature Transform (SIFT) keypoints and 128-D descriptors.
    Robust against scale, 3D viewpoint changes, and illumination variations.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features_lowe_ratio(desc1: np.ndarray, desc2: np.ndarray,
                              ratio_threshold: float = 0.75) -> List[Any]:
    """
    Performs KNN feature matching (k=2) and filters correspondences using Lowe's ratio test:
      ||d1 - d2_nearest|| / ||d1 - d2_second_nearest|| < ratio_threshold
    Rejects ambiguous correspondences arising from repetitive surface textures.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    # Use FLANN or BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    return good_matches


def compute_hog_features(image: np.ndarray,
                         cell_size: Tuple[int, int] = (8, 8),
                         block_size: Tuple[int, int] = (2, 2),
                         nbins: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Histogram of Oriented Gradients (HOG) descriptor.
    Encodes localized gradient structure invariant to illumination gradients.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize to standard dimensions for HOG if needed
    h, w = gray.shape
    h_win = (h // cell_size[0]) * cell_size[0]
    w_win = (w // cell_size[1]) * cell_size[1]
    resized = cv2.resize(gray, (w_win, h_win))

    win_size = (w_win, h_win)
    b_size = (block_size[1] * cell_size[1], block_size[0] * cell_size[0])
    b_stride = (cell_size[1], cell_size[0])
    c_size = (cell_size[1], cell_size[0])

    hog = cv2.HOGDescriptor(win_size, b_size, b_stride, c_size, nbins)
    descriptors = hog.compute(resized)

    return descriptors.flatten(), resized


def generate_gabor_filter_bank(ksize: int = 21,
                               sigmas: List[float] = [2.0, 4.0],
                               thetas: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                               lambdas: List[float] = [8.0, 16.0],
                               gamma: float = 0.5) -> List[np.ndarray]:
    """
    Constructs a 2D Gabor filter bank:
      g(x, y; lambda, theta, psi, sigma, gamma) = exp(-(x'^2 + gamma^2 * y'^2) / (2*sigma^2)) * cos(2*pi*x'/lambda + psi)
    Models primary visual cortex receptive fields for oriented texture analysis.
    """
    filters = []
    for s in sigmas:
        for t in thetas:
            for l in lambdas:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma=s, theta=t,
                                            lambd=l, gamma=gamma, psi=0,
                                            ktype=cv2.CV_32F)
                filters.append(kernel)
    return filters


def apply_gabor_texture_analysis(image: np.ndarray,
                                 filters: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters the surface with a Gabor bank to detect micro-scratches and directional abrasions.
    Returns the maximum response energy map and mean texture energy across filter orientations.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if filters is None:
        filters = generate_gabor_filter_bank()

    responses = []
    for kernel in filters:
        fimg = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(fimg))

    responses_stack = np.stack(responses, axis=0)
    max_response = np.max(responses_stack, axis=0)
    mean_response = np.mean(responses_stack, axis=0)

    max_norm = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_norm = cv2.normalize(mean_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return max_norm, mean_norm
