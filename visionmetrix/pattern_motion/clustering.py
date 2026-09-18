"""
VisionMetrix 3D - Unsupervised Clustering
Module: clustering.py

Theoretical Background:
1. K-Means Clustering:
   Partitions N observations into K clusters minimizing within-cluster sum of squares (WCSS):
     J = sum_{i=1}^K sum_{x in S_i} ||x - mu_i||^2
2. Gaussian Mixture Models (GMM):
   Models feature distributions as linear superposition of K Gaussians via Expectation-Maximization:
     p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)
"""

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple


def kmeans_color_clustering(image: np.ndarray,
                            k: int = 4,
                            max_iters: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs K-Means clustering on pixel color distributions.
    Useful for color quantization, separating metal substrates from oxidized defects.
    
    Returns:
      quantized_image: Image with pixels mapped to their cluster center colors
      labels: (H, W) cluster assignment map
    """
    h, w = image.shape[:2]
    data = image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iters, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape((h, w, 3))
    label_map = labels.reshape((h, w))

    return quantized, label_map


def gmm_defect_clustering(feature_vectors: np.ndarray,
                          n_components: int = 3,
                          covariance_type: str = "full") -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """
    Fits a Gaussian Mixture Model to defect feature descriptors (e.g., area, perimeter, intensity variance).
    Provides soft cluster membership probabilities p(c_k | x).
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                          random_state=42, max_iter=200)
    gmm.fit(feature_vectors)
    labels = gmm.predict(feature_vectors)
    probabilities = gmm.predict_proba(feature_vectors)
    return gmm, labels, probabilities
