"""
VisionMetrix 3D - Dimensionality Reduction & Anomaly Metrology
Module: dim_reduction.py

Theoretical Background:
1. Principal Component Analysis (PCA):
   Finds orthogonal axes of maximum variance via eigen-decomposition of the sample covariance matrix:
     Sigma = (1 / N) * X_c^T * X_c
     Sigma * v_k = lambda_k * v_k
   Enables projection to low-dimensional manifold and "Eigen-Defect" subspace modeling.
2. Anomaly Metric via Reconstruction Residual:
   Unseen severe structural defects cannot be accurately reconstructed by the normal PCA basis:
     Score(x) = || x - Reconstruct(Project(x)) ||^2
3. Linear Discriminant Analysis (LDA):
   Supervised dimensionality reduction maximizing Fisher's separation ratio S_B / S_W.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from typing import Tuple, Optional


class PrincipalComponentAnalysisCustom:
    """
    First-principles Principal Component Analysis (PCA) implementation.
    """
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Fits PCA on data matrix X of shape (N_samples, D_features).
        """
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute sample covariance matrix
        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        # Eigen-decomposition of symmetric covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = min(self.n_components, n_features)
        self.components_ = eigenvectors[:, :k]  # shape (D, k)
        self.explained_variance_ = eigenvalues[:k]

        total_var = np.sum(eigenvalues)
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(k)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects samples into principal component subspace: Z = (X - mu) * V.
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstructs original feature space from subspace coordinates: X_hat = Z * V^T + mu.
        """
        return (Z @ self.components_.T) + self.mean_


def compute_reconstruction_anomaly_score(pca: PrincipalComponentAnalysisCustom,
                                         X: np.ndarray) -> np.ndarray:
    """
    Calculates L2 reconstruction residual error:
      Error(x) = || x - x_hat ||_2
    Significant anomalies exhibit high reconstruction error.
    """
    Z = pca.transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    residuals = np.linalg.norm(X - X_reconstructed, axis=1)
    return residuals


def compute_lda_projection(X: np.ndarray, y: np.ndarray,
                           n_components: int = 2) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
    """
    Applies Linear Discriminant Analysis (LDA) to maximize class separability.
    """
    n_classes = len(np.unique(y))
    k = min(n_components, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=k)
    X_projected = lda.fit_transform(X, y)
    return lda, X_projected
