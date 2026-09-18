"""
VisionMetrix 3D - Supervised Pattern Classification
Module: classification.py

Implements Bayesian and Nearest-Neighbor classifiers for industrial defect categorization
(e.g., classifying surface anomalies as Scratch, Micro-Crack, Pinhole, or Solder Bridge):
1. Gaussian Naive Bayes:
   Maximum A Posteriori (MAP) classification with class conditional Gaussian likelihoods:
     y_hat = argmax_c P(C_c) * prod_j (1 / sqrt(2*pi*sigma_cj^2)) * exp(-(x_j - mu_cj)^2 / (2*sigma_cj^2))
2. K-Nearest Neighbors (KNN):
   Non-parametric classification based on Euclidean distance voting with confidence estimation.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, Tuple


class GaussianNaiveBayesDefectClassifier:
    """
    Gaussian Naive Bayes classifier for automated defect categorization.
    """
    def __init__(self):
        self.model = GaussianNB()
        self.classes_ = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class KNNDefectClassifier:
    """
    K-Nearest Neighbors (KNN) classifier with inverse distance weighting.
    """
    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights="distance")
        self.classes_ = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def evaluate_classifier_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
