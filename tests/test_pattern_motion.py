"""
Unit Tests - Pattern Analysis & Motion Tracking
"""
import pytest
import numpy as np

from visionmetrix.pattern_motion.clustering import (
    kmeans_color_clustering,
    gmm_defect_clustering,
)
from visionmetrix.pattern_motion.classification import (
    GaussianNaiveBayesDefectClassifier,
    KNNDefectClassifier,
    evaluate_classifier_metrics,
)
from visionmetrix.pattern_motion.dim_reduction import (
    PrincipalComponentAnalysisCustom,
    compute_reconstruction_anomaly_score,
)
from visionmetrix.pattern_motion.motion import (
    ConveyorBackgroundSubtractor,
    track_sparse_optical_flow_klt,
    compute_dense_optical_flow_farneback,
    estimate_motion_trajectory,
)
from visionmetrix.data.sample_generator import (
    generate_defect_feature_dataset,
    generate_conveyor_video_sequence,
)


def test_clustering_and_classification():
    X, y, class_names = generate_defect_feature_dataset(n_samples_per_class=50)

    # Train/Test Split
    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # KNN Classifier
    knn = KNNDefectClassifier(n_neighbors=3)
    knn.train(X_train, y_train)
    preds_knn = knn.predict(X_test)
    metrics_knn = evaluate_classifier_metrics(y_test, preds_knn)
    assert metrics_knn["accuracy"] > 0.85

    # Gaussian Naive Bayes
    gnb = GaussianNaiveBayesDefectClassifier()
    gnb.train(X_train, y_train)
    preds_gnb = gnb.predict(X_test)
    metrics_gnb = evaluate_classifier_metrics(y_test, preds_gnb)
    assert metrics_gnb["accuracy"] > 0.85

    # GMM clustering
    gmm, labels, probs = gmm_defect_clustering(X, n_components=4)
    assert len(labels) == len(X)
    assert probs.shape == (len(X), 4)


def test_pca_dim_reduction():
    X, _, _ = generate_defect_feature_dataset(n_samples_per_class=50)
    pca = PrincipalComponentAnalysisCustom(n_components=4)
    pca.fit(X)

    Z = pca.transform(X)
    assert Z.shape == (len(X), 4)

    # Reconstruction
    X_recon = pca.inverse_transform(Z)
    assert X_recon.shape == X.shape

    scores = compute_reconstruction_anomaly_score(pca, X)
    assert len(scores) == len(X)


def test_motion_tracking():
    frames = generate_conveyor_video_sequence(num_frames=6, width=200, height=150)
    subtractor = ConveyorBackgroundSubtractor()

    for frame in frames:
        clean_mask, _ = subtractor.apply(frame)
        assert clean_mask.shape == frame.shape[:2]

    # Dense Optical Flow
    prev_gray = frames[0][:, :, 0]
    curr_gray = frames[1][:, :, 0]
    flow, flow_vis = compute_dense_optical_flow_farneback(prev_gray, curr_gray)
    assert flow.shape == (150, 200, 2)
    assert flow_vis.shape == (150, 200, 3)

    # Sparse KLT tracking
    pts = np.array([[60, 110], [70, 110], [60, 120]], dtype=np.float32)
    p1, status, _ = track_sparse_optical_flow_klt(prev_gray, curr_gray, pts)
    assert len(p1) == len(pts)
