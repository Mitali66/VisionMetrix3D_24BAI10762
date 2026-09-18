"""
VisionMetrix 3D - Feature Extraction & Image Segmentation Module
"""

from .edge_detectors import (
    canny_edge_detector_custom,
    apply_log_edge_detector,
    apply_dog_filter,
)
from .hough import (
    detect_hough_lines,
    detect_hough_circles,
)
from .corners import (
    harris_corner_detector,
    hessian_blob_detector,
)
from .descriptors import (
    extract_sift_features,
    match_features_lowe_ratio,
    compute_hog_features,
    generate_gabor_filter_bank,
    apply_gabor_texture_analysis,
)
from .scale_space import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian_pyramid,
)
from .segmentation import (
    region_growing_segmentation,
    mean_shift_segmentation,
    otsu_thresholding,
    grabcut_defect_segmentation,
)
