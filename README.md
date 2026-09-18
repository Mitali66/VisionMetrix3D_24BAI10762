# VisionMetrix 3D (VM3D)
### Automated Spatial Metrology, Multi-View 3D Surface Profilometry, and Dynamic Scene Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 18 Passed](https://img.shields.io/badge/tests-18%20passed-brightgreen.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)

---

## 1. Project Overview
**VisionMetrix 3D (VM3D)** is a comprehensive, production-grade Computer Vision and Spatial Intelligence platform built for automated optical inspection (AOI), sub-millimeter geometric metrology, 3D surface topography reconstruction, and dynamic conveyor tracking.

Engineered to fulfill all four units of the Computer Vision curriculum, VisionMetrix 3D demonstrates how fundamental mathematical concepts—from 2D Fourier transforms and epipolar geometry to photometric shape-from-shading and dense optical flow—coalesce into an industrial-strength quality control system.

```
+-----------------------------------------------------------------------------------------+
|                                 VISIONMETRIX 3D ARCHITECTURE                            |
+-----------------------------------------------------------------------------------------+
| 1. Acquisition Tier     | Stereo Rig Sensors | Multi-Directional LEDs | Conveyor Motion |
| 2. Conditioning Tier    | DLT Homography     | 2D FFT Butterworth     | Wiener Deconv   |
| 3. Metrology Tier       | Custom 5-Stage Canny | Hough Lines/Circles   | Harris Keypoints|
| 4. 3D & Intelligence    | SGBM Disparity/PLY | Photometric Stereo     | Optical Flow/PCA|
| 5. Storage & Audit      | SQLite Audit DB    | Interactive GUI        | Unified CLI/PDF |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Curriculum Topics Covered

| Syllabus Unit | Course Concepts | VisionMetrix 3D Module & Implementation |
|---|---|---|
| **Unit 1: Digital Image Formation & Low-Level Processing** | Transformations (Euclidean, Affine, Projective), 2D Fourier Transform, Spatial Convolution, Filtering, Enhancement, Restoration | `visionmetrix.lowlevel`: `transformations.py` (DLT SVD homography), `filtering.py` (Sobel, Laplacian, Bilateral), `fourier.py` (2D FFT, Butterworth, Notch), `enhancement.py` (CLAHE, Gamma), `restoration.py` (Wiener deconvolution) |
| **Unit 2: Depth Estimation & Multi-Camera Views** | Binocular Stereopsis, Epipolar Geometry, Homography, Rectification, DLT, RANSAC, 3D Reconstruction, Photometric Stereo | `visionmetrix.stereo3d`: `calibration.py` (DLT camera calibration, RQ decomposition), `epipolar.py` (8-point RANSAC, F & E matrices), `disparity.py` (SGBM disparity), `reconstruction.py` (Point cloud triangulation, `.ply`), `photometric_stereo.py` (Woodham's normals & Frankot-Chellappa height) |
| **Unit 3: Feature Extraction & Image Segmentation** | Edges (Canny, LoG, DoG), Line/Circle Detectors (Hough), Corners (Harris, Hessian), Descriptors (SIFT, HOG, Gabor), Pyramids, Segmentation (Otsu, Region Growing, GrabCut) | `visionmetrix.features`: `edge_detectors.py` (handcrafted 5-stage Canny, LoG, DoG), `hough.py` (Hough lines & circles), `corners.py` (Harris response tensor), `descriptors.py` (HOG, Gabor bank, SIFT), `scale_space.py` (Gaussian/Laplacian pyramids), `segmentation.py` (first-principles Otsu, GrabCut) |
| **Unit 4: Pattern Analysis & Motion Analysis** | Clustering (K-Means, GMM), Classification (KNN, Naive Bayes), Dimensionality Reduction (PCA, LDA), Motion Analysis (MOG2, Optical Flow, KLT) | `visionmetrix.pattern_motion`: `clustering.py` (K-Means, GMM), `classification.py` (GaussianNB, KNN), `dim_reduction.py` (PCA eigen-subspace anomaly scoring, LDA), `motion.py` (MOG2 background subtraction, Lucas-Kanade KLT, Farneback dense optical flow) |

---

## 3. Key Features
- **Deterministic 5-Tier Pipeline:** Cleanly decouples image acquisition, conditioning, geometric metrology, 3D reconstruction, and audit storage.
- **Hybrid 3D Profilometry:** Combines binocular stereo disparity for macroscopic metric depth ($Z$) with multi-light photometric stereo for microscopic surface roughness ($N(x,y)$).
- **Interactive Desktop Dashboard (`dashboard.py`):** Visual Tkinter GUI with real-time parameter sliders (Canny thresholds, CLAHE clip limits, stereo block sizes) and live database views.
- **Unified CLI (`cli.py`):** One-line command execution for automated demos, custom inspections, photometric reconstruction, and PDF report compilation.
- **Self-Contained Data Engine:** Built-in generators synthesize calibrated workpieces, realistic defects, stereo pairs, and motion frames out-of-the-box with zero external downloads.
- **100% Automated Test Coverage:** 18 automated unit and integration tests verifying all mathematical invariants, numerical stability, and error bounds.
- **Complete Academic Report (`Project_Report.pdf`):** Programmatic 15-section PDF report generated via ReportLab matching course submission guidelines.

---

## 4. Technologies & Libraries
- **Language:** Python 3.10+ / 3.12
- **Computer Vision & Scientific Computing:** OpenCV (`cv2`), NumPy, SciPy, Scikit-Image, Scikit-Learn
- **Visualization & Plotting:** Matplotlib
- **Document Compilation:** ReportLab
- **Testing & Verification:** Pytest
- **Database Engine:** SQLite3 (embedded ACID relational storage)
- **GUI Engine:** Python Tkinter

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.

### Clone the Repository
```bash
git clone https://github.com/visionmetrix/visionmetrix-3d.git
cd visionmetrix-3d
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. How to Run the Project

### 1. Run the End-to-End Demonstration (CLI)
Executes all four modules on synthetic industrial workpieces, saves annotated outputs and point clouds to `output/`, and prints a database summary:
```bash
python cli.py demo
```

### 2. Launch the Interactive Desktop Dashboard (GUI)
Launches the interactive visual inspection dashboard with live parameter tuning, Canny/Hough sliders, 3D photometric height visualizer, and SQLite database audit views:
```bash
python dashboard.py
```

### 3. Inspect a Custom Image or Stereo Pair
```bash
python cli.py inspect --image path/to/left.png --right path/to/right.png --item-id PCB_101
```

### 4. Run Multi-Illumination Photometric Stereo 3D
```bash
python cli.py photometric --size 256
```

### 5. Generate the Official Academic PDF Report
Compiles the comprehensive 15-section academic capstone report (`Project_Report.pdf`):
```bash
python scripts/generate_report.py
```

---

## 7. Automated Testing & Verification
The test suite validates mathematical invariants, matrix decompositions, and end-to-end integration:

```bash
python -m pytest -v
```

### Test Suite Summary:
```
============================= test session starts =============================
tests/test_features.py::test_edge_detectors PASSED                       [  5%]
tests/test_features.py::test_hough_transforms PASSED                     [ 11%]
tests/test_features.py::test_corners_and_scale_space PASSED              [ 16%]
tests/test_features.py::test_segmentation PASSED                         [ 22%]
tests/test_lowlevel.py::test_orthogonal_transformations PASSED           [ 27%]
tests/test_lowlevel.py::test_euclidean_and_affine PASSED                 [ 33%]
tests/test_lowlevel.py::test_dlt_homography PASSED                       [ 38%]
tests/test_lowlevel.py::test_spatial_convolution_and_filtering PASSED    [ 44%]
tests/test_lowlevel.py::test_fourier_filtering PASSED                    [ 50%]
tests/test_lowlevel.py::test_enhancement_and_restoration PASSED          [ 55%]
tests/test_pattern_motion.py::test_clustering_and_classification PASSED  [ 61%]
tests/test_pattern_motion.py::test_pca_dim_reduction PASSED              [ 66%]
tests/test_pattern_motion.py::test_motion_tracking PASSED                [ 72%]
tests/test_pipeline.py::test_end_to_end_inspection_pipeline PASSED       [ 77%]
tests/test_stereo3d.py::test_camera_calibration_dlt PASSED               [ 83%]
tests/test_stereo3d.py::test_epipolar_geometry PASSED                    [ 88%]
tests/test_stereo3d.py::test_disparity_and_reconstruction PASSED         [ 94%]
tests/test_stereo3d.py::test_photometric_stereo PASSED                   [100%]
============================= 18 passed in 2.82s ==============================
```

---

## 8. Directory & File Structure
```
visionmetrix/
├── visionmetrix/
│   ├── lowlevel/                # Unit 1: Geometric transformations, FFT, Wiener deconvolution
│   ├── stereo3d/                # Unit 2: DLT, Epipolar RANSAC, SGBM disparity, Photometric stereo
│   ├── features/                # Unit 3: Custom Canny, Hough, Harris, HOG, Gabor, Pyramids, Otsu
│   ├── pattern_motion/          # Unit 4: K-Means/GMM, KNN/Bayes, PCA anomaly, Farneback flow
│   ├── storage/                 # Relational SQLite inspection and calibration database
│   ├── pipeline/                # End-to-end inspection orchestrator and decision engine
│   └── data/                    # Synthetic industrial workpiece, defect, and stereo generator
├── tests/                       # 18 automated unit and integration tests
├── docs/                        # Architectural, workflow, use case, and ER diagrams
├── scripts/
│   ├── generate_report.py       # Programmatic PDF report compiler
│   └── generate_diagrams.py     # High-resolution architectural diagram generator
├── cli.py                       # Unified command-line interface
├── dashboard.py                 # Interactive Tkinter visual dashboard
├── statement.md                 # Problem statement, scope, target users, high-level features
├── Project_Report.pdf           # Comprehensive 15-section academic PDF submission report
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 9. License
This project is licensed under the MIT License.
