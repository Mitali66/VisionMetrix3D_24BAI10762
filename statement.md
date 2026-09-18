# VisionMetrix 3D (VM3D) — Problem Statement & Scope Specification

## 1. Problem Statement
In modern precision manufacturing (aerospace, automotive, semiconductor electronics, and medical devices), quality assurance and spatial metrology demand automated visual inspection systems capable of:
1. **Handling Environmental Degradation:** High-speed factory vibrations induce motion blur, uneven lighting causes glare and deep shadows, and sensor thermal noise degrades edge sharpness.
2. **Overcoming Monocular Depth Ambiguity:** Standard 2D flat cameras cannot differentiate between harmless surface stains and critical 3D structural defects (e.g., solder bridging, substrate cracks, cavities, and warpage).
3. **Sub-Millimeter Defect Metrology:** High-precision manufacturing requires isolating micro-cracks, measuring hole diameters against strict tolerances ($\pm 0.1$ mm), and tracking keypoint displacements.
4. **Dynamic Conveyor Motion Tracking:** Workpieces translate continuously across conveyor belts, requiring real-time motion estimation, velocity calculation, and background subtraction.

Traditional manual visual inspection is subjective, slow, and expensive, while off-the-shelf black-box deep learning models lack geometric interpretability, require massive labeled datasets, and fail to provide calibrated 3D spatial depth.

---

## 2. Scope of the Project
**VisionMetrix 3D** establishes an end-to-end Computer Vision system unifying all four core academic syllabus units into an industrial-grade inspection and 3D profilometry pipeline:
- **Unit 1: Low-Level Image Processing:** Geometric transformations (Euclidean, Affine, Projective DLT Homography), frequency-domain 2D FFT filtering (Butterworth, Notch), radiometric enhancement (CLAHE, Gamma), and Point Spread Function (PSF) Wiener deconvolution.
- **Unit 2: Depth Estimation & Multi-Camera Views:** Direct Linear Transformation (DLT) camera calibration, Epipolar geometry (8-Point algorithm with RANSAC, Essential and Fundamental matrices), dense stereo disparity mapping (SGBM), 3D point cloud triangulation (`.ply`), and Woodham's Photometric Stereo (surface normals and Frankot-Chellappa height map integration).
- **Unit 3: Feature Extraction & Segmentation:** Handcrafted 5-stage Canny edge detector, Hough line and circle dimensional metrology, Harris corner response, Hessian blob detection, SIFT/ORB feature matching, HOG descriptors, Gabor texture filter banks, Laplacian pyramid scale-space, and first-principles Otsu/GrabCut segmentation.
- **Unit 4: Pattern Analysis & Motion Tracking:** Unsupervised K-Means and GMM clustering, supervised KNN and Gaussian Naive Bayes defect classification, PCA eigen-defect anomaly modeling, MOG2 conveyor background subtraction, and Lucas-Kanade/Farneback optical flow velocity tracking.
- **Data & Audit Persistence:** Self-contained synthetic industrial dataset generator and an embedded SQLite relational database logging inspection records, defect bounding boxes, and 3D point cloud assets.

---

## 3. Target Users
1. **Quality Assurance (QA) & Metrology Engineers:** Setting tolerances, running automated line inspections, reviewing PASS/FAIL yields, and logging defect distributions.
2. **Manufacturing & Process Engineers:** Monitoring conveyor belt line speed, identifying defect root causes (e.g., repeating solder bridges indicating solder wave nozzle issues), and auditing 3D height variations.
3. **Robotics & Automation Integrators:** Interfacing robotic pick-and-place actuators with real-time 3D spatial point clouds and conveyor velocity vectors.
4. **Academic Evaluators & Researchers:** Validating the mathematical and algorithmic rigor of classical computer vision techniques applied to real-world industrial inspection.

---

## 4. High-Level Features
- **Deterministic 5-Tier Pipeline:** Cleanly decouples image acquisition, conditioning, geometric metrology, 3D reconstruction, and audit storage.
- **Hybrid 3D Profilometry:** Combines binocular stereo disparity for macroscopic metric depth ($Z$) with multi-light photometric stereo for microscopic surface roughness ($N(x,y)$).
- **Interactive Desktop Dashboard:** Visual Tkinter GUI with real-time parameter sliders (Canny thresholds, CLAHE clip limits, stereo block sizes) and live database views.
- **Unified CLI:** One-line command execution for automated demos, custom inspections, photometric reconstruction, and PDF report compilation.
- **Zero External Dependencies:** Built-in generators synthesize calibrated workpieces, realistic defects, stereo pairs, and motion frames out-of-the-box.
- **100% Automated Test Coverage:** 18 automated unit and integration tests verifying all mathematical invariants and error bounds.
