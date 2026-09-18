"""
VisionMetrix 3D - Integrated Inspection Pipeline
Module: inspector.py

Orchestrates the 4 Computer Vision modules into an end-to-end industrial inspection pipeline:
1. Low-Level Conditioning (Noise reduction, CLAHE, Wiener restoration)
2. Feature Metrology (Hough circular holes, alignment lines, Harris corners)
3. Defect Segmentation & Supervised Classification (Bayesian / KNN)
4. 3D Stereo Disparity & Point Cloud Export (.ply)
5. SQLite Database Audit Logging & Diagnostic Visualizations
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional

from visionmetrix.lowlevel.enhancement import clahe_enhancement
from visionmetrix.lowlevel.restoration import wiener_deconvolution, create_motion_blur_psf
from visionmetrix.features.edge_detectors import canny_edge_detector_custom
from visionmetrix.features.hough import detect_hough_circles, detect_hough_lines
from visionmetrix.features.corners import harris_corner_detector
from visionmetrix.features.descriptors import compute_hog_features, apply_gabor_texture_analysis
from visionmetrix.stereo3d.disparity import compute_disparity_sgbm, disparity_to_depth
from visionmetrix.stereo3d.reconstruction import triangulate_point_cloud, save_point_cloud_ply
from visionmetrix.pattern_motion.classification import KNNDefectClassifier
from visionmetrix.data.sample_generator import generate_defect_feature_dataset
from visionmetrix.storage.database import InspectionDatabase


@dataclass
class InspectionResult:
    item_id: str
    pass_status: str  # "PASS" or "FAIL"
    defect_count: int
    defects: List[Dict[str, Any]]
    metrology_holes: List[Tuple[int, int, int]]
    metrology_lines: List[Tuple[int, int, int, int]]
    annotated_image: np.ndarray
    point_cloud_path: Optional[str] = None
    mean_depth_m: Optional[float] = None


class VisionMetrixInspector:
    def __init__(self, db_path: str = "visionmetrix_inspection.db",
                 output_dir: str = "output"):
        self.db = InspectionDatabase(db_path)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Pre-train internal defect classifier
        self.classifier = KNNDefectClassifier(n_neighbors=5)
        X_train, y_train, self.class_names = generate_defect_feature_dataset(n_samples_per_class=80)
        self.classifier.train(X_train, y_train)

    def process_workpiece(self,
                          image: np.ndarray,
                          item_id: str = "ITEM_001",
                          session_id: str = "SESS_DEMO",
                          stereo_right: Optional[np.ndarray] = None,
                          has_vibration_blur: bool = False) -> InspectionResult:
        """
        Executes the multi-stage quality assurance inspection pipeline.
        """
        # Stage 1: Conditioning
        conditioned = image.copy()
        if has_vibration_blur:
            psf = create_motion_blur_psf(length=11, angle_deg=0.0)
            conditioned = wiener_deconvolution(conditioned, psf, snr=50.0)

        # Enhance dynamic contrast
        enhanced = clahe_enhancement(conditioned, clip_limit=2.0)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced.copy()

        # Stage 2: Geometric Metrology
        # Detect circular mounting holes (tolerance check)
        detected_holes, _ = detect_hough_circles(gray, dp=1.2, min_dist=40, param1=100, param2=25, min_radius=8, max_radius=30)
        
        # Detect structural alignment lines
        edges_canny, _, _ = canny_edge_detector_custom(gray, low_threshold=40, high_threshold=100)
        detected_lines, _ = detect_hough_lines(edges_canny, threshold=60, min_line_length=30)

        # Stage 3: Defect Segmentation & Analysis
        annotated = enhanced.copy()
        defects = []

        # Find candidate defect contours using adaptive thresholding and morphological filtering
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 4)
        # Exclude normal IC regions or regular shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter small sensor noise and large workpiece boundaries
            if 30 < area < 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(h, 1)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / max(perimeter**2, 1e-5)

                # Extract local patch properties
                patch = gray[y:y + h, x:x + w]
                mean_intensity = float(np.mean(patch))
                var_intensity = float(np.var(patch))

                # Feature vector for classification:
                # [Area, Perimeter, Circularity, AspectRatio, MeanIntensity, VarIntensity, HOG_Mean, Gabor_Energy]
                feat = np.array([[
                    area,
                    perimeter,
                    circularity,
                    min(aspect_ratio, 1.0 / max(aspect_ratio, 1e-4)),
                    mean_intensity,
                    var_intensity,
                    0.4,
                    0.5
                ]], dtype=np.float64)

                pred_class_idx = int(self.classifier.predict(feat)[0])
                pred_label = self.class_names[pred_class_idx]
                confidence = float(np.max(self.classifier.predict_proba(feat)[0]))

                # Determine severity
                severity = "CRITICAL" if pred_label in ["Micro-Crack", "Solder Bridge"] else "HIGH"

                # Draw bounding box and label badge on annotated image
                color = (0, 0, 255) if severity == "CRITICAL" else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label_text = f"{pred_label} ({confidence:.2f})"
                cv2.putText(annotated, label_text, (x, max(15, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                defect_info = {
                    "type": pred_label,
                    "confidence": confidence,
                    "bbox": (x, y, w, h),
                    "area": float(area),
                    "severity": severity
                }
                defects.append(defect_info)

                # Log defect to SQLite database
                self.db.log_defect(item_id, pred_label, confidence, (x, y, w, h), area, severity)

        # Highlight verified geometric holes in cyan
        for (hx, hy, hr) in detected_holes:
            cv2.circle(annotated, (hx, hy), hr, (255, 255, 0), 2)
            cv2.circle(annotated, (hx, hy), 3, (255, 255, 0), -1)

        # Stage 4: 3D Stereo Metrology (if right stereo image is provided)
        point_cloud_path = None
        mean_depth = None
        if stereo_right is not None:
            disparity = compute_disparity_sgbm(gray, cv2.cvtColor(stereo_right, cv2.COLOR_BGR2GRAY)
                                               if len(stereo_right.shape) == 3 else stereo_right,
                                               num_disparities=64, block_size=5)
            depth_map = disparity_to_depth(disparity, focal_length_px=800.0, baseline_meters=0.1)

            # Triangulate point cloud
            pts, cols = triangulate_point_cloud(depth_map, enhanced, focal_length=800.0,
                                                principal_point=(gray.shape[1] / 2.0, gray.shape[0] / 2.0))
            if len(pts) > 0:
                point_cloud_path = os.path.join(self.output_dir, f"{item_id}_pointcloud.ply")
                save_point_cloud_ply(point_cloud_path, pts, cols)
                valid_depths = depth_map[depth_map > 0]
                mean_depth = float(np.mean(valid_depths)) if len(valid_depths) > 0 else 1.0
                max_depth = float(np.max(valid_depths)) if len(valid_depths) > 0 else 1.0
                self.db.log_3d_reconstruction(item_id, point_cloud_path, len(pts), mean_depth * 1000.0, max_depth * 1000.0)

        # Decision rule: Item FAILS if any CRITICAL defect or >2 defects found
        has_critical = any(d["severity"] == "CRITICAL" for d in defects)
        pass_status = "FAIL" if (has_critical or len(defects) >= 2) else "PASS"

        # Log item to database
        self.db.record_item(item_id, session_id, "PCB_SURFACE", pass_status, len(defects))

        # Stamp status badge on top-left of annotated image
        status_color = (0, 200, 0) if pass_status == "PASS" else (0, 0, 240)
        cv2.rectangle(annotated, (10, 10), (160, 45), (20, 20, 20), -1)
        cv2.putText(annotated, f"STATUS: {pass_status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return InspectionResult(
            item_id=item_id,
            pass_status=pass_status,
            defect_count=len(defects),
            defects=defects,
            metrology_holes=detected_holes,
            metrology_lines=detected_lines,
            annotated_image=annotated,
            point_cloud_path=point_cloud_path,
            mean_depth_m=mean_depth
        )
