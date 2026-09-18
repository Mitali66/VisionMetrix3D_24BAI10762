# VisionMetrix 3D (VM3D) — Architectural Specification

## 1. Architectural Overview
VisionMetrix 3D utilizes a deterministic Pipe-and-Filter architecture organized into five distinct layers:
1. **Acquisition & Simulation Layer:** Manages physical or synthetic camera triggers, stereo baselines, and multi-directional lighting sources.
2. **Preprocessing & Conditioning Layer:** Normalizes radiometric variations and restores optical/vibration blurs.
3. **Spatial Metrology & Feature Intelligence Layer:** Identifies edges, geometric primitives, corners, and surface textures.
4. **3D Reconstruction & Dynamic Pattern Layer:** Solves stereo disparity, photometric shape-from-shading, and conveyor motion tracking.
5. **Storage, UI & Decision Layer:** Persists audit logs in SQLite, renders interactive GUI controls, and manages PASS/FAIL verdicts.

---

## 2. System Architecture Diagram

```mermaid
flowchart TD
    subgraph Layer1["1. Acquisition Layer"]
        A1["Stereo Camera Rig (Left/Right)"]
        A2["Multi-Directional LED Illuminators"]
        A3["Conveyor Speed Sensor"]
    end

    subgraph Layer2["2. Conditioning & Low-Level Processing"]
        B1["DLT Homography & Perspective Rectification"]
        B2["2D FFT & Butterworth Filtering"]
        B3["CLAHE & Radiometric Gamma Correction"]
        B4["Wiener Deconvolution (PSF Blur Removal)"]
    end

    subgraph Layer3["3. Spatial Metrology & Feature Intelligence"]
        C1["Custom 5-Stage Canny Edge Detection"]
        C2["Hough Lines & Circles Dimensional Metrology"]
        C3["Harris Corner Structure Tensor"]
        C4["Gabor Filter Bank & HOG Descriptors"]
        C5["Laplacian Pyramid Scale-Space"]
    end

    subgraph Layer4["4. 3D Metrology & Motion Analysis"]
        D1["Stereo SGBM Disparity & Depth Mapping"]
        D2["3D Triangulation & PLY Point Cloud Export"]
        D3["Photometric Stereo (Normals & Height Map)"]
        D4["MOG2 Conveyor Background Subtraction"]
        D5["Lucas-Kanade KLT & Dense Farneback Flow"]
        D6["Supervised KNN/Bayes & PCA Anomaly Score"]
    end

    subgraph Layer5["5. Storage & Presentation Layer"]
        E1["SQLite Relational Inspection Database"]
        E2["Interactive Desktop Dashboard (Tkinter)"]
        E3["Unified CLI & Automated PDF Generator"]
    end

    A1 & A2 & A3 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Layer5
```

---

## 3. Sequence Diagram (Inspection Cycle)

```mermaid
sequenceDiagram
    autonumber
    actor QA as QA Engineer / Operator
    participant UI as Dashboard / CLI
    participant Insp as VisionMetrixInspector
    participant LL as LowLevel Engine
    participant Feat as Feature Engine
    participant S3D as Stereo3D Engine
    participant DB as SQLite Database

    QA->>UI: Trigger Inspection Run (Workpiece Capture)
    UI->>Insp: process_workpiece(image, stereo_right)
    Insp->>LL: clahe_enhancement(image)
    LL-->>Insp: enhanced_image
    Insp->>Feat: detect_hough_circles(gray)
    Feat-->>Insp: detected_holes, hole_overlay
    Insp->>Feat: canny_edge_detector_custom(gray)
    Feat-->>Insp: binary_edges, mag, angle
    Insp->>Feat: extract_defect_contours()
    Feat-->>Insp: defect_bounding_boxes
    Insp->>S3D: compute_disparity_sgbm(left, right)
    S3D-->>Insp: disparity_map, depth_map
    Insp->>S3D: triangulate_point_cloud(depth)
    S3D-->>Insp: point_cloud_ply_path
    Insp->>DB: log_defect(), record_item(), log_3d()
    DB-->>Insp: transaction_ack
    Insp-->>UI: InspectionResult (PASS/FAIL, defects, ply)
    UI-->>QA: Render Annotated Canvas & Status Badge
```

---

## 4. Class & Component Diagram

```mermaid
classDiagram
    class VisionMetrixInspector {
        +InspectionDatabase db
        +KNNDefectClassifier classifier
        +process_workpiece(image, stereo_right) InspectionResult
    }

    class InspectionResult {
        +str item_id
        +str pass_status
        +int defect_count
        +list defects
        +list metrology_holes
        +list metrology_lines
        +ndarray annotated_image
        +str point_cloud_path
        +float mean_depth_m
    }

    class InspectionDatabase {
        +str db_path
        +start_session(session_id, operator)
        +record_item(item_id, session_id, pass_status)
        +log_defect(item_id, defect_type, bbox, area)
        +log_3d_reconstruction(item_id, ply_path, vertex_count)
        +get_summary_statistics() dict
    }

    class LowLevelModule {
        +apply_projective(image, H)
        +wiener_deconvolution(image, psf, snr)
        +compute_fft2d(image)
        +clahe_enhancement(image)
    }

    class Stereo3DModule {
        +calibrate_camera_dlt(pts_3d, pts_2d)
        +estimate_fundamental_ransac(pts1, pts2)
        +compute_disparity_sgbm(left, right)
        +solve_photometric_stereo(images, lights)
        +integrate_surface_frankot_chellappa(normals)
    }

    class FeatureModule {
        +canny_edge_detector_custom(image)
        +detect_hough_circles(image)
        +detect_hough_lines(image)
        +harris_corner_detector(image)
        +otsu_thresholding(image)
    }

    class PatternMotionModule {
        +KNNDefectClassifier
        +GaussianNaiveBayesDefectClassifier
        +PrincipalComponentAnalysisCustom
        +ConveyorBackgroundSubtractor
        +compute_dense_optical_flow_farneback()
    }

    VisionMetrixInspector --> InspectionResult : creates
    VisionMetrixInspector --> InspectionDatabase : logs to
    VisionMetrixInspector --> LowLevelModule : uses
    VisionMetrixInspector --> Stereo3DModule : uses
    VisionMetrixInspector --> FeatureModule : uses
    VisionMetrixInspector --> PatternMotionModule : uses
```
