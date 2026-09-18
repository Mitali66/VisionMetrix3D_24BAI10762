"""
VisionMetrix 3D - Synthetic Industrial Data Generator
Module: sample_generator.py

Synthesizes high-fidelity industrial images and ground-truth geometries:
- Machined PCB/component surfaces with traces, pads, and circular drill holes.
- Calibrated stereo image pairs with known metric baseline and ground-truth depth.
- Multi-illumination Photometric Stereo series under Lambertian reflection.
- Consecutive conveyor belt video frames for optical flow and background subtraction.
- Multi-class defect feature datasets for pattern classification and PCA.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


def generate_synthetic_workpiece(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates a realistic synthetic industrial PCB workpiece.
    Contains metallic copper ground planes, IC chips, solder pads, circuit traces,
    and circular mounting drill holes.
    """
    # Base dark green PCB substrate
    workpiece = np.full((height, width, 3), (25, 60, 20), dtype=np.uint8)

    # Add metallic copper bus bars and ground planes
    cv2.rectangle(workpiece, (40, 40), (width - 40, 70), (45, 140, 180), -1)
    cv2.rectangle(workpiece, (40, height - 70), (width - 40, height - 40), (45, 140, 180), -1)

    # Add IC Chips (black packages with pins)
    ic_boxes = [
        (100, 120, 140, 140),
        (350, 150, 180, 120),
        (200, 300, 120, 80)
    ]
    for (x, y, w, h) in ic_boxes:
        # Silicon package
        cv2.rectangle(workpiece, (x, y), (x + w, y + h), (30, 30, 30), -1)
        cv2.rectangle(workpiece, (x, y), (x + w, y + h), (80, 80, 80), 2)
        # Laser marking text
        cv2.putText(workpiece, "ARM-V8", (x + 15, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # IC Pins
        for pin_x in range(x + 10, x + w - 10, 15):
            cv2.line(workpiece, (pin_x, y - 8), (pin_x, y), (180, 180, 180), 2)
            cv2.line(workpiece, (pin_x, y + h), (pin_x, y + h + 8), (180, 180, 180), 2)

    # Gold circular solder pads
    for py in range(120, height - 100, 40):
        for px in [60, 280, 560]:
            cv2.circle(workpiece, (px, py), 9, (30, 180, 220), -1)
            cv2.circle(workpiece, (px, py), 4, (10, 10, 10), -1)

    # Circuit traces connecting pads to ICs
    cv2.line(workpiece, (60, 120), (100, 140), (40, 150, 200), 2)
    cv2.line(workpiece, (60, 160), (100, 180), (40, 150, 200), 2)
    cv2.line(workpiece, (240, 140), (280, 140), (40, 150, 200), 2)
    cv2.line(workpiece, (280, 200), (350, 200), (40, 150, 200), 2)
    cv2.line(workpiece, (350, 250), (300, 300), (40, 150, 200), 2)

    # Circular mounting holes (drilled through)
    for hole_pos in [(60, 60), (width - 60, 60), (60, height - 60), (width - 60, height - 60)]:
        cv2.circle(workpiece, hole_pos, 18, (120, 120, 120), 2)
        cv2.circle(workpiece, hole_pos, 14, (5, 5, 5), -1)

    return workpiece


def inject_surface_defects(clean_image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Injects synthetic defects into the workpiece:
    1. Linear Scratch (abrasion)
    2. Hairline Crack (fracture)
    3. Pinhole / Void (corrosion cavity)
    4. Solder Bridge (short circuit between adjacent pads)
    
    Returns:
      defective_image: Image with realistic defects
      defect_annotations: List of ground-truth defect annotations
    """
    defective = clean_image.copy()
    defects = []

    # 1. Linear scratch (bright specular streak)
    pt1 = (180, 130)
    pt2 = (270, 210)
    cv2.line(defective, pt1, pt2, (240, 240, 240), 2)
    defects.append({
        "type": "Scratch",
        "bbox": (175, 125, 100, 90),
        "severity": "HIGH",
        "confidence": 0.94
    })

    # 2. Hairline crack (dark zigzag)
    crack_pts = [(420, 180), (435, 195), (430, 215), (445, 230), (440, 250)]
    for i in range(len(crack_pts) - 1):
        cv2.line(defective, crack_pts[i], crack_pts[i + 1], (10, 10, 10), 2)
    defects.append({
        "type": "Micro-Crack",
        "bbox": (415, 175, 40, 85),
        "severity": "CRITICAL",
        "confidence": 0.97
    })

    # 3. Pinhole / Void
    cv2.circle(defective, (280, 200), 7, (0, 0, 0), -1)
    defects.append({
        "type": "Pinhole",
        "bbox": (273, 193, 14, 14),
        "severity": "MEDIUM",
        "confidence": 0.91
    })

    # 4. Solder bridge anomaly
    cv2.rectangle(defective, (555, 140), (565, 180), (20, 190, 230), -1)
    defects.append({
        "type": "Solder Bridge",
        "bbox": (550, 135, 20, 50),
        "severity": "CRITICAL",
        "confidence": 0.98
    })

    return defective, defects


def generate_stereo_pair(base_image: np.ndarray,
                         baseline_m: float = 0.1,
                         focal_length_px: float = 800.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a geometrically consistent rectified stereo pair (Left and Right views)
    and corresponding ground-truth depth map.
    """
    h, w = base_image.shape[:2]

    # Create a synthetic 3D relief depth map (component heights above base substrate)
    # Substrate plane at 1.0 meter depth
    ground_truth_depth = np.full((h, w), 1.0, dtype=np.float32)

    # IC chips raised by 5mm (Z = 0.95m)
    ground_truth_depth[120:260, 100:240] = 0.95
    ground_truth_depth[150:270, 350:530] = 0.96
    ground_truth_depth[300:380, 200:320] = 0.94

    # Calculate horizontal disparity: d = (f * B) / Z
    disparity_gt = (focal_length_px * baseline_m) / ground_truth_depth

    # Left image is the reference view
    left_img = base_image.copy()

    # Right image is shifted horizontally along epipolar lines by disparity d(x, y)
    right_img = np.zeros_like(base_image)
    for y in range(h):
        for x in range(w):
            d = int(round(disparity_gt[y, x]))
            x_right = x - d
            if 0 <= x_right < w:
                right_img[y, x_right] = base_image[y, x]

    # Fill in occlusion seams via inpainting
    mask = (right_img[:, :, 0] == 0) & (right_img[:, :, 1] == 0) & (right_img[:, :, 2] == 0)
    right_img = cv2.inpaint(right_img, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return left_img, right_img, ground_truth_depth


def generate_photometric_stereo_series(size: int = 256) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Generates 4 multi-illumination images of a 3D hemispherical / embossed surface:
    Uses Lambertian model: I_k = rho * max(0, L_k . N)
    
    Returns:
      images: 4 images under Top, Bottom, Left, Right directional illuminations
      light_vectors: (4, 3) matrix of light directions
      gt_normals: (size, size, 3) ground-truth surface normal vectors
    """
    # Create hemispherical relief surface
    y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    r_sq = x**2 + y**2
    valid = r_sq <= 0.75

    z = np.zeros((size, size), dtype=np.float64)
    z[valid] = np.sqrt(0.75 - r_sq[valid])

    # Compute ground truth surface normals
    dz_dx = np.zeros((size, size))
    dz_dy = np.zeros((size, size))
    dz_dx[valid] = -x[valid] / np.maximum(z[valid], 1e-4)
    dz_dy[valid] = -y[valid] / np.maximum(z[valid], 1e-4)

    normals = np.zeros((size, size, 3), dtype=np.float64)
    normals[:, :, 0] = -dz_dx
    normals[:, :, 1] = -dz_dy
    normals[:, :, 2] = 1.0

    # Normalize normals
    norm_len = np.linalg.norm(normals, axis=2, keepdims=True)
    gt_normals = normals / np.maximum(norm_len, 1e-6)

    # 4 Light source directions [Lx, Ly, Lz]
    light_vectors = np.array([
        [0.5, 0.0, 0.8],   # Light from East
        [-0.5, 0.0, 0.8],  # Light from West
        [0.0, 0.5, 0.8],   # Light from South
        [0.0, -0.5, 0.8]   # Light from North
    ], dtype=np.float64)
    light_vectors /= np.linalg.norm(light_vectors, axis=1, keepdims=True)

    # Diffuse reflectance (albedo)
    albedo_u8 = np.full((size, size), 215, dtype=np.uint8)
    cv2.putText(albedo_u8, "VM3D", (size // 4, size // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 100, 2)
    albedo = albedo_u8.astype(np.float64) / 255.0

    images = []
    for k in range(4):
        L = light_vectors[k]
        # Dot product N . L
        dot = gt_normals[:, :, 0] * L[0] + gt_normals[:, :, 1] * L[1] + gt_normals[:, :, 2] * L[2]
        intensity = albedo * np.maximum(0.0, dot)
        img_uint8 = (np.clip(intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
        images.append(img_uint8)

    return images, light_vectors, gt_normals


def generate_conveyor_video_sequence(num_frames: int = 10,
                                     width: int = 400,
                                     height: int = 300) -> List[np.ndarray]:
    """
    Generates a temporal sequence of frames showing a component moving horizontally
    along a moving conveyor belt with conveyor grain texture.
    """
    frames = []
    np.random.seed(42)

    for t in range(num_frames):
        # Conveyor belt texture with slight horizontal translation
        belt = np.full((height, width, 3), 70, dtype=np.uint8)
        # Add conveyor belt grain
        noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
        belt = np.clip(belt.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Moving workpiece at speed vx = 12 px/frame, vy = 2 px/frame
        x = 50 + t * 12
        y = 100 + t * 2
        w, h = 90, 60

        if x + w < width and y + h < height:
            # Component body
            cv2.rectangle(belt, (x, y), (x + w, y + h), (20, 160, 220), -1)
            cv2.rectangle(belt, (x, y), (x + w, y + h), (10, 80, 120), 2)
            # Component text
            cv2.putText(belt, "PART", (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(belt)

    return frames


def generate_defect_feature_dataset(n_samples_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates synthetic multivariate feature vectors for 4 defect classes:
      0: Scratch (high eccentricity, high gradient magnitude)
      1: Micro-Crack (high perimeter-to-area ratio, low intensity)
      2: Pinhole (high circularity, small area, high contrast)
      3: Solder Bridge (high area, low circularity, high brightness)
      
    Features: [Area, Perimeter, Circularity, Eccentricity, MeanIntensity, VarianceIntensity, HOG_Mean, Gabor_Energy]
    """
    np.random.seed(101)
    class_names = ["Scratch", "Micro-Crack", "Pinhole", "Solder Bridge"]
    X_list = []
    y_list = []

    # Defect 0: Scratch
    f0 = np.random.normal(loc=[120, 85, 0.35, 0.92, 210, 45, 0.45, 0.85],
                          scale=[15, 8, 0.05, 0.03, 10, 5, 0.05, 0.04],
                          size=(n_samples_per_class, 8))
    X_list.append(f0)
    y_list.append(np.full(n_samples_per_class, 0))

    # Defect 1: Micro-Crack
    f1 = np.random.normal(loc=[60, 110, 0.20, 0.85, 35, 60, 0.75, 0.60],
                          scale=[10, 12, 0.04, 0.05, 8, 8, 0.06, 0.05],
                          size=(n_samples_per_class, 8))
    X_list.append(f1)
    y_list.append(np.full(n_samples_per_class, 1))

    # Defect 2: Pinhole
    f2 = np.random.normal(loc=[30, 20, 0.88, 0.15, 20, 15, 0.20, 0.15],
                          scale=[5, 3, 0.04, 0.04, 5, 3, 0.03, 0.02],
                          size=(n_samples_per_class, 8))
    X_list.append(f2)
    y_list.append(np.full(n_samples_per_class, 2))

    # Defect 3: Solder Bridge
    f3 = np.random.normal(loc=[250, 90, 0.65, 0.50, 195, 30, 0.35, 0.40],
                          scale=[25, 10, 0.06, 0.08, 12, 6, 0.04, 0.05],
                          size=(n_samples_per_class, 8))
    X_list.append(f3)
    y_list.append(np.full(n_samples_per_class, 3))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    return X[indices], y[indices], class_names
