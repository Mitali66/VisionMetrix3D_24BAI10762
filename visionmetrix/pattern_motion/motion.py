"""
VisionMetrix 3D - Motion Analysis & Temporal Tracking
Module: motion.py

Covers:
1. Dynamic Background Subtraction (MOG2 Gaussian Mixture Modeling) for conveyor belt object segmentation.
2. Sparse Optical Flow: Lucas-Kanade differential method (KLT tracking of high-confidence corners).
     A * v = b where A = [ sum Ix^2, sum Ix*Iy; sum Ix*Iy, sum Iy^2 ], b = - [ sum Ix*It; sum Iy*It ]
3. Dense Optical Flow: Gunnar Farneback polynomial expansion method.
4. Spatio-temporal motion velocity and trajectory estimation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class ConveyorBackgroundSubtractor:
    """
    Gaussian Mixture Model (MOG2) Background Subtractor with shadow suppression
    and morphological opening/closing for noise-free conveyor object detection.
    """
    def __init__(self, history: int = 100, var_threshold: float = 25.0, detect_shadows: bool = True):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def apply(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          fg_mask: Cleaned binary foreground mask (255 for moving parts, 0 for conveyor belt)
          raw_mask: Raw output from MOG2 (including shadows at gray value 127)
        """
        raw_mask = self.subtractor.apply(frame)

        # Shadows are marked as 127; threshold to isolate true moving foreground (255)
        _, thresh = cv2.threshold(raw_mask, 250, 255, cv2.THRESH_BINARY)

        # Morphological opening (remove noise) and closing (fill holes)
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.kernel)

        return clean_mask, raw_mask


def track_sparse_optical_flow_klt(prev_gray: np.ndarray,
                                  curr_gray: np.ndarray,
                                  prev_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative Lucas-Kanade Optical Flow with Pyramids (KLT Tracker).
    
    Returns:
      tracked_pts: New (x, y) coordinates of successfully tracked keypoints
      status: Array indicating tracking success (1 for tracked, 0 for lost)
      error: Tracking residual error
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    p0 = prev_pts.astype(np.float32)
    if len(p0.shape) == 2:
        p0 = p0[:, np.newaxis, :]

    p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    return p1.reshape(-1, 2), status.flatten(), err.flatten()


def compute_dense_optical_flow_farneback(prev_gray: np.ndarray,
                                         curr_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes dense optical flow field using the Gunnar Farneback polynomial algorithm.
    
    Returns:
      flow: (H, W, 2) vector field containing (dx, dy) velocities
      flow_vis_bgr: Color-coded BGR visualization (Hue encodes angle, Value encodes speed)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Visualize flow using HSV color wheel
    hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_vis_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow, flow_vis_bgr


def estimate_motion_trajectory(tracked_points_history: List[np.ndarray]) -> Dict[str, Any]:
    """
    Estimates average velocity, speed, and trajectory drift of moving parts.
    """
    if len(tracked_points_history) < 2:
        return {"mean_speed": 0.0, "mean_velocity_x": 0.0, "mean_velocity_y": 0.0}

    velocities_x = []
    velocities_y = []

    for t in range(len(tracked_points_history) - 1):
        p_prev = tracked_points_history[t]
        p_curr = tracked_points_history[t + 1]
        n = min(len(p_prev), len(p_curr))
        if n > 0:
            dx = p_curr[:n, 0] - p_prev[:n, 0]
            dy = p_curr[:n, 1] - p_prev[:n, 1]
            velocities_x.extend(dx.tolist())
            velocities_y.extend(dy.tolist())

    vx = float(np.mean(velocities_x)) if velocities_x else 0.0
    vy = float(np.mean(velocities_y)) if velocities_y else 0.0
    speed = float(np.sqrt(vx**2 + vy**2))

    return {
        "mean_speed_px_per_frame": speed,
        "mean_velocity_x": vx,
        "mean_velocity_y": vy,
        "trajectory_angle_deg": float(np.rad2deg(np.arctan2(vy, vx)))
    }
