"""
VisionMetrix 3D - Geometric Metrology & Hough Transform
Module: hough.py

Covers:
1. Standard & Probabilistic Hough Line Transform in polar coordinate space (rho, theta):
     rho = x * cos(theta) + y * sin(theta)
2. Hough Circle Transform for dimensional metrology of circular holes, drill vias, and gears:
     (x - a)^2 + (y - b)^2 = r^2
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def detect_hough_lines(edge_map: np.ndarray,
                       rho_res: float = 1.0,
                       theta_res: float = np.pi / 180.0,
                       threshold: int = 80,
                       min_line_length: float = 30.0,
                       max_line_gap: float = 10.0) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Detects line segments using Probabilistic Hough Transform (HoughLinesP).
    
    Returns:
      lines: List of line segments [(x1, y1, x2, y2), ...]
      overlay: Visual canvas with detected lines highlighted in neon green
    """
    h, w = edge_map.shape[:2]
    overlay = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR) if len(edge_map.shape) == 2 else edge_map.copy()

    raw_lines = cv2.HoughLinesP(edge_map, rho=rho_res, theta=theta_res,
                                threshold=threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)

    detected_segments = []
    if raw_lines is not None:
        for line in raw_lines:
            coords = line.flatten()
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            detected_segments.append((x1, y1, x2, y2))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

    return detected_segments, overlay


def detect_hough_circles(gray_image: np.ndarray,
                         dp: float = 1.2,
                         min_dist: float = 30.0,
                         param1: float = 100.0,
                         param2: float = 30.0,
                         min_radius: int = 10,
                         max_radius: int = 150) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Detects circular features via Hough Gradient Method for component metrology.
    
    Returns:
      circles: List of circles [(center_x, center_y, radius), ...]
      overlay: Visual canvas with circle centers and perimeters drawn
    """
    overlay = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if len(gray_image.shape) == 2 else gray_image.copy()
    if len(gray_image.shape) == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image.copy()

    # Pre-blur to prevent false positive noisy edge peaks
    blurred = cv2.medianBlur(gray, 5)

    raw_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                                   param1=param1, param2=param2,
                                   minRadius=min_radius, maxRadius=max_radius)

    detected_circles = []
    if raw_circles is not None:
        circles_arr = np.round(raw_circles[0, :]).astype(int)
        for (x, y, r) in circles_arr:
            detected_circles.append((int(x), int(y), int(r)))
            # Outer circle boundary in cyan
            cv2.circle(overlay, (x, y), r, (255, 255, 0), 2, cv2.LINE_AA)
            # Center point in red
            cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return detected_circles, overlay
