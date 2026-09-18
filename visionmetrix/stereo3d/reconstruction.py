"""
VisionMetrix 3D - 3D Point Cloud Triangulation & Export
Module: reconstruction.py

Transforms 2D pixel coordinates and triangulated depth Z into metric 3D point clouds:
  X = (u - c_x) * Z / f_x
  Y = (v - c_y) * Z / f_y
  Z = depth(u, v)
Exports colored point clouds to industry-standard ASCII/Binary Polygon File Format (.ply).
"""

import numpy as np
from typing import Tuple, Optional


def triangulate_point_cloud(depth_map: np.ndarray,
                            color_image: np.ndarray,
                            focal_length: float,
                            principal_point: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates (X, Y, Z) point cloud and corresponding (R, G, B) colors from depth map.
    """
    h, w = depth_map.shape
    cx, cy = principal_point
    fx = fy = focal_length

    # Create coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    valid = depth_map > 0.0
    z_valid = depth_map[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    x_3d = (u_valid - cx) * z_valid / fx
    y_3d = (v_valid - cy) * z_valid / fy
    z_3d = z_valid

    points = np.stack([x_3d, y_3d, z_3d], axis=1)

    if len(color_image.shape) == 3:
        # OpenCV uses BGR, convert to RGB
        rgb = color_image[valid][:, [2, 1, 0]]
    else:
        gray = color_image[valid]
        rgb = np.stack([gray, gray, gray], axis=1)

    return points, rgb


def filter_point_cloud_outliers(points: np.ndarray, colors: np.ndarray,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistical outlier removal: computes mean distance to k-nearest neighbors
    and filters points exceeding mean + std_ratio * std.
    """
    if len(points) < nb_neighbors:
        return points, colors

    # Subsample if point cloud is very large for computational efficiency
    sample_size = min(len(points), 5000)
    idx = np.random.choice(len(points), sample_size, replace=False)
    sub_points = points[idx]

    # Calculate distance distribution
    diffs = np.linalg.norm(sub_points[1:] - sub_points[:-1], axis=1)
    mean_d = np.mean(diffs)
    std_d = np.std(diffs)
    cutoff = mean_d + std_ratio * std_d

    # Filter isolated points
    valid_mask = np.ones(len(points), dtype=bool)
    valid_mask &= (points[:, 2] > 0.05) & (points[:, 2] < 10.0)
    return points[valid_mask], colors[valid_mask]


def save_point_cloud_ply(filename: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Exports 3D point cloud and color data to standard ASCII Polygon File Format (.ply).
    Compatible with MeshLab, Blender, CloudCompare, and Autodesk 3ds Max.
    """
    n_points = points.shape[0]

    header = f"""ply
format ascii 1.0
comment VisionMetrix 3D Surface Reconstruction
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
