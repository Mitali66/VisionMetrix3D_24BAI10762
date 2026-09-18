"""
VisionMetrix 3D - Storage & Metadata Management
Module: database.py

Implements relational SQLite storage for industrial optical inspection logs:
- Tables:
  1. inspection_sessions: High-level session metadata, timestamp, station ID, operator.
  2. inspected_items: Physical parts passed through inspection.
  3. defect_records: Detected defects with coordinates, bounding box, area, classification, and severity.
  4. reconstruction_3d_logs: 3D point cloud outputs, depth metrics, and PLY file paths.
  5. calibration_profiles: Camera calibration matrices (K, R, t) and reprojection errors.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


class InspectionDatabase:
    def __init__(self, db_path: str = "visionmetrix_inspection.db"):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Inspection Sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspection_sessions (
                    session_id TEXT PRIMARY KEY,
                    station_id TEXT NOT NULL,
                    operator_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL
                )
            """)

            # 2. Inspected Items
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inspected_items (
                    item_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    pass_status TEXT NOT NULL,
                    defect_count INTEGER DEFAULT 0,
                    FOREIGN KEY(session_id) REFERENCES inspection_sessions(session_id)
                )
            """)

            # 3. Defect Records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_records (
                    defect_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    area_px REAL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES inspected_items(item_id)
                )
            """)

            # 4. 3D Reconstruction Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reconstruction_3d_logs (
                    recon_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    ply_filepath TEXT NOT NULL,
                    vertex_count INTEGER NOT NULL,
                    mean_depth_mm REAL,
                    max_depth_mm REAL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES inspected_items(item_id)
                )
            """)

            # 5. Calibration Profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calibration_profiles (
                    profile_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    rmse_pixels REAL NOT NULL,
                    intrinsic_matrix_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def start_session(self, session_id: str, station_id: str = "STATION_01",
                      operator: str = "Lead QA Engineer") -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO inspection_sessions (session_id, station_id, operator_name, start_time, status)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, station_id, operator, datetime.now().isoformat(), "RUNNING"))
            conn.commit()

    def record_item(self, item_id: str, session_id: str, item_type: str,
                    pass_status: str, defect_count: int) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO inspected_items (item_id, session_id, item_type, timestamp, pass_status, defect_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (item_id, session_id, item_type, datetime.now().isoformat(), pass_status, defect_count))
            conn.commit()

    def log_defect(self, item_id: str, defect_type: str, confidence: float,
                   bbox: Tuple, area: float, severity: str = "MEDIUM") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO defect_records (item_id, defect_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, area_px, severity, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (item_id, defect_type, confidence, bbox[0], bbox[1], bbox[2], bbox[3], area, severity, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid

    def log_3d_reconstruction(self, item_id: str, ply_path: str,
                              vertex_count: int, mean_depth: float, max_depth: float) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reconstruction_3d_logs (item_id, ply_filepath, vertex_count, mean_depth_mm, max_depth_mm, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (item_id, ply_path, vertex_count, mean_depth, max_depth, datetime.now().isoformat()))
            conn.commit()
            return cursor.lastrowid

    def log_calibration(self, profile_id: str, camera_id: str, rmse: float, K: Any) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO calibration_profiles (profile_id, camera_id, rmse_pixels, intrinsic_matrix_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (profile_id, camera_id, rmse, json.dumps(K.tolist()), datetime.now().isoformat()))
            conn.commit()

    def get_summary_statistics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inspected_items")
            total_items = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM inspected_items WHERE pass_status = 'PASS'")
            passed_items = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM defect_records")
            total_defects = cursor.fetchone()[0]

            cursor.execute("SELECT defect_type, COUNT(*) as cnt FROM defect_records GROUP BY defect_type")
            defect_dist = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_items_inspected": total_items,
                "passed_items": passed_items,
                "yield_rate_pct": (passed_items / max(1, total_items)) * 100.0,
                "total_defects_found": total_defects,
                "defect_distribution": defect_dist
            }
