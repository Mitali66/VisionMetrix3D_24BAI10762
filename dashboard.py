"""
VisionMetrix 3D - Interactive Desktop Dashboard
Module: dashboard.py

A modern Tkinter graphical user interface providing visual inspection controls,
interactive parameter tuning (Canny, Hough, CLAHE, Wiener), 3D reconstruction visualization,
and real-time inspection database querying.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys

from visionmetrix.pipeline.inspector import VisionMetrixInspector
from visionmetrix.data.sample_generator import (
    generate_synthetic_workpiece,
    inject_surface_defects,
    generate_stereo_pair,
    generate_photometric_stereo_series,
)
from visionmetrix.lowlevel.enhancement import clahe_enhancement, gamma_correction
from visionmetrix.lowlevel.fourier import compute_fft2d
from visionmetrix.features.edge_detectors import canny_edge_detector_custom
from visionmetrix.features.hough import detect_hough_circles, detect_hough_lines
from visionmetrix.features.corners import harris_corner_detector
from visionmetrix.stereo3d.photometric_stereo import solve_photometric_stereo, integrate_surface_frankot_chellappa
from visionmetrix.storage.database import InspectionDatabase


class VisionMetrixDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VisionMetrix 3D - Industrial Inspection & 3D Metrology Suite")
        self.geometry("1180x760")
        self.minsize(980, 650)

        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        self.inspector = VisionMetrixInspector(db_path="visionmetrix_inspection.db", output_dir="output")
        self.db = InspectionDatabase("visionmetrix_inspection.db")

        # Active image caches
        self.curr_clean = generate_synthetic_workpiece(520, 390)
        self.curr_defective, _ = inject_surface_defects(self.curr_clean)
        self.left_img, self.right_img, _ = generate_stereo_pair(self.curr_defective)

        self._build_ui()
        self._refresh_preview()

    def _build_ui(self):
        # Top Header Banner
        header_frame = tk.Frame(self, bg="#1a237e", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="VisionMetrix 3D | Industrial Spatial Metrology & Surface Quality Suite",
            fg="white", bg="#1a237e", font=("Segoe UI", 15, "bold")
        )
        title_lbl.pack(side=tk.LEFT, padx=20, pady=12)

        status_btn = ttk.Button(header_frame, text="Generate PDF Report", command=self._on_generate_report)
        status_btn.pack(side=tk.RIGHT, padx=20, pady=12)

        # Notebook tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Live Metrology Pipeline
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="Full Metrology Pipeline")
        self._setup_pipeline_tab()

        # Tab 2: Interactive Feature Tuning
        self.tab_features = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_features, text="Feature Metrology & Tuning")
        self._setup_features_tab()

        # Tab 3: Photometric Stereo 3D
        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="3D Photometric Reconstruction")
        self._setup_3d_tab()

        # Tab 4: Database & Audit Logs
        self.tab_db = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_db, text="Database Audit Logs")
        self._setup_db_tab()

    def _setup_pipeline_tab(self):
        left_controls = ttk.LabelFrame(self.tab_pipeline, text="Inspection Controls", width=260)
        left_controls.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(left_controls, text="Run Full Inspection", command=self._run_inspection_pipeline).pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(left_controls, text="Generate New Workpiece", command=self._generate_new_workpiece).pack(fill=tk.X, padx=10, pady=8)

        self.lbl_status = ttk.Label(left_controls, text="Status: Ready", font=("Segoe UI", 11, "bold"))
        self.lbl_status.pack(pady=10)

        self.txt_pipeline_log = tk.Text(left_controls, width=32, height=18, bg="#f5f5f5", font=("Consolas", 9))
        self.txt_pipeline_log.pack(padx=8, pady=5, fill=tk.BOTH, expand=True)

        # Image view panel
        img_panel = ttk.LabelFrame(self.tab_pipeline, text="Live Inspection View")
        img_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_pipeline = tk.Label(img_panel, bg="#202020")
        self.canvas_pipeline.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _setup_features_tab(self):
        ctrl = ttk.LabelFrame(self.tab_features, text="Parameter Sliders", width=280)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl, text="Canny Low Threshold:").pack(anchor=tk.W, padx=10, pady=2)
        self.slider_canny_low = ttk.Scale(ctrl, from_=10, to_=150, value=50, command=lambda e: self._update_features())
        self.slider_canny_low.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(ctrl, text="Canny High Threshold:").pack(anchor=tk.W, padx=10, pady=2)
        self.slider_canny_high = ttk.Scale(ctrl, from_=50, to_=250, value=120, command=lambda e: self._update_features())
        self.slider_canny_high.pack(fill=tk.X, padx=10, pady=4)

        ttk.Label(ctrl, text="CLAHE Clip Limit:").pack(anchor=tk.W, padx=10, pady=2)
        self.slider_clahe = ttk.Scale(ctrl, from_=1.0, to_=6.0, value=2.0, command=lambda e: self._update_features())
        self.slider_clahe.pack(fill=tk.X, padx=10, pady=4)

        self.feature_mode = tk.StringVar(value="Canny Edges")
        ttk.Label(ctrl, text="Visualization Mode:").pack(anchor=tk.W, padx=10, pady=6)
        modes = ["Canny Edges", "Hough Circles", "Harris Corners", "FFT Spectrum"]
        for m in modes:
            ttk.Radiobutton(ctrl, text=m, value=m, variable=self.feature_mode, command=self._update_features).pack(anchor=tk.W, padx=20, pady=2)

        view = ttk.LabelFrame(self.tab_features, text="Interactive Feature Output")
        view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_features = tk.Label(view, bg="#202020")
        self.canvas_features.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _setup_3d_tab(self):
        ctrl = ttk.LabelFrame(self.tab_3d, text="3D Reconstruction Controls", width=260)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(ctrl, text="Compute Photometric Normals & Height", command=self._run_photometric_3d).pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(ctrl, text="Reconstructs fine surface micro-relief\nand surface normal vectors\nusing 4-light directional Lambertian\nreflectance model.", font=("Segoe UI", 9)).pack(padx=10, pady=5)

        view = ttk.LabelFrame(self.tab_3d, text="Photometric 3D Elevation Map")
        view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas_3d = tk.Label(view, bg="#202020")
        self.canvas_3d.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _setup_db_tab(self):
        ctrl = ttk.Frame(self.tab_db)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Button(ctrl, text="Refresh Database Records", command=self._refresh_db_table).pack(side=tk.LEFT, padx=5)

        cols = ("item_id", "session_id", "defect_type", "severity", "confidence", "area_px", "timestamp")
        self.tree_db = ttk.Treeview(self.tab_db, columns=cols, show="headings")
        for c in cols:
            self.tree_db.heading(c, text=c.replace("_", " ").title())
            self.tree_db.column(c, width=120)
        self.tree_db.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._refresh_db_table()

    def _refresh_preview(self):
        rgb = cv2.cvtColor(self.curr_defective, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((640, 480))
        img_tk = ImageTk.PhotoImage(im)
        self.canvas_pipeline.configure(image=img_tk)
        self.canvas_pipeline.image = img_tk

    def _generate_new_workpiece(self):
        self.curr_clean = generate_synthetic_workpiece(520, 390)
        self.curr_defective, _ = inject_surface_defects(self.curr_clean)
        self.left_img, self.right_img, _ = generate_stereo_pair(self.curr_defective)
        self._refresh_preview()
        self.lbl_status.config(text="Status: New Part Generated")

    def _run_inspection_pipeline(self):
        import uuid
        item_id = f"PCB_{uuid.uuid4().hex[:6].upper()}"
        res = self.inspector.process_workpiece(self.left_img, item_id=item_id, stereo_right=self.right_img)

        # Update display
        rgb = cv2.cvtColor(res.annotated_image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((640, 480))
        img_tk = ImageTk.PhotoImage(im)
        self.canvas_pipeline.configure(image=img_tk)
        self.canvas_pipeline.image = img_tk

        self.lbl_status.config(text=f"Status: {res.pass_status}")
        self.txt_pipeline_log.delete("1.0", tk.END)
        self.txt_pipeline_log.insert(tk.END, f"ITEM: {res.item_id}\n")
        self.txt_pipeline_log.insert(tk.END, f"RESULT: {res.pass_status}\n")
        self.txt_pipeline_log.insert(tk.END, f"DEFECTS FOUND: {res.defect_count}\n")
        for d in res.defects:
            self.txt_pipeline_log.insert(tk.END, f"- {d['type']} ({d['severity']})\n")
        self.txt_pipeline_log.insert(tk.END, f"HOLES MEASURED: {len(res.metrology_holes)}\n")
        self.txt_pipeline_log.insert(tk.END, f"LINES MEASURED: {len(res.metrology_lines)}\n")
        self._refresh_db_table()

    def _update_features(self):
        mode = self.feature_mode.get()
        low = int(self.slider_canny_low.get())
        high = int(self.slider_canny_high.get())
        clip = float(self.slider_clahe.get())

        enhanced = clahe_enhancement(self.curr_defective, clip_limit=clip)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        if mode == "Canny Edges":
            edges, _, _ = canny_edge_detector_custom(gray, low_threshold=low, high_threshold=high)
            vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        elif mode == "Hough Circles":
            _, vis = detect_hough_circles(enhanced, min_radius=10, max_radius=40)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        elif mode == "Harris Corners":
            _, _, vis = harris_corner_detector(enhanced)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        elif mode == "FFT Spectrum":
            _, mag, _ = compute_fft2d(gray)
            vis = cv2.applyColorMap(mag, cv2.COLORMAP_VIRIDIS)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(vis)
        im.thumbnail((640, 480))
        img_tk = ImageTk.PhotoImage(im)
        self.canvas_features.configure(image=img_tk)
        self.canvas_features.image = img_tk

    def _run_photometric_3d(self):
        imgs, lvecs, _ = generate_photometric_stereo_series(size=256)
        normals, albedos = solve_photometric_stereo(imgs, lvecs)
        height = integrate_surface_frankot_chellappa(normals)

        height_color = cv2.applyColorMap(height, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(height_color, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((500, 500))
        img_tk = ImageTk.PhotoImage(im)
        self.canvas_3d.configure(image=img_tk)
        self.canvas_3d.image = img_tk

    def _refresh_db_table(self):
        for row in self.tree_db.get_children():
            self.tree_db.delete(row)
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT d.item_id, i.session_id, d.defect_type, d.severity,
                           d.confidence, d.area_px, d.timestamp
                    FROM defect_records d
                    JOIN inspected_items i ON d.item_id = i.item_id
                    ORDER BY d.defect_id DESC LIMIT 50
                """)
                for r in cursor.fetchall():
                    self.tree_db.insert("", tk.END, values=(
                        r[0], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.1f}", r[6][:19]
                    ))
        except Exception:
            pass

    def _on_generate_report(self):
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/generate_report.py"], check=True)
            messagebox.showinfo("Report Ready", "Academic Project Report successfully generated as 'Project_Report.pdf'!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")


if __name__ == "__main__":
    app = VisionMetrixDashboard()
    app.mainloop()
