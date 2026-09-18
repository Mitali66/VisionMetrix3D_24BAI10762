"""
VisionMetrix 3D (VM3D) - Unified Command Line Interface
Usage:
  python cli.py demo
  python cli.py inspect --image <path> [--right <path>]
  python cli.py stereo --left <path> --right <path>
  python cli.py photometric --size <int>
  python cli.py motion --frames <int>
  python cli.py report
"""

import argparse
import sys
import os
import cv2
import numpy as np

from visionmetrix.pipeline.inspector import VisionMetrixInspector
from visionmetrix.data.sample_generator import (
    generate_synthetic_workpiece,
    inject_surface_defects,
    generate_stereo_pair,
    generate_photometric_stereo_series,
    generate_conveyor_video_sequence,
    generate_defect_feature_dataset,
)
from visionmetrix.stereo3d.photometric_stereo import (
    solve_photometric_stereo,
    integrate_surface_frankot_chellappa,
)
from visionmetrix.stereo3d.reconstruction import save_point_cloud_ply
from visionmetrix.pattern_motion.motion import (
    ConveyorBackgroundSubtractor,
    compute_dense_optical_flow_farneback,
)


def run_demo():
    print("=" * 70)
    print(" VISIONMETRIX 3D - AUTOMATED INDUSTRIAL METROLOGY PIPELINE DEMO")
    print("=" * 70)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print("[1/5] Synthesizing calibrated industrial workpiece & defects...")
    clean_pcb = generate_synthetic_workpiece(width=640, height=480)
    defective_pcb, defect_gt = inject_surface_defects(clean_pcb)
    left_img, right_img, gt_depth = generate_stereo_pair(defective_pcb)

    cv2.imwrite(os.path.join(output_dir, "sample_clean_pcb.png"), clean_pcb)
    cv2.imwrite(os.path.join(output_dir, "sample_defective_pcb.png"), defective_pcb)
    cv2.imwrite(os.path.join(output_dir, "stereo_left.png"), left_img)
    cv2.imwrite(os.path.join(output_dir, "stereo_right.png"), right_img)
    print("      Saved sample images to output/")

    print("[2/5] Initializing VisionMetrix Pipeline & Inspector...")
    inspector = VisionMetrixInspector(db_path="visionmetrix_inspection.db", output_dir=output_dir)
    inspector.db.start_session("SESSION_DEMO_01", operator="Quality Engineer")

    print("[3/5] Executing full multi-stage inspection on workpiece...")
    result = inspector.process_workpiece(
        image=left_img,
        item_id="WORKPIECE_SN_9042",
        session_id="SESSION_DEMO_01",
        stereo_right=right_img,
        has_vibration_blur=False
    )

    annotated_path = os.path.join(output_dir, "inspection_annotated_result.png")
    cv2.imwrite(annotated_path, result.annotated_image)
    print(f"      Status: {result.pass_status}")
    print(f"      Defects Detected: {result.defect_count}")
    for idx, d in enumerate(result.defects, 1):
        print(f"        {idx}. {d['type']} (Severity: {d['severity']}, Confidence: {d['confidence']:.2f}, Area: {d['area']:.1f}px)")
    print(f"      Hough Holes Measured: {len(result.metrology_holes)}")
    print(f"      Alignment Lines Measured: {len(result.metrology_lines)}")
    if result.point_cloud_path:
        print(f"      3D Point Cloud Exported: {result.point_cloud_path}")

    print("[4/5] Executing Multi-Illumination Photometric Stereo 3D Surface Reconstruction...")
    ps_images, light_vecs, _ = generate_photometric_stereo_series(size=256)
    normals, albedos = solve_photometric_stereo(ps_images, light_vecs)
    height_map = integrate_surface_frankot_chellappa(normals)

    # Save photometric stereo outputs
    norm_vis = ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "photometric_normals.png"), norm_vis)
    cv2.imwrite(os.path.join(output_dir, "photometric_albedo.png"), albedos)
    cv2.imwrite(os.path.join(output_dir, "photometric_height_map.png"), height_map)
    print("      Saved Photometric normal map, albedo, and height elevation map to output/")

    print("[5/5] Executing Conveyor Dynamic Motion & Optical Flow Analysis...")
    conveyor_frames = generate_conveyor_video_sequence(num_frames=6, width=320, height=240)
    subtractor = ConveyorBackgroundSubtractor()
    for f_idx, frame in enumerate(conveyor_frames):
        fg_mask, _ = subtractor.apply(frame)
    cv2.imwrite(os.path.join(output_dir, "conveyor_foreground_mask.png"), fg_mask)

    flow, flow_vis = compute_dense_optical_flow_farneback(conveyor_frames[0][:, :, 0], conveyor_frames[1][:, :, 0])
    cv2.imwrite(os.path.join(output_dir, "conveyor_optical_flow.png"), flow_vis)
    print("      Saved conveyor foreground mask and dense optical flow field to output/")

    stats = inspector.db.get_summary_statistics()
    print("-" * 70)
    print(" DATABASE AUDIT SUMMARY:")
    print(f"  Total Workpieces Inspected: {stats['total_items_inspected']}")
    print(f"  Passed Items:               {stats['passed_items']}")
    print(f"  Yield Rate:                 {stats['yield_rate_pct']:.1f}%")
    print(f"  Total Defect Records:       {stats['total_defects_found']}")
    print(f"  Defect Breakdown:           {stats['defect_distribution']}")
    print("=" * 70)
    print(" DEMO COMPLETE! All artifacts successfully saved to 'output/' folder.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VisionMetrix 3D (VM3D) - Industrial Metrology & CV Suite")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo
    subparsers.add_parser("demo", help="Run comprehensive pipeline demo with synthetic dataset")

    # Inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific workpiece image")
    inspect_parser.add_argument("--image", required=True, help="Path to input image")
    inspect_parser.add_argument("--right", default=None, help="Path to right stereo image (optional)")
    inspect_parser.add_argument("--item-id", default="CUSTOM_ITEM_001", help="Custom Item ID")

    # Photometric
    photo_parser = subparsers.add_parser("photometric", help="Run photometric stereo shape-from-shading")
    photo_parser.add_argument("--size", type=int, default=256, help="Surface grid size")

    # Report
    subparsers.add_parser("report", help="Generate the PDF project report")

    args = parser.parse_args()

    if args.command == "demo" or args.command is None:
        run_demo()
    elif args.command == "inspect":
        if not os.path.exists(args.image):
            print(f"Error: File not found {args.image}")
            sys.exit(1)
        img = cv2.imread(args.image)
        right_img = cv2.imread(args.right) if args.right else None
        inspector = VisionMetrixInspector()
        res = inspector.process_workpiece(img, item_id=args.item_id, stereo_right=right_img)
        print(f"Inspection Result: {res.pass_status} | Defects found: {res.defect_count}")
    elif args.command == "photometric":
        print(f"Running photometric stereo on {args.size}x{args.size} synthetic relief...")
        imgs, lvecs, _ = generate_photometric_stereo_series(size=args.size)
        normals, albedos = solve_photometric_stereo(imgs, lvecs)
        height = integrate_surface_frankot_chellappa(normals)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/cli_normals.png", ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8))
        cv2.imwrite("output/cli_height.png", height)
        print("Photometric reconstruction complete. Saved to output/cli_normals.png and output/cli_height.png")
    elif args.command == "report":
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_report.py"])


if __name__ == "__main__":
    main()
