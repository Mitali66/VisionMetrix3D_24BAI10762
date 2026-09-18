"""
Unit Tests - End-to-End Metrology Pipeline & Database
"""
import pytest
import os
import numpy as np

from visionmetrix.pipeline.inspector import VisionMetrixInspector
from visionmetrix.data.sample_generator import (
    generate_synthetic_workpiece,
    inject_surface_defects,
    generate_stereo_pair,
)
from visionmetrix.storage.database import InspectionDatabase


def test_end_to_end_inspection_pipeline(tmp_path):
    db_file = os.path.join(tmp_path, "test_inspection.db")
    out_dir = os.path.join(tmp_path, "output")

    inspector = VisionMetrixInspector(db_path=db_file, output_dir=out_dir)

    # Generate workpiece
    clean_pcb = generate_synthetic_workpiece(width=480, height=360)
    defective_pcb, defect_gt = inject_surface_defects(clean_pcb)
    left_img, right_img, _ = generate_stereo_pair(defective_pcb)

    # Start session
    inspector.db.start_session("TEST_SESSION_01")

    # Run inspection
    result = inspector.process_workpiece(
        image=left_img,
        item_id="PCB_UNIT_042",
        session_id="TEST_SESSION_01",
        stereo_right=right_img,
        has_vibration_blur=False
    )

    assert result.item_id == "PCB_UNIT_042"
    assert result.pass_status in ["PASS", "FAIL"]
    assert result.annotated_image.shape == left_img.shape
    assert len(result.metrology_holes) > 0

    if result.point_cloud_path is not None:
        assert os.path.exists(result.point_cloud_path)

    # Verify SQLite database records
    summary = inspector.db.get_summary_statistics()
    assert summary["total_items_inspected"] >= 1
    assert "yield_rate_pct" in summary
