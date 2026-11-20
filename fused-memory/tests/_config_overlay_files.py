"""Builders for config overlay test files (fused_memory/config/overlay.py)."""

import os
from pathlib import Path
from typing import Any

import yaml


def write_overlay(path: Path, data: dict[str, Any], *, mtime: int) -> Path:
    """Write ``data`` as overlay YAML with an explicit mtime, so two rewrites
    inside the filesystem's timestamp granularity still carry distinct provenance."""
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    os.utime(path, (mtime, mtime))
    return path


def calibration(report: str, **cutoffs: float) -> dict[str, Any]:
    """One write_triage_project_calibrations entry."""
    return {'calibration_report_path': report, 't_high_by_category': dict(cutoffs)}
