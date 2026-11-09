"""quality.db: where it lives, its tables, and the LandingRow / sample codec over them."""

from __future__ import annotations

from pathlib import Path

from dashboard.config import DashboardConfig


def test_quality_db_sits_beside_the_other_sampler_stores(tmp_path: Path) -> None:
    config = DashboardConfig(project_root=tmp_path)

    assert config.quality_db == config.project_root / 'data' / 'burndown' / 'quality.db'
    assert config.quality_db.parent == config.burndown_db.parent == config.metrics_db.parent
