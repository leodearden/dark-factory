"""Shared test fixtures for dashboard tests."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient


@pytest.fixture()
def dashboard_config(tmp_path):
    """Create a DashboardConfig with tmp_path-based project_root."""
    from dashboard.config import DashboardConfig

    return DashboardConfig(project_root=tmp_path)


@pytest.fixture()
def client():
    """Create a TestClient for the dashboard FastAPI app."""
    from dashboard.app import app

    with TestClient(app) as c:
        yield c
