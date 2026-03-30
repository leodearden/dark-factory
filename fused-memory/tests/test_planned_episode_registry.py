"""Tests for PlannedEpisodeRegistry — SQLite-backed planned episode tracker."""

from __future__ import annotations

import pytest

from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry


@pytest.fixture
async def registry(tmp_path):
    """PlannedEpisodeRegistry backed by a real (in-memory) SQLite DB."""
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
    await reg.initialize()
    yield reg
    await reg.close()
