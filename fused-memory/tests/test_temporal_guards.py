"""Integration tests for temporal guards — planned episode filtering pipeline."""

from __future__ import annotations

import pytest

from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry


@pytest.fixture
async def registry(tmp_path):
    """PlannedEpisodeRegistry backed by a real SQLite DB."""
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
    await reg.initialize()
    yield reg
    await reg.close()
