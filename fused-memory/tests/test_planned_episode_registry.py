"""Tests for PlannedEpisodeRegistry — SQLite-backed planned episode tracker."""

from __future__ import annotations

import pytest
import pytest_asyncio

from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry


@pytest_asyncio.fixture
async def registry(tmp_path):
    """PlannedEpisodeRegistry backed by a real (in-memory) SQLite DB."""
    reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
    await reg.initialize()
    yield reg
    await reg.close()


class TestPromoteAndAreAllPlanned:
    @pytest.mark.asyncio
    async def test_promote_removes_from_planned(self, registry):
        await registry.register('uuid-promote', 'proj-1')
        assert await registry.is_planned('uuid-promote') is True
        await registry.promote('uuid-promote')
        assert await registry.is_planned('uuid-promote') is False

    @pytest.mark.asyncio
    async def test_promote_nonexistent_is_no_op(self, registry):
        """Promoting a UUID that was never registered should not raise."""
        await registry.promote('uuid-never-existed')

    @pytest.mark.asyncio
    async def test_are_all_planned_all_planned(self, registry):
        await registry.register('uuid-A', 'proj-1')
        await registry.register('uuid-B', 'proj-1')
        assert await registry.are_all_planned(['uuid-A', 'uuid-B']) is True

    @pytest.mark.asyncio
    async def test_are_all_planned_empty_returns_false(self, registry):
        assert await registry.are_all_planned([]) is False

    @pytest.mark.asyncio
    async def test_are_all_planned_mixed_returns_false(self, registry):
        """If some are planned and some are not, result is False."""
        await registry.register('uuid-planned', 'proj-1')
        assert await registry.are_all_planned(['uuid-planned', 'uuid-not-planned']) is False

    @pytest.mark.asyncio
    async def test_are_all_planned_none_planned_returns_false(self, registry):
        assert await registry.are_all_planned(['uuid-not-planned-1']) is False

    @pytest.mark.asyncio
    async def test_promote_then_are_all_planned_returns_false(self, registry):
        """After promoting an episode, it should no longer count as planned."""
        await registry.register('uuid-C', 'proj-1')
        await registry.register('uuid-D', 'proj-1')
        await registry.promote('uuid-C')
        assert await registry.are_all_planned(['uuid-C', 'uuid-D']) is False


class TestRegisterAndCheck:
    @pytest.mark.asyncio
    async def test_register_then_is_planned_returns_true(self, registry):
        await registry.register('uuid-aaa', 'proj-1')
        assert await registry.is_planned('uuid-aaa') is True

    @pytest.mark.asyncio
    async def test_unregistered_uuid_is_not_planned(self, registry):
        assert await registry.is_planned('uuid-unknown') is False

    @pytest.mark.asyncio
    async def test_idempotent_reregister(self, registry):
        """Registering the same UUID twice should not raise."""
        await registry.register('uuid-bbb', 'proj-1')
        await registry.register('uuid-bbb', 'proj-1')
        assert await registry.is_planned('uuid-bbb') is True

    @pytest.mark.asyncio
    async def test_get_planned_uuids_returns_project_set(self, registry):
        await registry.register('uuid-001', 'proj-a')
        await registry.register('uuid-002', 'proj-a')
        await registry.register('uuid-003', 'proj-b')

        uuids_a = await registry.get_planned_uuids('proj-a')
        assert uuids_a == {'uuid-001', 'uuid-002'}

    @pytest.mark.asyncio
    async def test_get_planned_uuids_project_isolation(self, registry):
        """UUIDs from one project don't appear in another project's set."""
        await registry.register('uuid-x', 'proj-x')
        uuids_y = await registry.get_planned_uuids('proj-y')
        assert 'uuid-x' not in uuids_y

    @pytest.mark.asyncio
    async def test_get_planned_uuids_empty_project(self, registry):
        result = await registry.get_planned_uuids('proj-empty')
        assert result == set()
