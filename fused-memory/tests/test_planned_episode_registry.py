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


class TestIdempotentInitialize:
    """step-25/26: initialize() must be idempotent — safe to call twice."""

    @pytest.mark.asyncio
    async def test_double_initialize_preserves_connection_identity(self, tmp_path):
        """Second call to initialize() must return early, keeping original connection.

        Currently FAILS because initialize() unconditionally overwrites self._db.
        """
        reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
        await reg.initialize()

        # Register data so we can verify it's still accessible afterwards
        await reg.register('uuid-before', 'proj-1')

        # Capture the original connection object
        original_db = reg._db

        # Second call — must be a no-op (early return)
        await reg.initialize()

        # Connection identity must be preserved (no new connection opened)
        assert reg._db is original_db, (
            'Second initialize() must not open a new connection (would leak the old one)'
        )

        # Data registered before the second call must still be accessible
        assert await reg.is_planned('uuid-before') is True

        await reg.close()

    @pytest.mark.asyncio
    async def test_double_initialize_data_still_accessible(self, tmp_path):
        """Data registered before the second initialize() is still retrievable after."""
        reg = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
        await reg.initialize()
        await reg.register('uuid-X', 'proj-1')
        await reg.register('uuid-Y', 'proj-1')

        # Second call
        await reg.initialize()

        # All registered data remains accessible
        uuids = await reg.get_planned_uuids('proj-1')
        assert uuids == {'uuid-X', 'uuid-Y'}

        await reg.close()


class TestRapidCallStability:
    """step-27/28: Cursor resource-leak regression — methods must remain stable
    under rapid repeated calls (regression coverage for async-with refactor)."""

    @pytest.mark.asyncio
    async def test_is_planned_rapid_calls(self, registry):
        """is_planned() must return correct results across 50 rapid calls."""
        await registry.register('uuid-rapid', 'proj-r')
        for _ in range(50):
            result = await registry.is_planned('uuid-rapid')
            assert result is True
        for _ in range(50):
            result = await registry.is_planned('uuid-not-there')
            assert result is False

    @pytest.mark.asyncio
    async def test_get_planned_uuids_rapid_calls(self, registry):
        """get_planned_uuids() must return a consistent set across 50 rapid calls."""
        uuids = {f'uuid-r{i}' for i in range(5)}
        for uid in uuids:
            await registry.register(uid, 'proj-r')
        for _ in range(50):
            result = await registry.get_planned_uuids('proj-r')
            assert result == uuids

    @pytest.mark.asyncio
    async def test_are_all_planned_rapid_calls(self, registry):
        """are_all_planned() must return correct results across 50 rapid calls."""
        await registry.register('uuid-a', 'proj-r')
        await registry.register('uuid-b', 'proj-r')
        for _ in range(50):
            assert await registry.are_all_planned(['uuid-a', 'uuid-b']) is True
        for _ in range(50):
            assert await registry.are_all_planned(['uuid-a', 'uuid-missing']) is False
