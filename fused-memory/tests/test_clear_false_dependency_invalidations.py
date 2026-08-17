"""Tests for the clear_false_dependency_invalidations one-shot repair script.

Step 11: RED tests (fail until step-12 creates the script).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'clear_false_dependency_invalidations.py'
)


def _load_module() -> types.ModuleType:
    """Load clear_false_dependency_invalidations.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    mod_name = 'clear_false_dependency_invalidations'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()

# The 6 falsely-invalidated dependency edge UUIDs to be cleared.
EXPECTED_UUIDS = {
    '001060dd-71d7-43f0-b1cd-6e29e60fb351',
    'f30b8f29-9ff0-4f78-938b-dd600d13e5d1',
    '4bd30efc-7437-45b9-a0e6-08243c52c3c2',
    '5f44fe41-65a9-44a1-b80f-bf3124043151',
    'db0a4a8c-0995-4330-9086-478cebc376fc',
    'dc96732e-f784-476c-a417-d0fe3fe2bf62',
}


class TestClearFalseDependencyInvalidations:
    """Repair script exposes TARGET_EDGE_UUIDS and repair() behaves correctly."""

    def test_module_exposes_target_edge_uuids(self):
        """TARGET_EDGE_UUIDS must have 6 entries, each a well-formed UUID (8-4-4-4-12)."""
        uuid_re = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        )
        target_edge_uuids = _mod.TARGET_EDGE_UUIDS
        assert len(target_edge_uuids) == 6, (
            f'Expected exactly 6 UUIDs, got {len(target_edge_uuids)}'
        )
        for uuid in target_edge_uuids:
            assert uuid_re.match(uuid), (
                f'Malformed UUID in TARGET_EDGE_UUIDS: {uuid!r}'
            )

    @pytest.mark.asyncio
    async def test_repair_apply_calls_update_edge_for_each_uuid(self):
        """repair(apply=True) calls mock_memory.update_edge once per target edge."""
        mock_memory = AsyncMock()
        mock_memory.update_edge = AsyncMock(
            return_value={'status': 'updated', 'verified': True}
        )

        await _mod.repair(mock_memory, project_id='know_live', apply=True)

        assert mock_memory.update_edge.await_count == 6, (
            f'Expected 6 update_edge calls, got {mock_memory.update_edge.await_count}'
        )
        # Check every UUID was cleared
        called_uuids = {
            kw['edge_uuid']
            for _, kw in mock_memory.update_edge.call_args_list
        }
        assert called_uuids == EXPECTED_UUIDS, (
            f'Expected all 6 UUIDs to be cleared; got {called_uuids}'
        )
        # Check clear_invalid_at=True on every call
        for _args, kwargs in mock_memory.update_edge.call_args_list:
            assert kwargs.get('clear_invalid_at') is True, (
                f'Expected clear_invalid_at=True on all calls; got {kwargs}'
            )
            assert kwargs.get('project_id') == 'know_live', (
                f'Expected project_id=know_live; got {kwargs}'
            )

    @pytest.mark.asyncio
    async def test_repair_dry_run_makes_no_calls(self):
        """repair(apply=False) makes ZERO update_edge calls and reports dry_run=True."""
        mock_memory = AsyncMock()
        mock_memory.update_edge = AsyncMock()

        report = await _mod.repair(mock_memory, project_id='know_live', apply=False)

        mock_memory.update_edge.assert_not_awaited()
        assert report.get('dry_run') is True, (
            f'Expected dry_run=True in report; got {report}'
        )
        assert len(report.get('edges', [])) == 6, (
            f'Expected 6 listed edges; got {report}'
        )
