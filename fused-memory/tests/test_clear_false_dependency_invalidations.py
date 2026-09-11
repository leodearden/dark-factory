"""Tests for the clear_false_dependency_invalidations one-shot repair script.

Step 11: RED tests (fail until step-12 creates the script).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""

from __future__ import annotations

import importlib.util
import logging
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


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``repair(..., apply=True)`` runs a fail-closed capability preflight before
    it touches a single edge (task 4293). That probe touches the real
    filesystem, so without this fixture every ``--apply`` test would pass or
    fail according to whether the machine running pytest happens to be able to
    write mem0's history directory -- and it genuinely cannot inside an agent
    sandbox, which is the whole reason the guard exists. This suite is
    deliberately MOCK-unit (an AsyncMock memory service, no live store), so the
    environment must not be an input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test -- to refuse,
    to record, or to pass -- so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


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


# ===========================================================================
# Tests: --apply store-mutation preflight
# ===========================================================================

class TestRunApplyStoreMutationPreflight:
    """``--apply`` refuses to START when this process cannot write mem0's store.

    Ported from ``test_sweep_toolcall_xml_leak.TestRunApplyStoreMutationPreflight``
    (task 3686), which is the in-repo precedent for this contract.

    ``repair`` clears its six edges in a sequential loop whose body is wrapped
    in a per-edge ``except Exception``. ``StoreMutationUnavailable`` subclasses
    ``RuntimeError``, so a probe placed at the in-loop ``if apply:`` gate would
    be SWALLOWED by that handler and downgraded into six ``{'status': 'error'}``
    rows -- while ``update_edge`` was still attempted six times against a store
    this process cannot write a history for. Only a run-wide probe hoisted
    above the loop, outside the swallowing ``try``, bounds that.
    """

    @staticmethod
    def _memory() -> AsyncMock:
        """AsyncMock memory service that would clear all six edges."""
        memory = AsyncMock()
        memory.update_edge = AsyncMock(
            return_value={'status': 'updated', 'verified': True}
        )
        return memory

    @staticmethod
    def _deny(monkeypatch):
        """Rig the preflight to refuse, as it would inside an agent sandbox."""
        def _raise(*_args, **_kwargs):
            raise _mod.StoreMutationUnavailable('SENTINEL-store-unwritable')

        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', _raise)

    @staticmethod
    def _fail_closed_records(caplog) -> list:
        """The guard site's OWN diagnosis.

        ``main`` has no handler at all here -- ``_run`` re-raises through its
        ``finally`` and ``asyncio.run`` lets it out, so the refusal exits the
        interpreter as an uncaught traceback -- which means this ERROR record
        is the ONLY place the operator is told what was refused and what to do
        instead. Pinned on the fail-closed marker and the remedy noun ONLY, so
        every other word of the message stays free to reword.

        Asserting on message CONTENT is deliberate, and is the narrow exception
        to the repo's don't-pin-guard-message-prose norm (task 3799): the record
        this test is about is defined BY its content -- mere record-existence
        would still pass if the whole diagnosis were replaced by "boom",
        precisely the regression this exists to catch. Verified non-vacuous:
        mutating the marker in the script turns this assertion red (task 4127
        amendment).

        NOTE the logger name is ``clear_false_dep_invalidations``, which is NOT
        the module name -- filtering on the module name would silently match
        nothing and make every assertion below vacuous.
        """
        return [
            rec for rec in caplog.records
            if rec.name == 'clear_false_dep_invalidations'
            and rec.levelname == 'ERROR'
            and 'NOT started (fail-closed)' in rec.getMessage()
            and 'MCP server' in rec.getMessage()
        ]

    @pytest.mark.asyncio
    async def test_apply_performs_zero_mutations_when_the_store_is_unwritable(
        self, monkeypatch
    ):
        """The whole point: refuse to start rather than half-complete.

        The refusal must PROPAGATE. If it were raised at the in-loop gate the
        per-edge ``except Exception`` at the mutation site would absorb it,
        ``repair`` would return a normally-shaped report carrying six
        ``{'status': 'error'}`` rows, and the caller could not tell an
        environment-level denial from six ordinary edge failures.
        """
        self._deny(monkeypatch)
        memory = self._memory()

        with pytest.raises(
            _mod.StoreMutationUnavailable, match='SENTINEL-store-unwritable'
        ):
            await _mod.repair(memory, project_id='know_live', apply=True)

        memory.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_dry_run_is_never_gated_on_write_capability(self, monkeypatch):
        """A read-only run mutates nothing, so it must not require the ability
        to mutate -- the repair report stays obtainable from anywhere, with the
        deny still installed."""
        self._deny(monkeypatch)
        memory = self._memory()

        report = await _mod.repair(memory, project_id='know_live', apply=False)

        assert report['dry_run'] is True
        assert report['cleared'] == 0
        assert len(report['edges']) == 6
        memory.update_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_apply_is_unchanged_when_the_preflight_passes(self, monkeypatch):
        """Happy path: a writable environment repairs exactly as before."""
        monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)
        memory = self._memory()

        report = await _mod.repair(memory, project_id='know_live', apply=True)

        assert memory.update_edge.await_count == 6
        assert report['dry_run'] is False
        assert report['cleared'] == 6
        assert report['edges_total'] == 6

    @pytest.mark.asyncio
    async def test_the_probe_names_the_operation_being_gated(self, monkeypatch):
        """The refusal has to be attributable in a log, so the operation string
        identifies this script and its mutating mode -- and it is probed once
        for the RUN, not once for each of the six edges."""
        calls: list[dict] = []
        monkeypatch.setattr(
            _mod, 'assert_store_mutation_allowed', lambda **kw: calls.append(kw)
        )
        memory = self._memory()

        await _mod.repair(memory, project_id='know_live', apply=True)

        assert len(calls) == 1, 'probed ONCE per run, not once per edge'
        assert 'clear_false_dependency_invalidations' in calls[0]['operation']
        assert '--apply' in calls[0]['operation']

    @pytest.mark.asyncio
    async def test_the_refusal_is_loud(self, monkeypatch, caplog):
        """The guard site logs its own fail-closed diagnosis before raising.

        Nothing downstream will: this script's ``main`` has no blanket handler,
        so without this record the operator sees a bare traceback naming an
        exception class and no remedy.
        """
        self._deny(monkeypatch)
        memory = self._memory()

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(_mod.StoreMutationUnavailable),
        ):
            await _mod.repair(memory, project_id='know_live', apply=True)

        assert self._fail_closed_records(caplog), (
            'nothing else explains this traceback -- the guard site must log '
            'the fail-closed diagnosis before raising; got: '
            f'{[rec.getMessage() for rec in caplog.records]}'
        )
        memory.update_edge.assert_not_awaited()
