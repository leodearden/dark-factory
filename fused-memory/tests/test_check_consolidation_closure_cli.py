"""ACCEPTANCE 2 (task 4808): the operator CLI derives `unstamped_live_ids` too.

WHY THIS FILE LIVES HERE AND NOT IN ``scripts/tests/``.
``scripts/check_consolidation_closure.py`` imports ``TaskInterceptor`` at
MODULE scope — that is the INV-5 single-sourcing of
``_CONSOLIDATION_SCROLL_LIMIT`` and ``_extract_metadata_dict`` which makes
"the same mechanical check" literally true rather than aspirational.  That
import transitively pulls in ``services.memory_service`` and therefore
``graphiti_core``, which is absent from the ``shared`` venv that runs
``scripts/tests/``.  Measured: ``uv run --project shared python -c "import
check_consolidation_closure"`` dies with ``ModuleNotFoundError: No module
named 'graphiti_core'``.  The fused-memory venv has it, so the script is
testable here and only here.

The script is driven through its own ``run(args, *, memory=...)`` injection
seam with a fake store, so no Qdrant is required.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    evaluate_closure,
)

_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = _ROOT / 'scripts' / 'check_consolidation_closure.py'


def _load_module() -> types.ModuleType:
    """Load the script from its path, with ``scripts/`` on ``sys.path``.

    The ``sys.path`` insert is REQUIRED, not tidiness: the script imports its
    flat sibling ``_task_db_scan``, which normally resolves only from a
    DIRECTLY-EXECUTED script's ``sys.path[0]``.
    """
    scripts_dir = str(_ROOT / 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    mod_name = 'check_consolidation_closure'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


mod = _load_module()

_TOPIC = 'cli-demo-topic'
_PROJECT_ID = 'dark_factory'


def _uuid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def _member(mid, *, canonical=None, supersedes=None):
    meta = {'topic': _TOPIC}
    if canonical is not None:
        meta['canonical'] = canonical
    if supersedes is not None:
        meta['supersedes'] = supersedes
    return {'id': mid, 'created_at': '2026-08-24T00:00:00+00:00', 'metadata': meta}


_WELL_FORMED = [_member(_uuid(1), canonical=True), _member(_uuid(2))]


def _gate_blob(observed=None, members_override=None):
    block: dict[str, Any] = {'topic': _TOPIC}
    if observed is not None:
        block['provenance'] = {
            'report_run': 'run-abc',
            'observed_members': list(observed),
            'detector': 'topic-cluster-scan',
            'authoritative': False,
        }
    if members_override:
        block.update(members_override)
    return {
        'execution_class': 'operational',
        'operational_mode': 'gate',
        GATE_METADATA_KEY: block,
    }


class FakeMemory:
    """The three store methods the CLI calls, and nothing else.

    ``get_memory_by_id`` returns a payload dict for a live id and ``None``
    for an absent one — ``MemoryService.get_memory_by_id``'s real contract,
    whose docstring makes the timeout-propagates-not-collapses rule explicit.
    """

    def __init__(self, members=_WELL_FORMED, *, live_ids=(), raises=None):
        self.members = list(members)
        self.live = {str(i).lower() for i in live_ids}
        self.raises = raises
        self.point_reads = []

    async def count_memories_by_metadata(self, project_id, filters):
        return len(self.members)

    async def get_memories_by_metadata(self, project_id, filters, limit=None):
        return list(self.members)

    async def get_memory_by_id(self, project_id, memory_id):
        self.point_reads.append((project_id, memory_id))
        if self.raises is not None:
            raise self.raises
        if str(memory_id).lower() in self.live:
            return {'id': memory_id, 'content': 'x', 'metadata': {}}
        return None


def _args(metadata, *, as_json=False):
    return argparse.Namespace(
        task_id=None,
        metadata_file=None,
        metadata_json=metadata,
        project_root=str(_ROOT),
        tag='master',
        project_id=_PROJECT_ID,
        as_json=as_json,
    )


def _run(metadata, memory, *, as_json=False):
    return asyncio.run(mod.run(_args(metadata, as_json=as_json), memory=memory))


class TestCliUnstampedClusterMember:
    def test_a_stray_is_reported_and_named(self, capsys):
        stray = _uuid(42)
        memory = FakeMemory(live_ids=[stray])
        rc = _run(_gate_blob([stray]), memory)
        out = capsys.readouterr().out
        assert rc == mod.EXIT_NOT_CLOSED == 1
        assert '[unstamped_cluster_member]' in out
        assert stray in out

    def test_json_mode_carries_the_reason_and_discloses_the_derivation(
        self, capsys
    ):
        import json  # noqa: PLC0415

        stray = _uuid(42)
        memory = FakeMemory(live_ids=[stray])
        rc = _run(_gate_blob([stray]), memory, as_json=True)
        payload = json.loads(capsys.readouterr().out)

        assert rc == mod.EXIT_NOT_CLOSED
        named = [
            r for r in payload['reasons']
            if r['code'] == 'unstamped_cluster_member'
        ]
        assert named and named[0]['ids'] == [stray]
        # The scroll facts must DISCLOSE what was derived, or an operator
        # cannot see what the check probed.
        assert payload['scroll']['unstamped'] == [stray]

    def test_a_stamped_observed_member_still_closes(self):
        """The measured corpus shape: zero derived candidates, zero reads."""
        memory = FakeMemory()
        rc = _run(_gate_blob([_uuid(1), _uuid(2)]), memory)
        assert rc == mod.EXIT_CLOSED == 0
        assert memory.point_reads == []

    def test_the_delete_arm_closes_and_is_never_probed(self):
        absorbed = _uuid(42)
        members = [
            _member(_uuid(1), canonical=True, supersedes=[absorbed]),
            _member(_uuid(2)),
        ]
        # Armed to report it LIVE, so the SUPPRESSION is what closes the gate.
        memory = FakeMemory(members, live_ids=[absorbed])
        rc = _run(_gate_blob([_uuid(1), absorbed]), memory)
        assert rc == mod.EXIT_CLOSED
        assert memory.point_reads == []

    def test_a_hard_deleted_unclaimed_id_closes_after_one_probe(self):
        gone = _uuid(42)
        memory = FakeMemory(live_ids=[])
        rc = _run(_gate_blob([gone]), memory)
        assert rc == mod.EXIT_CLOSED
        assert memory.point_reads == [(_PROJECT_ID, gone)]

    def test_the_probe_is_scoped_to_the_requested_project(self):
        """An argument-order slip would probe the wrong scope and silently
        report every candidate absent."""
        stray = _uuid(42)
        memory = FakeMemory(live_ids=[stray])
        _run(_gate_blob([stray]), memory)
        assert memory.point_reads == [(_PROJECT_ID, stray)]

    def test_a_probe_failure_is_EXIT_USAGE_not_EXIT_NOT_CLOSED(self, capsys):
        """The script's own EXIT-CODE CONTRACT: an unreachable store and a
        genuinely malformed cluster are different facts and must not share an
        exit code."""
        memory = FakeMemory(live_ids=[], raises=TimeoutError('qdrant point read'))
        rc = _run(_gate_blob([_uuid(42)]), memory)
        assert rc == mod.EXIT_USAGE == 2
        assert 'COULD NOT CHECK' in capsys.readouterr().out

    def test_a_block_less_blob_still_exits_usage(self, capsys):
        """The operator-facing half of the SECOND gap, unchanged by step-14."""
        rc = _run({'execution_class': 'operational', 'operational_mode': 'gate'}, FakeMemory())
        out = capsys.readouterr().out
        assert rc == mod.EXIT_USAGE
        assert GATE_METADATA_KEY in out


class TestCliAndSeamAgree:
    """Guards against the CLI and the seam disagreeing about the very cluster
    an operator is checking — the false reassurance the whole gate exists to
    prevent."""

    @pytest.mark.parametrize(
        'observed,live_ids,members',
        [
            ([_uuid(42)], [_uuid(42)], _WELL_FORMED),          # stray, live
            ([_uuid(42)], [], _WELL_FORMED),                    # stray, gone
            ([_uuid(1), _uuid(2)], [], _WELL_FORMED),           # all stamped
        ],
    )
    def test_same_inputs_same_reason_codes(self, observed, live_ids, members, capsys):
        import json  # noqa: PLC0415

        memory = FakeMemory(members, live_ids=live_ids)
        _run(_gate_blob(observed), memory, as_json=True)
        cli_codes = [
            r['code'] for r in json.loads(capsys.readouterr().out)['reasons']
        ]

        unstamped = [i for i in observed if str(i).lower() in {
            str(x).lower() for x in live_ids
        } and i not in {m['id'] for m in members}]
        verdict = evaluate_closure(
            _gate_blob(observed)[GATE_METADATA_KEY],
            members=members,
            scroll_total=len(members),
            scroll_truncated=False,
            scroll_available=True,
            unstamped_live_ids=unstamped,
        )
        assert cli_codes == [r['code'] for r in verdict.reasons]
