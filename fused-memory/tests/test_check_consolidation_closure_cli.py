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
from unittest.mock import AsyncMock

import pytest

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.consolidation_gate import (
    GATE_METADATA_KEY,
    closure_exists_probe,
)
from fused_memory.reconciliation.event_buffer import EventBuffer

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
    """The CLI's verdict and the REAL seam's verdict, over one cluster.

    The comparison is deliberately CLI-output versus
    ``TaskInterceptor.set_task_status(..., 'done')`` — the actual chokepoint
    an operator is predicting — and NOT versus a local
    ``evaluate_closure(unstamped_live_ids=<list comprehension>)`` call. A
    hand-rolled reference derivation can only agree with production by
    accident: the earlier version of this test omitted the canonical's
    ``supersedes`` subtraction entirely (the subtlest rule in
    ``unstamped_candidates``, and the one keeping every delete-arm
    consolidation closeable) and stayed green only because no case exercised
    it. Both sides now run the same production code, and the fourth case
    below exercises exactly that rule.

    The two sides are wired to ONE ``FakeMemory``: the CLI reaches it through
    ``scroll_cluster``, the seam through the three injected collaborators
    ``server/main.py::_wire_closure_collaborators`` binds in production.
    """

    @staticmethod
    async def _seam_codes(metadata, memory, tmp_path):
        """Drive the real chokepoint and return its reason codes."""
        buf = EventBuffer(db_path=tmp_path / 'cli_seam_eb.db', buffer_size_threshold=100)
        await buf.initialize()
        try:
            taskmaster = AsyncMock()
            taskmaster.get_task = AsyncMock(
                return_value={'id': '9001', 'status': 'pending', 'metadata': metadata}
            )
            taskmaster.set_task_status = AsyncMock(return_value={'success': True})
            taskmaster.set_status_and_stamp_audit = AsyncMock(
                return_value={'success': True}
            )
            interceptor = TaskInterceptor(taskmaster, AsyncMock(), buf)

            async def scroll(filters, *, limit, project_id):
                return await memory.get_memories_by_metadata(
                    project_id, filters, limit=limit
                )

            async def count(filters, *, project_id):
                return await memory.count_memories_by_metadata(project_id, filters)

            interceptor.set_consolidation_scroll(
                scroll, count=count, exists=closure_exists_probe(memory)
            )
            result = await interceptor.set_task_status(
                '9001', 'done', project_root=str(_ROOT)
            )
            return [r['code'] for r in (result.get('reasons') or [])]
        finally:
            await buf.close()

    @pytest.mark.parametrize(
        'observed,live_ids,members',
        [
            ([_uuid(42)], [_uuid(42)], _WELL_FORMED),          # stray, live
            ([_uuid(42)], [], _WELL_FORMED),                    # stray, gone
            ([_uuid(1), _uuid(2)], [], _WELL_FORMED),           # all stamped
            # THE DELETE ARM: the canonical claims it absorbed _uuid(42), and
            # the store would report that id LIVE if either side probed it. Both
            # must subtract the claim and close clean — if one side forgets, the
            # CLI reassures an operator the seam is about to refuse (or the
            # reverse, which makes a correct consolidation look uncloseable).
            (
                [_uuid(1), _uuid(42)],
                [_uuid(42)],
                [
                    _member(_uuid(1), canonical=True, supersedes=[_uuid(42)]),
                    _member(_uuid(2)),
                ],
            ),
        ],
    )
    def test_same_inputs_same_reason_codes(
        self, observed, live_ids, members, tmp_path, capsys
    ):
        import json  # noqa: PLC0415

        metadata = _gate_blob(observed)

        cli_memory = FakeMemory(members, live_ids=live_ids)
        _run(metadata, cli_memory, as_json=True)
        cli_codes = [
            r['code'] for r in json.loads(capsys.readouterr().out)['reasons']
        ]

        seam_memory = FakeMemory(members, live_ids=live_ids)
        seam_codes = asyncio.run(
            self._seam_codes(metadata, seam_memory, tmp_path)
        )

        assert cli_codes == seam_codes
        # Same verdict AND same store traffic: a side that reached a matching
        # verdict while probing a different set of ids has a different
        # derivation and would diverge on the next case.
        assert [c[1] for c in cli_memory.point_reads] == [
            c[1] for c in seam_memory.point_reads
        ]

    def test_the_delete_arm_case_really_would_have_probed(self, tmp_path):
        """Guards the guard: the ``supersedes`` case above is only meaningful
        because the claimed id WOULD otherwise be a probe candidate. Drop the
        claim and both sides go looking for it."""
        unclaimed = [_member(_uuid(1), canonical=True), _member(_uuid(2))]
        memory = FakeMemory(unclaimed, live_ids=[_uuid(42)])
        codes = asyncio.run(
            self._seam_codes(_gate_blob([_uuid(1), _uuid(42)]), memory, tmp_path)
        )
        assert 'unstamped_cluster_member' in codes
        assert [c[1] for c in memory.point_reads] == [_uuid(42)]
