"""``get_pending_escalations``' computed ``pins_recovery`` annotation (task 3543).

Spec S8 / PRD task iota.  Every datum the join needs already exists — the
pending record, the task's status, the workflow's liveness, and
``escalation.pins.classify_pins``' severity-aware verdict — but nothing ever
put them together, so seven tripwire L0s that share one cause read as seven
unrelated strands.  This module pins the annotation.

The annotation answers exactly one question per record: *is this escalation
what is stopping that task from being recovered?*  It is therefore the
conjunction of four things, and dropping any one of them makes it a lie:

    the task is in-progress or blocked   (there is something to recover)
  ∧ no live claimant                     (it is actually stranded, not running)
  ∧ the record is a live queue handoff   (classify_pins, not bool(open))
  ∧ the status read succeeded            (else the key is OMITTED, never [])

The omission contract is the same fail-safe that ``PinReport``'s
``store_unavailable`` third state encodes: a false ``[]`` reads as "nothing
pins this task", which is precisely the collapse (esc-3163) that routes a
genuinely-pinned strand down the wrong branch.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

IN_PROGRESS = 'in-progress'


def _file(queue: EscalationQueue, task_id: str, **kw: Any) -> Escalation:
    """Submit one pending escalation and return it."""
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=kw.pop('agent_role', 'implementer'),
        severity=kw.pop('severity', 'blocking'),
        category=kw.pop('category', 'scope_violation'),
        summary=kw.pop('summary', 'pins-recovery fixture'),
        level=kw.pop('level', 1),
        **kw,
    )
    queue.submit(esc)
    return esc


class _Scheduler:
    """Minimal stand-in for orchestrator.scheduler's status accessors.

    ``get_statuses`` returns a ``(statuses, error)`` TUPLE — the real shape at
    orchestrator/src/orchestrator/scheduler.py:2523, and the reason a caller
    that assumes a bare dict silently treats an error as "no tasks".
    """

    def __init__(self, statuses: dict[str, str], error: Exception | None = None):
        self._statuses = statuses
        self._error = error
        self.calls: list[Any] = []

    async def get_statuses(self, ids: list[str] | None = None):
        self.calls.append(ids)
        if self._error is not None:
            return {}, self._error
        if ids is None:
            return dict(self._statuses), None
        return {i: self._statuses[i] for i in ids if i in self._statuses}, None


def _harness(statuses: dict[str, str], *, live: set[str] | None = None, **kw: Any):
    live_ids = live or set()
    scheduler = _Scheduler(statuses, error=kw.pop('error', None))
    return types.SimpleNamespace(
        scheduler=scheduler,
        is_workflow_active=lambda tid: tid in live_ids,
        **kw,
    )


async def _get_pending(server, **kwargs: Any) -> list[dict[str, Any]]:
    """get_pending_escalations is an ASYNC def as of task 3543 — it awaits a
    batched scheduler status read to compute pins_recovery."""
    tool = await server.get_tool('get_pending_escalations')
    return await tool.fn(**kwargs)


@pytest.mark.asyncio
class TestPinsRecoveryJoin:
    async def test_pending_l1_on_stranded_in_progress_task_pins(self, tmp_path):
        """The canonical case: an open L1 on an in-progress task nobody is running."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '900', level=1)
        server = create_server(queue, harness=_harness({'900': IN_PROGRESS}))

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == ['900']

    async def test_blocked_task_with_open_l2_pins(self, tmp_path):
        """A blocked task is recoverable too — it must not be excluded."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '901', level=2, severity='critical')
        server = create_server(queue, harness=_harness({'901': 'blocked'}))

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == ['901']

    @pytest.mark.parametrize('status', ['pending', 'done', 'cancelled', 'deferred'])
    async def test_non_recoverable_task_status_does_not_pin(self, tmp_path, status):
        """There is nothing to recover, so the record pins nothing."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '902', level=1)
        server = create_server(queue, harness=_harness({'902': status}))

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == []

    async def test_live_workflow_does_not_pin(self, tmp_path):
        """A task with a live claimant is running, not stranded — nothing to unpin.

        The record is still a genuine queue handoff (classify_pins says so);
        what is missing is a recovery for it to block.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '903', level=1)
        server = create_server(
            queue, harness=_harness({'903': IN_PROGRESS}, live={'903'}),
        )

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == []

    async def test_info_severity_does_not_pin(self, tmp_path):
        """classify_pins link 1: an info record never pins, at any level.

        Derived from the classifier's buckets — a locally re-rolled
        ``bool(open_escalations)`` would pin here and be wrong.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '904', level=1, severity='info')
        server = create_server(queue, harness=_harness({'904': IN_PROGRESS}))

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == []

    async def test_dead_l0_does_not_pin(self, tmp_path):
        """classify_pins link 4: an L0 whose filing incarnation is gone.

        Its handoff has no consumer left, so conversion proceeds and the
        record does NOT pin recovery — even though it still vetoes a done-flip.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '905', level=0, filing_claimant_run_id='r1/s1/pid=42')
        server = create_server(queue, harness=_harness({'905': IN_PROGRESS}))

        [rec] = await _get_pending(server)
        assert rec['pins_recovery'] == []


@pytest.mark.asyncio
class TestPinsRecoveryBatching:
    async def test_status_read_is_batched_across_records(self, tmp_path):
        """One get_statuses call for N records over M tasks — not one per record."""
        queue = EscalationQueue(tmp_path / 'esc')
        for tid in ('910', '911', '912'):
            _file(queue, tid, level=1)
            _file(queue, tid, level=2, severity='critical')
        harness = _harness({t: IN_PROGRESS for t in ('910', '911', '912')})
        server = create_server(queue, harness=harness)

        recs = await _get_pending(server)

        assert len(recs) == 6
        assert len(harness.scheduler.calls) == 1, (
            f'expected one batched status read, got {harness.scheduler.calls}'
        )
        assert sorted(harness.scheduler.calls[0]) == ['910', '911', '912']

    async def test_several_records_on_one_task_annotate_consistently(self, tmp_path):
        """Two open handoffs on one task both name it — they share the cause."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '913', level=1)
        _file(queue, '913', level=2, severity='urgent')
        server = create_server(queue, harness=_harness({'913': IN_PROGRESS}))

        recs = await _get_pending(server)
        assert [r['pins_recovery'] for r in recs] == [['913'], ['913']]

    async def test_level_filter_classifies_against_the_full_open_set(self, tmp_path):
        """A filtered VIEW must not become a filtered CLASSIFICATION.

        Asking for level=0 only must still judge the L0 against the task's
        whole open set; classifying the visible subset would answer a question
        about a store state that never existed.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        l0 = _file(queue, '914', level=0, filing_claimant_run_id='r1/s1/pid=7')
        _file(queue, '914', level=1)
        server = create_server(queue, harness=_harness({'914': IN_PROGRESS}))

        recs = await _get_pending(server, level=0)
        assert [r['id'] for r in recs] == [l0.id]
        # The L0's own filer is gone => dead_l0 => it does not pin, even though
        # its sibling L1 does.  The annotation is per-record, not per-task.
        assert recs[0]['pins_recovery'] == []


@pytest.mark.asyncio
class TestPinsRecoveryOmission:
    """Unknown must stay distinguishable from "nothing pins this"."""

    async def test_key_omitted_when_no_harness(self, tmp_path):
        """A standalone escalation server cannot resolve status or liveness."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '920', level=1)
        server = create_server(queue)

        [rec] = await _get_pending(server)
        assert 'pins_recovery' not in rec

    async def test_key_omitted_when_scheduler_absent(self, tmp_path):
        """A harness without a scheduler is the same unknown."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '921', level=1)
        harness = types.SimpleNamespace(is_workflow_active=lambda tid: False)
        server = create_server(queue, harness=harness)

        [rec] = await _get_pending(server)
        assert 'pins_recovery' not in rec

    async def test_key_omitted_when_status_read_reports_an_error(self, tmp_path):
        """get_statuses' error element must be read, not discarded.

        Its ``({}, exc)`` failure shape is indistinguishable from "no tasks"
        if the caller unpacks only the dict — and an empty status map would
        make every record report ``[]``, i.e. "nothing pins this".
        """
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '922', level=1)
        server = create_server(
            queue, harness=_harness({}, error=RuntimeError('fused-memory down')),
        )

        [rec] = await _get_pending(server)
        assert 'pins_recovery' not in rec

    async def test_key_omitted_when_status_read_raises(self, tmp_path):
        """A raising status read must never fail the tool itself."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '923', level=1)

        class _Boom:
            async def get_statuses(self, ids=None):
                raise RuntimeError('boom')

        harness = types.SimpleNamespace(
            scheduler=_Boom(), is_workflow_active=lambda tid: False,
        )
        server = create_server(queue, harness=harness)

        [rec] = await _get_pending(server)
        assert rec['id'].startswith('esc-923')
        assert 'pins_recovery' not in rec

    async def test_key_omitted_for_a_task_missing_from_the_status_map(self, tmp_path):
        """An unresolved task is unknown, not "not pinning".

        The read succeeded for its siblings, so the failure is scoped to the
        one record rather than collapsing the whole call.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '924', level=1)
        _file(queue, '925', level=1)
        server = create_server(queue, harness=_harness({'924': IN_PROGRESS}))

        by_task = {r['task_id']: r for r in await _get_pending(server)}
        assert by_task['924']['pins_recovery'] == ['924']
        assert 'pins_recovery' not in by_task['925']


@pytest.mark.asyncio
class TestPinsRecoveryCompactProjection:
    async def test_compact_carries_the_annotation(self, tmp_path):
        """The dashboard reads compact records — dropping the key here would
        make the whole PINNING surface read as "nothing pins"."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '930', level=1, detail='x' * 4000)
        server = create_server(queue, harness=_harness({'930': IN_PROGRESS}))

        [rec] = await _get_pending(server, compact=True)
        assert rec['pins_recovery'] == ['930']
        assert 'detail' not in rec  # compact is still compact

    async def test_compact_projects_when_the_key_is_absent(self, tmp_path):
        """Projection must not KeyError on the omitted-annotation shape."""
        queue = EscalationQueue(tmp_path / 'esc')
        _file(queue, '931', level=1)
        server = create_server(queue)

        [rec] = await _get_pending(server, compact=True)
        assert 'pins_recovery' not in rec
        assert rec['id'].startswith('esc-931')
        assert rec['task_id'] == '931'
