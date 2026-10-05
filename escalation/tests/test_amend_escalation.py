"""The explicit ``amend`` verb: framing appended to a pending record (task 4886).

PRD ``docs/prds/truth-propagation-record-mechanics.md`` leaf alpha.  A ruling
made elsewhere must be writable onto the record it rules on WITHOUT the side
effects of the only writer that existed before: a ``promote_to_l2`` fold,
which also applies a severity floor and feeds the over-fold signal.

The contract pinned here is what ``amend`` must NOT do as much as what it
does.  It appends one capped amendment and bumps ``updated_at`` (the watcher's
re-assess trigger), and every other field of the record, and every other file
in the queue, is left exactly as it was.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _pending_tool_fixtures import _file

from escalation.dedupe import DedupeConfig
from escalation.models import Amendment, Escalation
from escalation.queue import (
    _MAX_AMENDMENT_DETAIL_CHARS,
    _MAX_AMENDMENT_LINE_CHARS,
    _MAX_AMENDMENT_OPTIONS,
    _MAX_AMENDMENTS,
    AmendResult,
    EscalationQueue,
)
from escalation.server import (
    _AMENDMENT_TRUNCATION_ANCHOR_TASK_ID,
    _AMENDMENT_TRUNCATION_STORM_THRESHOLD,
    create_server,
)

# The only keys an amend may change on the record it amends.
_AMEND_WRITES = frozenset({'amendments', 'updated_at', 'amendments_chars_elided'})


def _record_path(queue: EscalationQueue, esc_id: str) -> Path:
    return queue.queue_dir / f'{esc_id}.json'


def _on_disk(queue: EscalationQueue, esc_id: str) -> Escalation:
    return Escalation.from_json(_record_path(queue, esc_id).read_text())


def _json_files(queue: EscalationQueue) -> dict[str, bytes]:
    """Every record file under the queue, archive included, by relative path."""
    return {
        str(p.relative_to(queue.queue_dir)): p.read_bytes()
        for p in sorted(queue.queue_dir.rglob('*.json'))
    }


def _others(snapshot: dict[str, bytes], esc_id: str) -> dict[str, bytes]:
    return {k: v for k, v in snapshot.items() if k != f'{esc_id}.json'}


def _framed_l2(queue: EscalationQueue, **kw: Any) -> Escalation:
    """A pending L2 carrying its own framing, plus an unrelated neighbour."""
    _file(queue, 'neighbour', level=1, summary='an unrelated record')
    return _file(
        queue, 'task-1', level=2,
        members=kw.pop('members', ['esc-m-1', 'esc-m-2']),
        severity='blocking',
        summary='ORIGINAL one-line hypothesis',
        detail='ORIGINAL detail text',
        root_cause='Bad merge strategy',
        options=['A: fix', 'B: rollback'],
        **kw,
    )


def _amendment(i: int) -> Amendment:
    return {
        'timestamp': f'2026-01-01T00:00:{i:02d}+00:00',
        'agent_role': 'seed',
        'root_cause': f'seeded root cause {i}',
        'summary': f'seeded summary {i}',
        'detail': f'seeded detail {i}',
        'options': [],
    }


class TestQueueAmend:
    def test_amend_appends_one_entry_and_touches_nothing_else(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        triaged = queue.stamp_triage(l2.id, triaged_by='watcher', triage_note='probe')
        assert triaged is not None and triaged.triaged_at is not None
        before = _on_disk(queue, l2.id).to_dict()
        others_before = _others(_json_files(queue), l2.id)

        called_at = datetime.now(UTC)
        result = queue.amend(
            l2.id,
            summary='ruled: the merge strategy is fine',
            detail='ruling made in task 4886, see the PRD',
            options=['A: close', 'B: keep open'],
            root_cause='Not a merge strategy problem',
            agent_role='interactive',
        )
        returned_at = datetime.now(UTC)

        assert result['status'] == 'amended'
        assert result['recorded'] is True
        assert result['dropped'] == 0
        stored = _on_disk(queue, l2.id)
        assert result['escalation'] is not None
        assert result['escalation'].to_dict() == stored.to_dict()

        [entry] = stored.amendments
        assert entry['agent_role'] == 'interactive'
        assert entry['summary'] == 'ruled: the merge strategy is fine'
        assert entry['detail'] == 'ruling made in task 4886, see the PRD'
        assert entry['options'] == ['A: close', 'B: keep open']
        assert entry['root_cause'] == 'Not a merge strategy problem'
        assert called_at <= datetime.fromisoformat(entry['timestamp']) <= returned_at, (
            'the queue stamps the amendment timestamp; a caller cannot supply one'
        )

        assert stored.updated_at is not None
        assert (
            datetime.fromisoformat(stored.updated_at)
            > datetime.fromisoformat(triaged.triaged_at)
        ), 'a recorded amendment is the re-assess trigger: updated_at > triaged_at'

        after = stored.to_dict()
        for key, value in before.items():
            if key in _AMEND_WRITES:
                continue
            assert after[key] == value, f'amend must not write {key!r}'
        assert after['root_cause_variants'] == before['root_cause_variants'], (
            'an explicit amend must not feed the over-fold signal'
        )
        assert after['root_cause_variants_truncated'] == before['root_cause_variants_truncated']

        assert _others(_json_files(queue), l2.id) == others_before

    def test_unknown_id_is_not_found_and_creates_nothing(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        _framed_l2(queue)
        snapshot = _json_files(queue)

        result = queue.amend('esc-nope-1', summary='anything', agent_role='interactive')

        assert result == {
            'status': 'not_found', 'escalation': None, 'recorded': False, 'dropped': 0,
        }
        assert not _record_path(queue, 'esc-nope-1').exists()
        assert _json_files(queue) == snapshot

    def test_archived_record_is_not_found_and_never_resurrected(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue, members=[])
        assert queue.resolve(l2.id, 'decided') is not None
        assert not _record_path(queue, l2.id).exists()
        snapshot = _json_files(queue)

        result = queue.amend(l2.id, summary='too late', agent_role='interactive')

        assert result['status'] == 'not_found'
        assert result['escalation'] is None
        assert result['recorded'] is False
        assert not _record_path(queue, l2.id).exists(), (
            'amending an archived record must not resurrect it into the queue root'
        )
        assert _json_files(queue) == snapshot

    def test_non_pending_record_in_queue_root_is_refused(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        closed = _on_disk(queue, l2.id)
        closed.status = 'resolved'
        closed.resolution = 'decided'
        closed.resolved_at = datetime.now(UTC).isoformat()
        _record_path(queue, l2.id).write_text(closed.to_json())
        snapshot = _json_files(queue)

        result = queue.amend(l2.id, summary='too late', agent_role='interactive')

        assert result['status'] == 'not_pending'
        assert result['escalation'] is not None
        assert result['escalation'].status == 'resolved'
        assert result['recorded'] is False
        assert result['dropped'] == 0
        assert _json_files(queue) == snapshot

    def test_repeat_framing_is_a_true_no_op(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        snapshot = _json_files(queue)

        echo = queue.amend(
            l2.id,
            root_cause=l2.root_cause, summary=l2.summary,
            detail=l2.detail, options=list(l2.options),
            agent_role='interactive',
        )

        assert echo['status'] == 'repeat_framing'
        assert echo['recorded'] is False
        assert echo['dropped'] == 0
        assert _on_disk(queue, l2.id).updated_at is None
        assert _on_disk(queue, l2.id).amendments == []
        assert _json_files(queue) == snapshot, (
            "re-sending the record's own framing must not rewrite anything"
        )

        ruling = 'a genuinely new ruling'
        assert queue.amend(
            l2.id, summary=ruling, agent_role='interactive',
        )['status'] == 'amended'
        after_real = _json_files(queue)
        again = queue.amend(l2.id, summary=ruling, agent_role='interactive')
        assert again['status'] == 'repeat_framing'
        assert again['recorded'] is False
        assert len(_on_disk(queue, l2.id).amendments) == 1
        assert _json_files(queue) == after_real

    def test_empty_framing_is_refused_without_a_write(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        snapshot = _json_files(queue)

        result = queue.amend(
            l2.id, summary='', detail='', options=[], root_cause='',
            agent_role='interactive',
        )

        assert result['status'] == 'no_framing'
        assert result['recorded'] is False
        assert result['dropped'] == 0
        assert _json_files(queue) == snapshot

    def test_parked_record_is_amendable(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l1 = _file(queue, 'task-2', level=1, summary='needs a human')
        parked = queue.park(l1.id, 'parked: waiting on the operator')
        assert parked is not None
        assert (parked.level, parked.resolution_action, parked.status) == (2, 'park', 'pending')
        others_before = _others(_json_files(queue), l1.id)

        result = queue.amend(
            l1.id, summary='operator ruled: proceed', agent_role='interactive',
        )

        assert result['status'] == 'amended'
        stored = _on_disk(queue, l1.id)
        assert [a['summary'] for a in stored.amendments] == ['operator ruled: proceed']
        assert (stored.level, stored.resolution_action, stored.status) == (2, 'park', 'pending')
        assert stored.resolution == 'parked: waiting on the operator'
        assert _others(_json_files(queue), l1.id) == others_before

    def test_entry_cap_sheds_the_oldest_on_this_path(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        seeded = [_amendment(i) for i in range(_MAX_AMENDMENTS)]
        l2 = _framed_l2(queue, amendments=seeded)
        others_before = _others(_json_files(queue), l2.id)

        result = queue.amend(l2.id, summary='one past the cap', agent_role='interactive')

        assert result['status'] == 'amended'
        assert result['recorded'] is True
        assert result['dropped'] == 1
        stored = _on_disk(queue, l2.id)
        assert len(stored.amendments) == _MAX_AMENDMENTS
        assert stored.amendments_truncated == 1
        assert stored.amendments[0] == seeded[1], 'the OLDEST entry is the one shed'
        assert stored.amendments[-1]['summary'] == 'one past the cap'
        assert _others(_json_files(queue), l2.id) == others_before

    def test_field_caps_bind_on_this_path(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        others_before = _others(_json_files(queue), l2.id)
        over = 50

        result = queue.amend(
            l2.id,
            root_cause='R' * (_MAX_AMENDMENT_LINE_CHARS + over),
            summary='S' * (_MAX_AMENDMENT_LINE_CHARS + over),
            detail='D' * (_MAX_AMENDMENT_DETAIL_CHARS + over),
            options=[f'option {i}' for i in range(_MAX_AMENDMENT_OPTIONS + 2)],
            agent_role='interactive',
        )

        assert result['status'] == 'amended'
        stored = _on_disk(queue, l2.id)
        [entry] = stored.amendments
        assert entry['root_cause'].count('R') == _MAX_AMENDMENT_LINE_CHARS
        assert entry['summary'].count('S') == _MAX_AMENDMENT_LINE_CHARS
        assert entry['detail'].count('D') == _MAX_AMENDMENT_DETAIL_CHARS
        assert entry['detail'].endswith(' ...]'), 'elision is marked in-band'
        assert str(over) in entry['detail']
        assert len(entry['options']) == _MAX_AMENDMENT_OPTIONS
        assert stored.amendments_chars_elided > 3 * over
        assert _others(_json_files(queue), l2.id) == others_before


async def _amend(server, **kw: Any) -> dict[str, Any]:
    """``amend_escalation`` is an ASYNC tool: it awaits the storm reporter."""
    tool = await server.get_tool('amend_escalation')
    return await tool.fn(**kw)


async def _sync_tool(server, name: str, **kw: Any) -> dict[str, Any]:
    tool = await server.get_tool(name)
    return tool.fn(**kw)


class _AmendThreadProbeQueue(EscalationQueue):
    """A REAL queue recording which thread ``amend`` ran on, then delegating.

    The ``test_write_path_scans_off_loop.py::_ThreadProbeQueue`` shape: not a
    mock, so the probed call still does the real flock, read and rewrite.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.amend_threads: list[int] = []

    def amend(self, escalation_id: str, **kw: Any) -> AmendResult:
        self.amend_threads.append(threading.get_ident())
        return super().amend(escalation_id, **kw)


@pytest.mark.asyncio
class TestAmendEscalationTool:
    async def test_amend_is_visible_and_changes_no_state_field(self, tmp_path):
        """PRD alpha acceptance: the ruling lands on the record, nothing else moves."""
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        server = create_server(queue, startup_sweep=False)
        triaged = await _sync_tool(
            server, 'stamp_triage', escalation_id=l2.id, triage_note='probe',
        )
        assert 'error' not in triaged
        before = await _sync_tool(server, 'get_escalation', escalation_id=l2.id)

        response = await _amend(
            server, escalation_id=l2.id,
            summary='ruled: proceed', detail='ruling evidence',
            options=['A: proceed'], root_cause='ruled elsewhere',
            agent_role='interactive',
        )

        assert response['amendment_recorded'] is True
        assert 'no_op_reason' not in response
        after = await _sync_tool(server, 'get_escalation', escalation_id=l2.id)
        assert {k: v for k, v in response.items() if k != 'amendment_recorded'} == after, (
            'the success response is the full amended record'
        )
        [entry] = after['amendments']
        assert entry['summary'] == 'ruled: proceed'
        assert entry['detail'] == 'ruling evidence'
        assert entry['agent_role'] == 'interactive'
        assert datetime.fromisoformat(entry['timestamp'])
        assert (
            datetime.fromisoformat(after['updated_at'])
            > datetime.fromisoformat(after['triaged_at'])
        )
        for key in ('status', 'severity', 'level', 'members'):
            assert after[key] == before[key], f'amend_escalation must not move {key!r}'

    async def test_unknown_id_is_not_found(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(queue, startup_sweep=False)

        response = await _amend(
            server, escalation_id='esc-nope-1', summary='x', agent_role='interactive',
        )

        assert response['code'] == 'not_found'
        assert response['error']

    async def test_archived_record_is_not_pending_and_untouched(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue, members=[])
        assert queue.resolve(l2.id, 'decided') is not None
        server = create_server(queue, startup_sweep=False)
        snapshot = _json_files(queue)

        response = await _amend(
            server, escalation_id=l2.id, summary='too late', agent_role='interactive',
        )

        assert response['code'] == 'not_pending'
        assert response['status'] == 'resolved'
        assert response['error']
        assert _json_files(queue) == snapshot, 'the archived file is byte-identical'
        assert not _record_path(queue, l2.id).exists(), 'nothing re-created in the root'

    async def test_empty_amendment_is_refused(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        server = create_server(queue, startup_sweep=False)
        snapshot = _json_files(queue)

        response = await _amend(
            server, escalation_id=l2.id, summary='', detail='', options=[],
            root_cause='', agent_role='interactive',
        )

        assert response['code'] == 'empty_amendment'
        assert response['error']
        assert _json_files(queue) == snapshot

    async def test_repeat_framing_is_a_success_that_records_nothing(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        server = create_server(queue, startup_sweep=False)
        ruling = {'escalation_id': l2.id, 'summary': 'ruled', 'agent_role': 'interactive'}
        first = await _amend(server, **ruling)
        assert first['amendment_recorded'] is True

        response = await _amend(server, **ruling)

        assert 'error' not in response
        assert response['amendment_recorded'] is False
        assert response['no_op_reason'] == 'repeat_framing'
        assert response['updated_at'] == first['updated_at']
        assert len(response['amendments']) == 1

    async def test_queue_write_runs_off_the_event_loop(self, tmp_path):
        queue = _AmendThreadProbeQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue)
        server = create_server(queue, startup_sweep=False)

        response = await _amend(
            server, escalation_id=l2.id, summary='ruled', agent_role='interactive',
        )

        assert response['amendment_recorded'] is True
        assert queue.amend_threads, 'the amend probe recorded no call at all'
        assert threading.get_ident() not in queue.amend_threads, (
            'queue.amend ran ON the event-loop thread'
        )


@pytest.mark.asyncio
class TestAmendTruncationStorm:
    """Amend-driven truncation feeds the SAME INV-4 escape as a fold's.

    An alarm whose census excluded a truncation source would under-report
    exactly the cap pressure it exists to detect.  Dedupe is off for the
    reason ``test_server.py::TestAmendmentTruncationStorm`` states: otherwise
    a second record would fold and "exactly one" would pass for the wrong
    reason.
    """

    async def test_amend_truncations_file_one_info_escalation(self, tmp_path):
        queue = EscalationQueue(tmp_path / 'esc')
        l2 = _framed_l2(queue, amendments=[_amendment(i) for i in range(_MAX_AMENDMENTS)])
        server = create_server(
            queue, startup_sweep=False,
            dedupe_config=DedupeConfig(infra_dedupe_enabled=False),
        )
        known = {e.id for e in queue.get_pending()}

        def new_pending() -> set[str]:
            return {e.id for e in queue.get_pending()} - known

        for i in range(_AMENDMENT_TRUNCATION_STORM_THRESHOLD - 1):
            below = await _amend(
                server, escalation_id=l2.id, summary=f'ruling {i}', agent_role='interactive',
            )
            assert below['amendment_recorded'] is True
            assert below['amendments_truncated'] == i + 1
        assert new_pending() == set(), 'below the threshold nothing may be filed'

        await _amend(
            server, escalation_id=l2.id, summary='ruling at threshold',
            agent_role='interactive',
        )

        [storm_id] = new_pending()
        storm = queue.get(storm_id)
        assert storm is not None
        assert storm.severity == 'info'
        assert storm.category == 'infra_issue'
        assert storm.task_id == _AMENDMENT_TRUNCATION_ANCHOR_TASK_ID
        assert l2.id in storm.summary
