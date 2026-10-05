"""Tests for scripts/sitting/ownership.py — the amendment-A ownership and in-flight sweep (task 5376)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from orchestrator.session_registry import SessionRecord, Status
from sitting import ownership as mod
from sitting.inventory import OpenItem, escalation_key

NOW = datetime(2026, 9, 26, 5, 30, tzinfo=UTC)
QUEUE = '/src/dark-factory/data/escalations'


def _item(esc_id: str = 'esc-4797-15', task_id: str | None = '4797', **fields) -> OpenItem:
    fields.setdefault('filed_at', '2026-09-20T00:00:00+00:00')
    return OpenItem(
        key=escalation_key(QUEUE, esc_id), escalation_id=esc_id, queue_dir=QUEUE,
        project='dark_factory', task_id=task_id, **fields,
    )


def _session(root: Path, slug: str, *, status: Status, result: str | None = None, **fields) -> Path:
    record_dir = root / slug
    record_dir.mkdir(parents=True)
    fields.setdefault('start_ts', '2026-09-21T00:00:00+00:00')
    if result is not None:
        (record_dir / 'result.md').write_text(result)
        fields['result_file'] = str(record_dir / 'result.md')
    record = SessionRecord(session_slug=slug, status=status, **fields)
    (record_dir / 'record.json').write_text(record.to_json())
    return record_dir


@pytest.fixture
def sessions_root(tmp_path):
    root = tmp_path / 'fleet' / 'sessions'
    root.mkdir(parents=True)
    return root


@pytest.fixture
def project_root(tmp_path):
    root = tmp_path / 'src' / 'dark-factory'
    (root / '.taskmaster' / 'tasks').mkdir(parents=True)
    return root


@pytest.fixture
def seed_tasks(project_root, make_tasks_db):
    def _seed(rows):
        make_tasks_db(rows, directory=project_root / '.taskmaster' / 'tasks')
        return mod.load_task_rows(project_root)
    return _seed


def _probe(finding: mod.OwnershipFinding, name: str) -> mod.ProbeResult:
    return next(result for result in finding.probes if result.probe == name)


def _sweep(item, *, sessions_root, task_rows=None, handover_path=None):
    return mod.sweep(
        item,
        sessions=mod.index_sessions(sessions_root),
        task_rows=task_rows if task_rows is not None else mod.TaskRows(rows={}),
        handover_path=handover_path,
        now=NOW,
    )


class TestIndexSessions:
    def test_indexes_by_escalation_and_by_canonical_task(self, sessions_root):
        _session(sessions_root, 'unblock-df-4797-1', status=Status.RUNNING, role='unblock', project='df',
                 task_id='4797', escalation_id='esc-4797-15')
        _session(sessions_root, 'session-x-2', status=Status.EXITED, role='session')
        (sessions_root / 'broken-3').mkdir()
        (sessions_root / 'broken-3' / 'record.json').write_text('{"session_slug": "broken-3", "sta')

        index = mod.index_sessions(sessions_root)

        assert index.available
        assert [s.slug for s in index.by_escalation['esc-4797-15']] == ['unblock-df-4797-1']
        assert [s.slug for s in index.by_task[('dark_factory', '4797')]] == ['unblock-df-4797-1']
        assert [Path(s.path).parent.name for s in index.shortfalls] == ['broken-3']

    def test_missing_root_is_unavailable_not_empty(self, tmp_path):
        index = mod.index_sessions(tmp_path / 'nowhere')

        assert not index.available


class TestParseResultOutcome:
    @pytest.mark.parametrize(('text', 'expected'), [
        ('outcome: done\nchanged: x\n', 'done'),
        ('---\noutcome: handed-off\nchanged: task/5140\naction_needed: poll\n---\nprose', 'handed-off'),
        ('outcome: blocked', 'blocked'),
        ('outcome: abandoned\n', 'abandoned'),
        ('outcome: maybe\n', None),
        ('', None),
        ('I finished the work.\n', None),
        ('---\nchanged: x\n---\noutcome: done\n', None),
    ])
    def test_header_outcome_or_none(self, text, expected):
        assert mod.parse_result_outcome(text) == expected


class TestSpawnedSessionProbe:
    def test_in_flight_session_owns(self, sessions_root):
        _session(sessions_root, 'unblock-df-4797-7', status=Status.RUNNING, role='unblock', project='dark_factory',
                 task_id='4797', escalation_id='esc-4797-15')

        result = _probe(_sweep(_item(), sessions_root=sessions_root), 'spawned_session')

        assert result.status == 'owns'
        assert result.owner == 'session unblock-df-4797-7 (in flight)'

    def test_finished_done_session_owns(self, sessions_root):
        _session(sessions_root, 'unblock-df-4797-8', status=Status.EXITED, result='outcome: done\nchanged: x\n',
                 project='dark_factory', escalation_id='esc-4797-15')

        result = _probe(_sweep(_item(), sessions_root=sessions_root), 'spawned_session')

        assert result.status == 'owns'
        assert result.owner == 'session unblock-df-4797-8 finished: done'

    @pytest.mark.parametrize('result_text', [None, '', 'no header here\n', 'outcome: abandoned\n'])
    def test_finished_without_an_owning_outcome_is_only_a_mention(self, sessions_root, result_text):
        _session(sessions_root, 'unblock-df-4797-9', status=Status.EXITED, result=result_text,
                 project='dark_factory', escalation_id='esc-4797-15')

        result = _probe(_sweep(_item(), sessions_root=sessions_root), 'spawned_session')

        assert result.status == 'mentions'
        assert 'unblock-df-4797-9' in result.evidence

    def test_same_id_in_another_project_does_not_match(self, sessions_root):
        _session(sessions_root, 'unblock-reify-4797-1', status=Status.RUNNING, project='reify',
                 escalation_id='esc-4797-15')

        assert _probe(_sweep(_item(), sessions_root=sessions_root), 'spawned_session').status == 'empty'


class TestUnblockRunProbe:
    def test_unblock_under_another_spelling_that_finished_done_owns(self, sessions_root):
        _session(sessions_root, 'unblock-df-4797-3', status=Status.EXITED, result='outcome: done\n',
                 role='unblock', project='df', task_id='4797', start_ts='2026-09-22T00:00:00+00:00')

        result = _probe(_sweep(_item(), sessions_root=sessions_root), 'unblock_run')

        assert result.status == 'owns'
        assert result.owner == 'session unblock-df-4797-3 finished: done'

    def test_unblock_that_started_before_the_item_was_filed_is_ignored(self, sessions_root):
        _session(sessions_root, 'unblock-df-4797-2', status=Status.EXITED, result='outcome: done\n',
                 role='unblock', project='df', task_id='4797', start_ts='2026-09-01T00:00:00+00:00')

        assert _probe(_sweep(_item(), sessions_root=sessions_root), 'unblock_run').status == 'empty'

    def test_non_unblock_roles_do_not_count(self, sessions_root):
        _session(sessions_root, 'implementer-df-4797-1', status=Status.RUNNING, role='implementer',
                 project='dark_factory', task_id='4797')

        assert _probe(_sweep(_item(), sessions_root=sessions_root), 'unblock_run').status == 'empty'


class TestCoalesceFoldProbe:
    def test_fold_target_owns_with_its_live_status_and_title(self, sessions_root, seed_tasks):
        rows = seed_tasks([
            {'id': 4803, 'status': 'deferred', 'metadata': {'x_coalesced_into': 5255}},
            {'id': 5255, 'status': 'pending', 'title': 'carrier for the fold'},
        ])

        finding = _sweep(_item('esc-4803-2', '4803'), sessions_root=sessions_root, task_rows=rows)
        result = _probe(finding, 'coalesce_fold')

        assert result.status == 'owns'
        assert 'task 5255' in result.owner
        assert 'carrier for the fold' in result.owner
        assert result.owner_task_id == '5255'
        assert result.owner_task_status == 'pending'


class TestTaskRulingProbe:
    @pytest.mark.parametrize(('key', 'value', 'quoted'), [
        ('x_ruling', 'esc-5580-4 / Leo 2026-09-21: action D', 'esc-5580-4 / Leo 2026-09-21: action D'),
        ('x_ruling', {'ruled_by': 'leo', 'option': 'B', 'escalation_id': 'esc-3353-4'}, '"option": "B"'),
        ('x_operator_ruling', {'ruled_by': 'leo', 'disposition': 'RE-AFFIRM'}, '"disposition": "RE-AFFIRM"'),
        ('x_ruled_by', 'leo', 'leo'),
    ])
    def test_every_measured_ruling_shape_owns_and_is_quoted(self, sessions_root, seed_tasks, key, value, quoted):
        rows = seed_tasks([{'id': 3353, 'status': 'pending', 'metadata': {key: value}}])

        result = _probe(_sweep(_item('esc-3353-9', '3353'), sessions_root=sessions_root, task_rows=rows),
                        'task_ruling')

        assert result.status == 'owns'
        assert key in result.evidence
        assert quoted in result.evidence
        assert result.owner_task_id == '3353'
        assert result.owner_task_status == 'pending'


FOLLOWUP_SPELLINGS = pytest.mark.parametrize('key', ['x_origin_escalation', 'origin_escalation'])


class TestSpawnedFollowupProbe:
    @FOLLOWUP_SPELLINGS
    def test_live_followup_owns(self, sessions_root, seed_tasks, key):
        rows = seed_tasks([
            {'id': 6798, 'status': 'blocked'},
            {'id': 7001, 'status': 'pending', 'title': 'follow-up', 'metadata': {key: 'esc-6798-1'}},
        ])

        result = _probe(_sweep(_item('esc-6798-1', '6798'), sessions_root=sessions_root, task_rows=rows),
                        'spawned_followup')

        assert result.status == 'owns'
        assert result.owner_task_id == '7001'
        assert result.owner_task_status == 'pending'

    @FOLLOWUP_SPELLINGS
    @pytest.mark.parametrize('status', ['done', 'cancelled'])
    def test_terminal_followup_is_empty(self, sessions_root, seed_tasks, status, key):
        rows = seed_tasks([{'id': 7001, 'status': status, 'metadata': {key: 'esc-6798-1'}}])

        result = _probe(_sweep(_item('esc-6798-1', '6798'), sessions_root=sessions_root, task_rows=rows),
                        'spawned_followup')

        assert result.status == 'empty'


HANDOVER = """# L2 escalation-watcher — handover

## 1. Pending L2 escalations

### With a human already on it

**`esc-4797-15`** (blocking) — Leo has an `/unblock` terminal open.
Real rebase conflict onto a frozen tip.

### Untriaged — start here

**`esc-5756-3`** (blocking) — not investigated; see task 5756.
"""


class TestHandoverProbe:
    def test_prefers_the_data_path_then_the_newest_plans_file(self, tmp_path):
        plans = tmp_path / 'plans'
        plans.mkdir()
        (plans / 'l2-watcher-handover-2026-09-01.md').write_text('old')
        (plans / 'l2-watcher-handover-2026-09-22.md').write_text('new')

        assert mod.resolve_handover_path(tmp_path) == plans / 'l2-watcher-handover-2026-09-22.md'

        data = tmp_path / 'data' / 'escalations' / 'l2-handover.md'
        data.parent.mkdir(parents=True)
        data.write_text('current')

        assert mod.resolve_handover_path(tmp_path) == data

    def test_no_handover_resolves_to_none(self, tmp_path):
        assert mod.resolve_handover_path(tmp_path) is None

    def test_a_mentioning_paragraph_is_evidence_never_ownership(self, tmp_path, sessions_root):
        handover = tmp_path / 'handover.md'
        handover.write_text(HANDOVER)

        result = _probe(_sweep(_item(), sessions_root=sessions_root, handover_path=handover), 'handover')

        assert result.status == 'mentions'
        assert result.evidence == (
            '### With a human already on it\n'
            '**`esc-4797-15`** (blocking) — Leo has an `/unblock` terminal open.\n'
            'Real rebase conflict onto a frozen tip.'
        )

    def test_a_task_citation_also_mentions(self, tmp_path, sessions_root):
        handover = tmp_path / 'handover.md'
        handover.write_text(HANDOVER)

        result = _probe(_sweep(_item('esc-5756-9', '5756'), sessions_root=sessions_root, handover_path=handover),
                        'handover')

        assert result.status == 'mentions'
        assert result.evidence.startswith('### Untriaged — start here\n')

    def test_unmentioned_is_empty_and_missing_is_unavailable(self, tmp_path, sessions_root):
        handover = tmp_path / 'handover.md'
        handover.write_text(HANDOVER)

        assert _probe(_sweep(_item('esc-1-1', '1'), sessions_root=sessions_root, handover_path=handover),
                      'handover').status == 'empty'
        assert _probe(_sweep(_item(), sessions_root=sessions_root), 'handover').status == 'unavailable'


class TestSweep:
    def test_finding_carries_every_probe_in_order(self, sessions_root):
        finding = _sweep(_item(), sessions_root=sessions_root)

        assert [r.probe for r in finding.probes] == list(mod.PROBES)
        assert all(r.measured_at == NOW.isoformat() for r in finding.probes)
        assert finding.item_key == _item().key

    def test_owned_iff_any_probe_owns(self, sessions_root, seed_tasks):
        rows = seed_tasks([{'id': 4797, 'status': 'blocked', 'metadata': {'x_ruled_by': 'leo'}}])

        owned = _sweep(_item(), sessions_root=sessions_root, task_rows=rows)
        unowned = _sweep(_item('esc-1-1', '1'), sessions_root=sessions_root, task_rows=rows)

        assert owned.owned
        assert [r.probe for r in owned.owners] == ['task_ruling']
        assert not unowned.owned
        assert unowned.owners == ()

    def test_empty_probes_exclude_unavailable_ones(self, tmp_path):
        finding = mod.sweep(
            _item(),
            sessions=mod.index_sessions(tmp_path / 'no-sessions'),
            task_rows=mod.load_task_rows(tmp_path / 'no-project'),
            handover_path=None,
            now=NOW,
        )

        assert not finding.owned
        assert finding.empty_probes == ()
        assert {r.status for r in finding.probes} == {'unavailable'}

    def test_empty_probes_name_the_checks_that_found_nothing(self, tmp_path, sessions_root, seed_tasks):
        rows = seed_tasks([{'id': 4797, 'status': 'blocked'}])
        handover = tmp_path / 'handover.md'
        handover.write_text('# nothing relevant\n')

        finding = _sweep(_item(), sessions_root=sessions_root, task_rows=rows, handover_path=handover)

        assert finding.empty_probes == mod.PROBES


class TestLoadTaskRows:
    def test_parses_metadata_once_per_row(self, seed_tasks):
        rows = seed_tasks([
            {'id': 1, 'status': 'pending', 'title': 't1', 'metadata': {'x_ruled_by': 'leo'}},
            {'id': 2, 'status': 'done', 'metadata': '{not json'},
            {'id': 3, 'status': 'done'},
        ])

        assert rows.unavailable == ''
        assert rows.rows['1'] == mod.TaskRow(id='1', status='pending', title='t1', metadata={'x_ruled_by': 'leo'})
        assert rows.rows['2'].metadata == {}
        assert rows.rows['3'].metadata == {}

    def test_unreadable_store_is_unavailable_with_its_reason(self, tmp_path):
        rows = mod.load_task_rows(tmp_path / 'absent-project')

        assert rows.rows == {}
        assert 'tasks.db' in rows.unavailable


def _snapshot(*roots: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in roots
        for path in root.rglob('*') if path.is_file()
    }


class TestReadOnly:
    def test_sweep_writes_nothing(self, tmp_path, sessions_root, project_root, seed_tasks):
        _session(sessions_root, 'unblock-df-4797-1', status=Status.EXITED, result='outcome: done\n',
                 role='unblock', project='df', task_id='4797', escalation_id='esc-4797-15')
        rows = seed_tasks([{'id': 4797, 'status': 'blocked', 'metadata': json.dumps({'x_coalesced_into': 5255})}])
        handover = tmp_path / 'handover.md'
        handover.write_text(HANDOVER)
        before = _snapshot(sessions_root, project_root, handover.parent)

        _sweep(_item(), sessions_root=sessions_root, task_rows=rows, handover_path=handover)

        assert _snapshot(sessions_root, project_root, handover.parent) == before
