"""Tests for scripts/sitting/inventory.py — the fleet-wide open-question inventory (task 5376)."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from escalation.models import Escalation
from orchestrator.session_registry import (
    UNKNOWN_QUEUE,
    DecisionRecord,
    list_decisions,
    normalize_escalations_dir,
)
from sitting import inventory as mod

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def _write_decision(root: Path, **fields) -> DecisionRecord:
    fields.setdefault('project', 'dark_factory')
    fields.setdefault('text', f"question {fields['id']}")
    fields.setdefault('filed_at', '2026-09-20T12:00:00+00:00')
    record = DecisionRecord(**fields)
    decisions = root / 'decisions'
    decisions.mkdir(parents=True, exist_ok=True)
    (decisions / f'{record.id}.json').write_text(record.to_json())
    return record


def _write_escalation(queue: Path, *, subdir: str = '', **fields) -> Escalation:
    fields.setdefault('task_id', fields['id'].split('-')[1])
    fields.setdefault('agent_role', 'escalation-watcher-auto')
    fields.setdefault('severity', 'blocking')
    fields.setdefault('category', 'design_concern')
    fields.setdefault('summary', f"summary of {fields['id']}")
    fields.setdefault('timestamp', '2026-09-21T12:00:00+00:00')
    esc = Escalation(**fields)
    target = queue / subdir if subdir else queue
    target.mkdir(parents=True, exist_ok=True)
    (target / f'{esc.id}.json').write_text(esc.to_json())
    return esc


@pytest.fixture
def fleet(tmp_path):
    root = tmp_path / 'fleet'
    root.mkdir()
    return root


@pytest.fixture
def df_root(tmp_path):
    root = tmp_path / 'src' / 'dark-factory'
    (root / 'data' / 'escalations').mkdir(parents=True)
    (root / 'data' / 'reconciliation' / 'escalations').mkdir(parents=True)
    return root


def _queue(root: Path) -> str:
    return normalize_escalations_dir(root / 'data' / 'escalations')


def _recon_queue(root: Path) -> str:
    return normalize_escalations_dir(root / 'data' / 'reconciliation' / 'escalations')


def _collect(queue_dirs, fleet, **kwargs):
    return mod.collect_open_items(queue_dirs=queue_dirs, decisions_root=fleet, now=NOW, **kwargs)


class TestEscalationQueueDirs:
    def test_union_of_root_queues_and_registry_dirs(self, tmp_path, df_root):
        other = tmp_path / 'src' / 'reify'
        (other / 'data' / 'escalations').mkdir(parents=True)
        registry_only = tmp_path / 'src' / 'pump-web-ui' / 'data' / 'escalations'
        registry_only.mkdir(parents=True)
        decisions = [
            DecisionRecord('d1', 'pump_web_ui', 't', 'x', escalations_dir=str(registry_only)),
            DecisionRecord('d2', 'reify', 't', 'x', escalations_dir=str(other / 'data' / 'escalations')),
        ]

        dirs = mod.escalation_queue_dirs([str(df_root), str(other)], decisions)

        assert dirs == (
            _queue(df_root),
            _recon_queue(df_root),
            _queue(other),
            normalize_escalations_dir(registry_only),
        )

    def test_sentinels_missing_dirs_and_queueless_roots_are_skipped(self, tmp_path):
        queueless = tmp_path / 'src' / 'no-queue'
        queueless.mkdir(parents=True)
        decisions = [
            DecisionRecord('d1', 'p', 't', 'x', escalations_dir=''),
            DecisionRecord('d2', 'p', 't', 'x', escalations_dir=UNKNOWN_QUEUE),
            DecisionRecord('d3', 'p', 't', 'x', escalations_dir=str(tmp_path / 'gone' / 'data' / 'escalations')),
        ]

        assert mod.escalation_queue_dirs([str(queueless)], decisions) == ()

    def test_spellings_of_one_queue_dedupe(self, df_root):
        spelled = str(df_root / 'data' / '..' / 'data' / 'escalations') + '/'
        decisions = [DecisionRecord('d1', 'dark_factory', 't', 'x', escalations_dir=spelled)]

        dirs = mod.escalation_queue_dirs([str(df_root)], decisions)

        assert dirs == (_queue(df_root), _recon_queue(df_root))


class TestCollectOpenItems:
    def test_pending_l2_and_open_decisions_only(self, fleet, df_root):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(queue, id='esc-10-1', level=2)
        _write_escalation(queue, id='esc-11-1', level=1)
        _write_escalation(queue, id='esc-12-1', level=0)
        _write_escalation(queue, id='esc-13-1', level=2, status='resolved')
        _write_escalation(queue, subdir='archive/2026-09-01', id='esc-14-1', level=2, status='dismissed')
        _write_decision(fleet, id='open-1')
        _write_decision(fleet, id='answered-1', state='answered')
        _write_decision(fleet, id='dropped-1', state='dropped')

        inv = _collect([_queue(df_root)], fleet)

        assert {(i.kind, i.escalation_id, i.decision_id) for i in inv.items} == {
            ('esc', 'esc-10-1', None),
            ('decision', None, 'open-1'),
        }
        assert inv.shortfalls == ()
        assert all(isinstance(i, mod.OpenItem) for i in inv.items)
        with pytest.raises(AttributeError):
            inv.items[0].text = 'mutated'  # type: ignore[misc]

    def test_escalation_item_carries_the_record_fields(self, fleet, df_root):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(
            queue, id='esc-3105-3', level=2, severity='critical', summary='pin me',
            options=['A: close', 'B: hold'], members=['esc-3105-1', 'esc-3105-2'],
            root_cause='veto-pin-do-not-close:3105', pin_declared_by=['leo'],
            triage_note='verified 2026-09-25', timestamp='2026-09-16T12:00:00+00:00',
        )

        (item,) = _collect([_queue(df_root)], fleet).items

        assert item.queue_dir == _queue(df_root)
        assert item.project == 'dark_factory'
        assert item.task_id == '3105'
        assert item.severity == 'critical'
        assert item.text == 'pin me'
        assert item.options == ('A: close', 'B: hold')
        assert item.members == ('esc-3105-1', 'esc-3105-2')
        assert item.root_cause == 'veto-pin-do-not-close:3105'
        assert item.pin_declared_by == ('leo',)
        assert item.triage_note == 'verified 2026-09-25'
        assert item.filed_at == '2026-09-16T12:00:00+00:00'

    def test_project_tokens_fold_and_filter(self, fleet, df_root):
        _write_decision(fleet, id='a', project='dark_factory')
        _write_decision(fleet, id='b', project='df')
        _write_decision(fleet, id='c', project='dark-factory')
        _write_decision(fleet, id='d', project='reify')
        _write_escalation(df_root / 'data' / 'escalations', id='esc-20-1', level=2)

        everything = _collect([_queue(df_root)], fleet)
        folded = _collect([_queue(df_root)], fleet, project='df')

        assert {i.project for i in everything.items if i.decision_id in {'a', 'b', 'c'}} == {'dark_factory'}
        assert {i.decision_id or i.escalation_id for i in folded.items} == {'a', 'b', 'c', 'esc-20-1'}

    def test_decision_matching_a_pending_l2_in_its_queue_folds_into_one_item(self, fleet, df_root):
        _write_escalation(df_root / 'data' / 'escalations', id='esc-30-1', level=2)
        _write_decision(fleet, id='df-esc-30-1', escalation_id='esc-30-1', escalations_dir=_queue(df_root))

        (item,) = _collect([_queue(df_root)], fleet).items

        assert item.kind == 'esc'
        assert item.escalation_id == 'esc-30-1'
        assert item.decision_id == 'df-esc-30-1'

    def test_same_escalation_id_in_a_different_queue_stays_separate(self, fleet, df_root):
        _write_escalation(df_root / 'data' / 'escalations', id='esc-31-1', level=2)
        _write_decision(fleet, id='recon-esc-31-1', escalation_id='esc-31-1', escalations_dir=_recon_queue(df_root))

        items = _collect([_queue(df_root), _recon_queue(df_root)], fleet).items

        assert {(i.kind, i.escalation_id, i.decision_id) for i in items} == {
            ('esc', 'esc-31-1', None),
            ('decision', 'esc-31-1', 'recon-esc-31-1'),
        }

    def test_key_is_structured_and_round_trips(self, fleet, df_root):
        _write_escalation(df_root / 'data' / 'escalations', id='esc-40-1', level=2)
        _write_decision(fleet, id='lone-1')

        items = {i.kind: i for i in _collect([_queue(df_root)], fleet).items}

        assert items['esc'].key == ('esc', _queue(df_root), 'esc-40-1')
        assert items['decision'].key == ('decision', 'lone-1')
        for item in items.values():
            text = mod.key_str(item.key)
            assert isinstance(text, str)
            assert mod.parse_key(text) == item.key

    def test_age_days_uses_the_injected_now(self, fleet, df_root):
        _write_escalation(
            df_root / 'data' / 'escalations', id='esc-50-1', level=2, timestamp='2026-09-16T12:00:00+00:00',
        )
        _write_decision(fleet, id='aged', filed_at='2026-09-25T00:00:00+00:00')

        items = {i.kind: i for i in _collect([_queue(df_root)], fleet).items}

        assert items['esc'].age_days == pytest.approx(10.0)
        assert items['decision'].age_days == pytest.approx(1.5)

    def test_escalation_index_covers_root_and_archive_per_queue(self, fleet, df_root):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(queue, id='esc-60-1', level=2)
        _write_escalation(queue, subdir='archive/2026-09-01', id='esc-60-2', status='resolved')
        _write_escalation(df_root / 'data' / 'reconciliation' / 'escalations', id='esc-60-2', status='resolved')

        inv = _collect([_queue(df_root), _recon_queue(df_root)], fleet)

        assert inv.escalation_index[_queue(df_root)] == {
            'esc-60-1': queue / 'esc-60-1.json',
            'esc-60-2': queue / 'archive' / '2026-09-01' / 'esc-60-2.json',
        }
        assert set(inv.escalation_index[_recon_queue(df_root)]) == {'esc-60-2'}


class TestFailOpen:
    def test_truncated_files_in_either_store_are_counted_shortfalls(self, fleet, df_root):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(queue, id='esc-70-1', level=2)
        (queue / 'esc-70-2.json').write_text('{"id": "esc-70-2", "task_')
        _write_decision(fleet, id='good')
        (fleet / 'decisions' / 'bad.json').write_text('{"id": "bad", "pro')

        inv = _collect([_queue(df_root)], fleet)

        assert {i.escalation_id or i.decision_id for i in inv.items} == {'esc-70-1', 'good'}
        assert sorted(Path(s.path).name for s in inv.shortfalls) == ['bad.json', 'esc-70-2.json']

    def test_missing_decisions_root_yields_no_decision_items(self, tmp_path, df_root):
        _write_escalation(df_root / 'data' / 'escalations', id='esc-71-1', level=2)

        inv = _collect([_queue(df_root)], tmp_path / 'no-fleet-here')

        assert [i.kind for i in inv.items] == ['esc']
        assert inv.shortfalls == ()


class TestGlossary:
    def test_escalation_and_task_citations_are_glossed(self, fleet, df_root, make_tasks_db):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(
            queue, id='esc-80-1', level=2, summary='the question', members=['esc-80-2'],
            triage_note='see esc-81-1 and task 90',
        )
        _write_escalation(queue, subdir='archive/2026-09-01', id='esc-80-2', status='resolved', summary='member')
        _write_escalation(queue, subdir='archive/2026-09-02', id='esc-81-1', status='resolved', summary='cited')
        make_tasks_db(
            [{'id': 80, 'title': 'subject task'}, {'id': 90, 'title': 'cited task'}],
            directory=df_root / '.taskmaster' / 'tasks',
        )
        inv = _collect([_queue(df_root)], fleet)

        glossary = mod.build_glossary(inv.items, inv.escalation_index)

        assert glossary.escalations[_queue(df_root)] == {
            'esc-80-1': 'the question',
            'esc-80-2': 'member',
            'esc-81-1': 'cited',
        }
        assert glossary.tasks['dark_factory'] == {'80': 'subject task', '90': 'cited task'}
        assert glossary.shortfalls == ()

    def test_missing_tasks_db_is_a_named_shortfall(self, fleet, df_root):
        _write_escalation(df_root / 'data' / 'escalations', id='esc-82-1', level=2)
        inv = _collect([_queue(df_root)], fleet)

        glossary = mod.build_glossary(inv.items, inv.escalation_index)

        assert glossary.tasks == {}
        (shortfall,) = glossary.shortfalls
        assert shortfall.path.endswith('.taskmaster/tasks/tasks.db')


def _snapshot(*roots: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in roots if root.exists()
        for path in root.rglob('*') if path.is_file()
    }


class TestReadOnly:
    def test_collect_and_glossary_write_nothing(self, tmp_path, fleet, df_root, make_tasks_db):
        queue = df_root / 'data' / 'escalations'
        _write_escalation(queue, id='esc-90-1', level=2, members=['esc-90-2'])
        _write_escalation(queue, subdir='archive/2026-09-01', id='esc-90-2', status='resolved')
        _write_decision(fleet, id='df-esc-90-1', escalation_id='esc-90-1', escalations_dir=_queue(df_root))
        make_tasks_db([{'id': 90}], directory=df_root / '.taskmaster' / 'tasks')
        absent_queue = tmp_path / 'src' / 'reify' / 'data' / 'escalations'
        before = _snapshot(fleet, df_root)

        dirs = mod.escalation_queue_dirs([str(df_root), str(tmp_path / 'src' / 'reify')], list_decisions(fleet))
        inv = _collect([*dirs, str(absent_queue)], fleet)
        mod.build_glossary(inv.items, inv.escalation_index)

        assert _snapshot(fleet, df_root) == before
        assert not [p for p in tmp_path.rglob('*.lock')]
        assert not absent_queue.exists()
