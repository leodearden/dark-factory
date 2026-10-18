"""dump_scheduler_candidate_fixture: the live pending population, as the scheduler would score it."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import dump_scheduler_candidate_fixture as dump
import pytest
from orchestrator.scheduler import Scheduler

ANCHOR = '2026-10-01T12:00:00.000Z'

# A(1) <- B(2, done) <- C(3): A unlocks C through the done B.
ROWS = [
    {'id': 1, 'status': 'pending', 'priority': 'high', 'metadata': {'pending_since': ANCHOR}},
    {'id': 2, 'status': 'done', 'priority': 'high', 'metadata': {}},
    {'id': 3, 'status': 'pending', 'priority': 'low', 'metadata': {'pending_since': ANCHOR}},
    {'id': 4, 'status': 'pending', 'priority': 'medium', 'metadata': {'pending_since': ANCHOR}},
    {'id': 5, 'status': 'pending', 'priority': 'medium', 'metadata': {'other': 'x'}},
    {'id': 6, 'status': 'cancelled', 'priority': 'low', 'metadata': {}},
    {'id': 10, 'status': 'pending', 'priority': None, 'metadata': None},
    {'id': 7, 'tag': 'feature', 'status': 'pending', 'priority': 'high', 'metadata': {}},
]
DEPENDENCIES = [('master', 2, 1), ('master', 3, 2), ('feature', 7, 1)]


@pytest.fixture
def store(make_tasks_db):
    db = make_tasks_db(ROWS)
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE dependencies (tag TEXT NOT NULL, task_id INTEGER NOT NULL, '
        'depends_on INTEGER NOT NULL, PRIMARY KEY (tag, task_id, depends_on))'
    )
    conn.executemany('INSERT INTO dependencies VALUES (?, ?, ?)', DEPENDENCIES)
    conn.commit()
    conn.close()
    return db


def _dump(db, out, *extra) -> dict:
    assert dump.main(['--db', str(db), '--out', str(out), *extra]) == 0
    return json.loads(out.read_text())


def _scheduler_d(rows, deps, tag='master') -> dict[str, int]:
    tagged = [r for r in rows if r.get('tag', 'master') == tag]
    tasks = [
        {'id': str(r['id']), 'dependencies': [str(on) for t, tid, on in deps if t == tag and tid == r['id']]}
        for r in tagged
    ]
    return Scheduler._compute_transitive_counts(
        {t['id']: t for t in tasks},
        Scheduler._build_reverse_index(tasks),
        {str(r['id']): r['status'] for r in tagged},
    )


def test_emits_only_pending_rows_of_the_selected_tag(store, tmp_path):
    fixture = _dump(store, tmp_path / 'out.json')

    assert [c['id'] for c in fixture['candidates']] == ['1', '3', '4', '5', '10']


def test_another_tag_is_selectable(store, tmp_path):
    fixture = _dump(store, tmp_path / 'out.json', '--tag', 'feature')

    assert [c['id'] for c in fixture['candidates']] == ['7']


def test_each_candidate_carries_the_scoring_inputs(store, tmp_path):
    by_id = {c['id']: c for c in _dump(store, tmp_path / 'out.json')['candidates']}

    assert by_id['1'] == {'id': '1', 'tier': 'high', 'pending_since': ANCHOR, 'transitive_count': 1}
    assert by_id['5']['pending_since'] is None
    assert by_id['10']['pending_since'] is None
    assert by_id['10']['tier'] == 'medium'
    assert by_id['3']['tier'] == 'low'


def test_transitive_count_is_the_schedulers_own(store, tmp_path):
    candidates = _dump(store, tmp_path / 'out.json')['candidates']
    expected = _scheduler_d(ROWS, DEPENDENCIES)

    assert {c['id']: c['transitive_count'] for c in candidates} == {
        c['id']: expected[c['id']] for c in candidates
    }
    assert expected['1'] == 1


def test_provenance_names_the_capture(store, tmp_path):
    fixture = _dump(store, tmp_path / 'out.json')
    provenance = fixture['provenance']

    assert datetime.fromisoformat(provenance['captured_at']).utcoffset() is not None
    assert provenance['source_db'] == str(store.resolve())
    assert provenance['tag'] == 'master'
    assert provenance['row_count'] == len(fixture['candidates'])
    assert 'pending' in provenance['population']


def test_the_source_store_is_left_byte_identical(store, tmp_path):
    before = store.read_bytes()

    _dump(store, tmp_path / 'out.json')

    assert store.read_bytes() == before


def _assert_refused(argv, out, capsys, named):
    assert dump.main([*argv, '--out', str(out)]) != 0
    assert named in capsys.readouterr().err
    assert not out.exists()


def test_refuses_a_missing_store(tmp_path, capsys):
    missing = tmp_path / 'nowhere' / 'tasks.db'

    _assert_refused(['--db', str(missing)], tmp_path / 'out.json', capsys, str(missing))


def test_refuses_the_zero_byte_decoy(tmp_path, capsys):
    decoy = tmp_path / 'tasks.db'
    decoy.write_bytes(b'')

    _assert_refused(['--db', str(decoy)], tmp_path / 'out.json', capsys, str(decoy))


def test_refuses_to_write_an_empty_population(store, tmp_path, capsys):
    _assert_refused(['--db', str(store), '--tag', 'no-such-tag'], tmp_path / 'out.json', capsys, 'no-such-tag')


def test_project_root_resolves_its_task_store(store, tmp_path):
    root = tmp_path / 'root'
    (root / '.taskmaster' / 'tasks').mkdir(parents=True)
    (root / '.taskmaster' / 'tasks' / 'tasks.db').write_bytes(store.read_bytes())

    assert dump.main(['--project-root', str(root), '--out', str(tmp_path / 'out.json')]) == 0
