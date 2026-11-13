"""TaskArtifacts' scope-request channels (task 3090).

scope_requests.json carries an agent's requests to the workflow;
scope_drift_brief.json carries a boundary's report to the next writer.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scope_drift import ScopeBoundary, ScopeRequest, build_scope_drift_report

_PLAN = {
    'files': ['pkg/a.py'],
    'steps': [{'id': 'step-1', 'status': 'pending', 'commit': None}],
}


@pytest.fixture
def lane(tmp_path: Path) -> Path:
    root = tmp_path / 'lane'
    root.mkdir()
    return root


@pytest.fixture
def task(tmp_path: Path) -> Path:
    root = tmp_path / 'task'
    root.mkdir()
    return root


@pytest.fixture
def arts(tmp_path: Path, lane: Path, task: Path) -> TaskArtifacts:
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    return TaskArtifacts(worktree, meta_root=lane, task_root=task)


def _artifact_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == 'orchestrator.artifacts' and r.levelno == logging.WARNING
    ]


def test_scope_requests_resolve_under_the_lane_root(arts, lane):
    assert arts.artifact_path('scope_requests.json') == lane / 'scope_requests.json'


def test_a_fresh_root_reads_no_requests(arts):
    assert arts.read_scope_requests() == ()


def test_requests_read_back_in_write_order(arts):
    first = ScopeRequest('pkg/b.py', 'plan omitted it')
    second = ScopeRequest('pkg/c.py', 'the helper lives there')
    arts.write_scope_request(first)
    arts.write_scope_request(second)
    assert arts.read_scope_requests() == (first, second)


def test_rewriting_a_path_keeps_one_entry_with_the_latest_reason(arts):
    arts.write_scope_request(ScopeRequest('pkg/b.py', 'first reason'))
    arts.write_scope_request(ScopeRequest('pkg/c.py', 'other'))
    arts.write_scope_request(ScopeRequest('pkg/b.py', 'latest reason'))
    requests = arts.read_scope_requests()
    assert [r.path for r in requests].count('pkg/b.py') == 1
    assert ScopeRequest('pkg/b.py', 'latest reason') in requests
    assert ScopeRequest('pkg/c.py', 'other') in requests
    assert len(requests) == 2


def test_invalid_json_reads_as_no_requests_with_a_warning(arts, caplog):
    arts.artifact_path('scope_requests.json').write_text('{not json', encoding='utf-8')
    with caplog.at_level(logging.WARNING, logger='orchestrator.artifacts'):
        assert arts.read_scope_requests() == ()
    assert any('scope_requests.json' in message for message in _artifact_warnings(caplog))


def test_an_invalid_entry_is_dropped_with_a_warning_naming_it(arts, caplog):
    arts.artifact_path('scope_requests.json').write_text(
        json.dumps({
            'requests': [
                {'path': '../escape.py', 'reason': 'bad'},
                {'path': 'pkg/b.py', 'reason': 'good'},
            ]
        }),
        encoding='utf-8',
    )
    with caplog.at_level(logging.WARNING, logger='orchestrator.artifacts'):
        assert arts.read_scope_requests() == (ScopeRequest('pkg/b.py', 'good'),)
    assert any('../escape.py' in message for message in _artifact_warnings(caplog))


def test_clear_removes_the_artifact_and_tolerates_absence(arts):
    request = ScopeRequest('pkg/b.py', 'plan omitted it')
    arts.write_scope_request(request)
    arts.clear_scope_requests([request])
    assert not arts.artifact_path('scope_requests.json').exists()
    assert arts.read_scope_requests() == ()
    arts.clear_scope_requests([request])


def test_clear_keeps_a_request_filed_after_the_applied_ones_were_read(arts):
    applied = ScopeRequest('pkg/b.py', 'plan omitted it')
    later = ScopeRequest('pkg/c.py', 'filed meanwhile')
    arts.write_scope_request(applied)
    read = arts.read_scope_requests()
    arts.write_scope_request(later)
    arts.clear_scope_requests(read)
    assert arts.read_scope_requests() == (later,)


def test_write_returns_the_pending_count(arts):
    assert arts.write_scope_request(ScopeRequest('pkg/b.py', 'one')) == 1
    assert arts.write_scope_request(ScopeRequest('pkg/c.py', 'two')) == 2
    assert arts.write_scope_request(ScopeRequest('pkg/b.py', 'again')) == 2


def test_concurrent_writes_from_threads_lose_no_request(arts):
    paths = [f'pkg/m{n}.py' for n in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda path: arts.write_scope_request(ScopeRequest(path, 'r')), paths))
    assert sorted(r.path for r in arts.read_scope_requests()) == sorted(paths)


def test_the_trio_never_touches_plan_json(arts):
    arts.write_plan(json.loads(json.dumps(_PLAN)))
    plan_path = arts.artifact_path('plan.json')
    before = plan_path.read_bytes()
    arts.write_scope_request(ScopeRequest('pkg/b.py', 'plan omitted it'))
    arts.clear_scope_requests(arts.read_scope_requests())
    assert plan_path.read_bytes() == before


def _contested_report():
    return build_scope_drift_report(
        ['pkg/b.py'], [Blocker('pkg/b.py', '777', BlockerKind.HELD)],
        lock_depth=12, boundary=ScopeBoundary.DEBUG_FIX, role='debugger',
        iteration=3, task_id='42',
    )


def test_the_brief_resolves_under_the_task_root(arts, task):
    assert arts.artifact_path('scope_drift_brief.json') == task / 'scope_drift_brief.json'


def test_a_fresh_root_has_no_brief(arts):
    assert arts.read_scope_drift_brief() is None


def test_the_brief_reads_back_and_clears(arts):
    report = _contested_report()
    arts.write_scope_drift_brief(report)
    assert arts.read_scope_drift_brief() == report
    arts.clear_scope_drift_brief()
    assert arts.read_scope_drift_brief() is None
    arts.clear_scope_drift_brief()


def test_a_later_dispatch_into_another_lane_reads_the_brief(arts, tmp_path, task):
    arts.write_scope_drift_brief(_contested_report())
    other_lane = tmp_path / 'lane-2'
    other_lane.mkdir()
    redispatched = TaskArtifacts(tmp_path / 'wt', meta_root=other_lane, task_root=task)
    assert redispatched.read_scope_drift_brief() == _contested_report()


def test_a_malformed_brief_reads_as_none_with_a_warning(arts, caplog):
    arts.artifact_path('scope_drift_brief.json').write_text(
        json.dumps({'reason': 'made_up'}), encoding='utf-8',
    )
    with caplog.at_level(logging.WARNING, logger='orchestrator.artifacts'):
        assert arts.read_scope_drift_brief() is None
    assert any('scope_drift_brief.json' in message for message in _artifact_warnings(caplog))
