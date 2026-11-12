"""TaskArtifacts' scope_requests.json trio, the agent->workflow scope-request channel (task 3090)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.scope_drift import ScopeRequest

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
    arts.write_scope_request(ScopeRequest('pkg/b.py', 'plan omitted it'))
    arts.clear_scope_requests()
    assert not arts.artifact_path('scope_requests.json').exists()
    assert arts.read_scope_requests() == ()
    arts.clear_scope_requests()


def test_the_trio_never_touches_plan_json(arts):
    arts.write_plan(json.loads(json.dumps(_PLAN)))
    plan_path = arts.artifact_path('plan.json')
    before = plan_path.read_bytes()
    arts.write_scope_request(ScopeRequest('pkg/b.py', 'plan omitted it'))
    arts.read_scope_requests()
    arts.clear_scope_requests()
    assert plan_path.read_bytes() == before
