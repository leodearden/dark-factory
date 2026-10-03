"""Every eval id and path the harness mints is one the containment recognizer knows.

Eval-lane escalation containment (``shared/src/shared/eval_lane.py``) recognises
eval provenance by a fixture-id grammar and an eval-worktree path marker. Those
are conventions owned by the minters below, so each minter's real output is fed
to the recognizer here: a rename on either side fails a test instead of
silently letting eval-lane escalations reach the production human L2 queue.
``runner.load_task`` enforces the same grammar at run time.

Hermetic: no LLM, no git, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.eval_lane import is_eval_fixture_task_id, is_eval_worktree_path

from orchestrator.evals import runner
from orchestrator.evals.live_fixture import ShadowShape, build_live_fixture
from orchestrator.evals.snapshots import eval_worktree_root
from orchestrator.evals.task_sampler import (
    CompletedTaskCandidate,
    ReferenceCapture,
    build_fixture_record,
    default_verify_commands,
)

_EVALS_DIR = Path(runner.__file__).parent
_FIXTURE_DIRS = (_EVALS_DIR / 'tasks', _EVALS_DIR / 'tasks_hard_v2')
_SHA = 'a' * 40


def _shipped_fixtures() -> list[Path]:
    return sorted(path for d in _FIXTURE_DIRS for path in d.glob('*.json'))


@pytest.mark.parametrize('fixture_dir', _FIXTURE_DIRS, ids=lambda d: d.name)
def test_each_fixture_dir_ships_fixtures(fixture_dir: Path) -> None:
    assert sorted(fixture_dir.glob('*.json')), f'{fixture_dir} holds no fixtures'


@pytest.mark.parametrize('path', _shipped_fixtures(), ids=lambda p: f'{p.parent.name}/{p.stem}')
def test_shipped_fixture_id_is_recognised(path: Path) -> None:
    fixture_id = json.loads(path.read_text())['id']
    assert is_eval_fixture_task_id(fixture_id), fixture_id
    assert fixture_id == path.stem


@pytest.mark.parametrize(
    ('project', 'project_root'),
    [('dark_factory', '/home/leo/src/dark-factory'), ('reify', '/home/leo/src/reify')],
)
def test_sampler_minted_id_is_recognised(project: str, project_root: str) -> None:
    candidate = CompletedTaskCandidate(
        task_id='4242',
        project=project,
        project_root=project_root,
        title='Add capability to the module',
        pre_commit='b' * 40,
        post_commit='c' * 40,
        merge_sha='c' * 40,
    )
    record = build_fixture_record(
        candidate,
        ReferenceCapture(post_task_commit='c' * 40, files=1, insertions=1, deletions=0),
        {'test': 'true', 'lint': 'true', 'typecheck': 'true'},
        plan=None,
        cohort='containment',
        sampled_at='2026-10-01T00:00:00+00:00',
    )
    assert is_eval_fixture_task_id(record['id']), record['id']


def test_live_shadow_minted_id_is_recognised(tmp_path: Path) -> None:
    fixture = build_live_fixture(
        {'id': 5383, 'title': 't', 'metadata': {}},
        base_sha=_SHA,
        project_root=tmp_path,
        plan={'task_id': '5383', 'steps': [{'id': 'step-1', 'status': 'pending'}]},
        verify_commands=default_verify_commands('df'),
        shape=ShadowShape.IMPLEMENTER,
        cell_id='01J9ZK3QWERTY0123456789ABC',
    )
    assert is_eval_fixture_task_id(fixture['id']), fixture['id']


@pytest.mark.parametrize(
    'project_root', ['/home/leo/src/dark-factory', '/home/leo/src/eval-worktree-tools']
)
def test_eval_worktree_root_is_recognised(project_root: str) -> None:
    assert not is_eval_worktree_path(project_root)
    run_dir = eval_worktree_root(Path(project_root)) / 'df_task_12' / 'run-abc12345'
    assert is_eval_worktree_path(run_dir)


def _write_fixture(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps({'project_root': '$REPO_ROOT', **payload}))
    return path


@pytest.mark.parametrize(
    'payload',
    [{'id': 'my-fixture'}, {}],
    ids=['off-grammar-id', 'missing-id'],
)
def test_load_task_rejects_an_unrecognised_fixture_id(tmp_path: Path, payload: dict) -> None:
    path = _write_fixture(tmp_path, payload)

    with pytest.raises(ValueError) as excinfo:
        runner.load_task(path)

    message = str(excinfo.value)
    assert str(path) in message
    assert repr(payload.get('id')) in message


@pytest.mark.parametrize('fixture_id', ['df_task_9999_adv_smoke', 'shadow_5383_01JCELL'])
def test_load_task_accepts_a_recognised_fixture_id(tmp_path: Path, fixture_id: str) -> None:
    loaded = runner.load_task(_write_fixture(tmp_path, {'id': fixture_id}))

    assert loaded['id'] == fixture_id
