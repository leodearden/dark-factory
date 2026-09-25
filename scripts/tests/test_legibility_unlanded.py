"""Tests for scripts/legibility/unlanded.py — putting back a legibility write
whose commit did not land.

Every test runs against a REAL tmp git repo, because the behaviour under test
is what git does to the index and the worktree. The conftest autouse fixture
``_isolate_legibility_trickle_state`` already points the legibility state
root, and therefore the quarantine, at a per-test tmp dir.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from legibility import trickle_state, unlanded

_CODEBOOK = 'docs/legibility/confusion-codebook.yaml'
_CENSUS_STATE = 'docs/legibility/census-state.json'
_HEAD_CODEBOOK = b'version: 2\nentries: []\n'
_HEAD_CENSUS_STATE = b'{"last_census_at": "2026-06-01"}\n'
_REFUSED = b'refused bytes\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    (repo / 'docs' / 'legibility').mkdir(parents=True)
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@example.com')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / _CODEBOOK).write_bytes(_HEAD_CODEBOOK)
    (repo / _CENSUS_STATE).write_bytes(_HEAD_CENSUS_STATE)
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'initial')
    return repo


def _head_bytes(repo: Path, rel: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repo), 'show', f'HEAD:{rel}'], check=True, capture_output=True,
    ).stdout


def _porcelain(repo: Path, *paths: str) -> str:
    return _git(repo, 'status', '--porcelain', '--', *paths)


def test_roll_back_restores_tracked_paths_to_head_and_quarantines_the_refused_bytes(
    tmp_path,
):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED + b'codebook\n')
    (repo / _CENSUS_STATE).write_bytes(_REFUSED + b'state\n')
    _git(repo, 'add', _CODEBOOK)

    rb = unlanded.roll_back(
        repo,
        [Path(_CODEBOOK), repo / _CENSUS_STATE],
        project_id='proj',
        label='census-2026-07-14',
    )

    assert rb.restored is True
    assert rb.failure is None
    assert (repo / _CODEBOOK).read_bytes() == _head_bytes(repo, _CODEBOOK)
    assert (repo / _CENSUS_STATE).read_bytes() == _head_bytes(repo, _CENSUS_STATE)
    assert _porcelain(repo, _CODEBOOK, _CENSUS_STATE) == ''
    assert rb.quarantine_dir is not None
    assert rb.quarantine_dir.parent == unlanded.quarantine_root('proj')
    assert rb.quarantine_dir.name.startswith('census-2026-07-14-')
    assert (rb.quarantine_dir / _CODEBOOK).read_bytes() == _REFUSED + b'codebook\n'
    assert (rb.quarantine_dir / _CENSUS_STATE).read_bytes() == _REFUSED + b'state\n'
    assert set(rb.paths) == {_CODEBOOK, _CENSUS_STATE}


def test_roll_back_removes_written_paths_head_does_not_track(tmp_path):
    repo = _repo(tmp_path)
    staged_new = 'plans/confusion-census-2026-07-14.md'
    untracked = 'plans/confusion-census-2026-07-14-payloads.json'
    (repo / 'plans').mkdir()
    (repo / staged_new).write_bytes(b'# report\n')
    (repo / untracked).write_bytes(b'[]\n')
    _git(repo, 'add', staged_new)

    rb = unlanded.roll_back(
        repo, [staged_new, untracked], project_id='proj', label='census-2026-07-14',
    )

    assert rb.restored is True
    assert not (repo / staged_new).exists()
    assert not (repo / untracked).exists()
    assert _git(repo, 'ls-files', '--', staged_new, untracked) == ''
    assert rb.quarantine_dir is not None
    assert (rb.quarantine_dir / staged_new).read_bytes() == b'# report\n'
    assert (rb.quarantine_dir / untracked).read_bytes() == b'[]\n'


def test_roll_back_leaves_every_other_path_alone(tmp_path):
    repo = _repo(tmp_path)
    (repo / 'unrelated-new.txt').write_bytes(b'staged elsewhere\n')
    _git(repo, 'add', 'unrelated-new.txt')
    (repo / _CENSUS_STATE).write_bytes(b'someone else is editing this\n')
    (repo / _CODEBOOK).write_bytes(_REFUSED)

    rb = unlanded.roll_back(repo, [_CODEBOOK], project_id='proj', label='trickle-2026-07-13')

    assert rb.restored is True
    assert _porcelain(repo, 'unrelated-new.txt') == 'A  unrelated-new.txt\n'
    assert (repo / 'unrelated-new.txt').read_bytes() == b'staged elsewhere\n'
    assert _porcelain(repo, _CENSUS_STATE) == f' M {_CENSUS_STATE}\n'
    assert (repo / _CENSUS_STATE).read_bytes() == b'someone else is editing this\n'


def test_roll_back_never_touches_the_checkout_when_the_quarantine_cannot_be_written(
    tmp_path, monkeypatch,
):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    blocker = tmp_path / 'a-regular-file'
    blocker.write_text('not a directory\n')
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(blocker / 'root'))

    rb = unlanded.roll_back(repo, [_CODEBOOK], project_id='proj', label='trickle-2026-07-13')

    assert rb.restored is False
    assert rb.quarantine_dir is None
    assert rb.failure
    assert (repo / _CODEBOOK).read_bytes() == _REFUSED


def test_roll_back_keeps_the_quarantine_when_the_restore_fails(tmp_path):
    not_a_repo = tmp_path / 'not-a-repo'
    not_a_repo.mkdir()
    (not_a_repo / 'written.yaml').write_bytes(_REFUSED)

    rb = unlanded.roll_back(
        not_a_repo, ['written.yaml'], project_id='proj', label='trickle-2026-07-13',
    )

    assert rb.restored is False
    assert rb.failure
    assert rb.quarantine_dir is not None
    assert (rb.quarantine_dir / 'written.yaml').read_bytes() == _REFUSED
    assert (not_a_repo / 'written.yaml').read_bytes() == _REFUSED
    assert str(rb.quarantine_dir) in rb.describe()
    assert rb.failure in rb.describe()


def test_roll_back_refuses_a_path_outside_the_repo(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    outside = tmp_path / 'outside.txt'
    outside.write_bytes(b'not the repo\'s\n')

    rb = unlanded.roll_back(
        repo, [_CODEBOOK, outside], project_id='proj', label='trickle-2026-07-13',
    )

    assert rb.restored is False
    assert outside.read_bytes() == b'not the repo\'s\n'
    assert (repo / _CODEBOOK).read_bytes() == _REFUSED


def test_quarantine_root_lives_in_the_per_project_legibility_state_dir(
    tmp_path, monkeypatch,
):
    state_root = tmp_path / 'state-root'
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(state_root))

    root = unlanded.quarantine_root('p')

    assert root.parent == trickle_state.trickle_state_path('p').parent
    assert root.is_relative_to(state_root)


def test_describe_names_the_quarantine_and_the_restored_paths(tmp_path):
    repo = _repo(tmp_path)
    (repo / _CODEBOOK).write_bytes(_REFUSED)
    (repo / _CENSUS_STATE).write_bytes(_REFUSED)

    rb = unlanded.roll_back(
        repo, [_CODEBOOK, _CENSUS_STATE], project_id='proj', label='census-2026-07-14',
    )

    assert rb.restored is True
    description = rb.describe()
    assert str(rb.quarantine_dir) in description
    assert _CODEBOOK in description
    assert _CENSUS_STATE in description
