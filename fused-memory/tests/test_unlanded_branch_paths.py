"""Behaviour of ``services/unlanded_branch_paths.py::probe_unlanded_paths`` against REAL git.

Every fixture is a tmp repository whose task branch is merged into ``main``
with ``--no-ff``, the merge lane's shape. The verdict IS git semantics, so a
canned ``run_git`` fake would only pin our assumptions about git.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo
from _merged_branch_repo import (
    INCIDENT_GUARD,
    commit_file,
    delete_file,
    git,
    incident_merge,
    merge_no_ff,
)

from fused_memory.services.unlanded_branch_paths import (
    ProbeOutcome,
    UnlandedPath,
    UnlandedPathsProbe,
    probe_unlanded_paths,
)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Stop git discovering any repository above tmp_path (the non-git case
    # must be genuinely non-git wherever pytest puts its tmp tree).
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'repo'
    root.mkdir()
    _init_git_repo(root)
    return root


def _branch(repo: Path, name: str) -> None:
    git(repo, 'checkout', '-q', '-b', name, 'main')


async def _probe(repo: Path, merge_sha: str) -> UnlandedPathsProbe:
    return await probe_unlanded_paths(str(repo), merge_sha)


def _paths(probe: UnlandedPathsProbe) -> set[str]:
    return {item.path for item in probe.unlanded}


@pytest.mark.asyncio
async def test_incident_shape_reports_the_added_then_deleted_path(repo: Path) -> None:
    incident = incident_merge(repo, '107')

    probe = await _probe(repo, incident.merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert probe.merge_commit == incident.merge
    assert probe.unlanded == (UnlandedPath(path=INCIDENT_GUARD, branch_commits=incident.guard_commits),)


@pytest.mark.asyncio
async def test_deleting_a_pre_existing_file_is_a_landed_deletion(repo: Path) -> None:
    commit_file(repo, 'old.py', 'old\n', 'seed old')
    _branch(repo, 'task/1')
    delete_file(repo, 'old.py', 'remove old')
    merge = merge_no_ff(repo, 'task/1')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert probe.unlanded == ()


@pytest.mark.asyncio
async def test_modify_then_revert_of_a_pre_existing_file_is_not_unlanded(repo: Path) -> None:
    commit_file(repo, 'keep.py', 'original\n', 'seed keep')
    _branch(repo, 'task/2')
    commit_file(repo, 'keep.py', 'changed\n', 'change keep')
    commit_file(repo, 'keep.py', 'original\n', 'revert keep')
    commit_file(repo, 'new.py', 'new\n', 'add new')
    merge = merge_no_ff(repo, 'task/2')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert probe.unlanded == ()


@pytest.mark.asyncio
async def test_path_main_added_independently_is_present_at_merge(repo: Path) -> None:
    _branch(repo, 'task/3')
    commit_file(repo, 'x.py', 'branch\n', 'branch adds x')
    delete_file(repo, 'x.py', 'branch drops x')
    git(repo, 'checkout', '-q', 'main')
    commit_file(repo, 'x.py', 'main\n', 'main adds x')
    merge = merge_no_ff(repo, 'task/3')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert 'x.py' not in _paths(probe)


@pytest.mark.asyncio
async def test_main_commits_merged_into_the_branch_are_not_attributed_to_it(repo: Path) -> None:
    _branch(repo, 'task/4')
    commit_file(repo, 'own.py', 'own\n', 'branch work')
    git(repo, 'checkout', '-q', 'main')
    commit_file(repo, 'q.py', 'q\n', 'main adds q')
    delete_file(repo, 'q.py', 'main drops q')
    git(repo, 'checkout', '-q', 'task/4')
    git(repo, 'merge', '-q', '--no-ff', '-m', 'Merge main into task/4', 'main')
    merge = merge_no_ff(repo, 'task/4')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert 'q.py' not in _paths(probe)


@pytest.mark.asyncio
async def test_rename_reports_the_pre_rename_name_only(repo: Path) -> None:
    _branch(repo, 'task/5')
    commit_file(repo, 'a.py', 'body\n', 'add a')
    git(repo, 'mv', 'a.py', 'b.py')
    git(repo, 'commit', '-q', '-m', 'rename a to b')
    merge = merge_no_ff(repo, 'task/5')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert _paths(probe) == {'a.py'}


@pytest.mark.asyncio
async def test_clean_branch_is_analyzed_with_nothing_unlanded(repo: Path) -> None:
    _branch(repo, 'task/6')
    commit_file(repo, 'src/one.py', '1\n', 'add one')
    commit_file(repo, 'src/two.py', '2\n', 'add two')
    merge = merge_no_ff(repo, 'task/6')

    probe = await _probe(repo, merge)

    assert probe.outcome is ProbeOutcome.ANALYZED
    assert probe.unlanded == ()


@pytest.mark.asyncio
async def test_non_merge_commit_is_not_a_merge(repo: Path) -> None:
    seed = git(repo, 'rev-parse', 'HEAD')

    probe = await _probe(repo, seed)

    assert probe.outcome is ProbeOutcome.NOT_A_MERGE
    assert probe.merge_commit == seed
    assert probe.unlanded == ()


@pytest.mark.asyncio
async def test_unknown_sha_is_unknown(repo: Path) -> None:
    probe = await _probe(repo, '0' * 40)

    assert probe.outcome is ProbeOutcome.UNKNOWN
    assert probe.unlanded == ()


@pytest.mark.asyncio
async def test_non_git_directory_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    plain = tmp_path / 'plain'
    plain.mkdir()

    probe = await probe_unlanded_paths(str(plain), '0' * 40)

    assert probe.outcome is ProbeOutcome.UNKNOWN
    assert probe.unlanded == ()


def test_probe_and_unlanded_path_are_frozen() -> None:
    item = UnlandedPath(path=INCIDENT_GUARD, branch_commits=('a' * 40,))
    probe = UnlandedPathsProbe(outcome=ProbeOutcome.ANALYZED, merge_commit='b' * 40, unlanded=(item,))

    for frozen in (item, probe):
        for field in dataclasses.fields(frozen):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(frozen, field.name, None)
