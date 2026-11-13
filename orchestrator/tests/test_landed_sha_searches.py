"""The steward's rendered landed-sha searches agree with the merge lane's own verdicts.

`orchestrator/src/orchestrator/agents/landed_sha_searches.py::LandedShaSearches`
renders shell commands for the steward briefing. Each test here runs one of
those commands in a purpose-built real-git history and asserts it lands on the
commit the production authority picks in the same repo:
`orchestrator/src/orchestrator/git_ops.py::GitOps.find_merge_marker` for the
marker search, `GitOps.is_ancestor_checked` for the containment check,
`GitOps.find_task_citation_commit` for the citation walk. A
change to either rule then fails here instead of leaving the steward a stale
copy.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

import pytest
from _ladder_git import commit, git, git_env, new_repo
from shared.merge_subject import format_merge_subject

from orchestrator.agents.landed_sha_searches import (
    CitationGateUnavailable,
    LandedShaSearches,
    already_merged_step,
)
from orchestrator.config import GitConfig
from orchestrator.git_ops import CitationPatternInvalid, GitOps, task_citation_pattern

TASK_ID = '49'


def _run_rendered(root: Path, cmd: str) -> str:
    result = subprocess.run(
        ['bash', '-c', cmd], cwd=root, env=git_env(root), capture_output=True, text=True,
    )
    assert result.returncode == 0, f'rendered command failed: {cmd}\n{result.stderr}'
    return result.stdout.strip()


def _repo_on(tmp_path: Path, main_branch: str) -> Path:
    root = tmp_path / 'repo'
    new_repo(root)
    git(root, 'symbolic-ref', 'HEAD', f'refs/heads/{main_branch}')
    commit(root, 'init', 'init.txt')
    return root


def _commit_with_body(root: Path, subject: str, body: str, filename: str) -> str:
    (root / filename).write_text(f'{subject}\n')
    git(root, 'add', filename)
    git(root, 'commit', '-m', subject, '-m', body)
    return git(root, 'rev-parse', 'HEAD')


def _land_no_ff(root: Path, branch: str, main_branch: str, work: str = 'work') -> str:
    git(root, 'checkout', '-B', branch)
    commit(root, f'{work} on {branch}', f'{branch.replace("/", "-")}-{work}.txt')
    git(root, 'checkout', main_branch)
    git(root, 'merge', '--no-ff', '-m', format_merge_subject(branch, main_branch), branch)
    return git(root, 'rev-parse', 'HEAD')


_BRANCH_CONFIGS = [
    pytest.param(GitConfig(main_branch='main', branch_prefix='task/'), id='main-task'),
    pytest.param(GitConfig(main_branch='trunk', branch_prefix='feat/'), id='trunk-feat'),
]


@pytest.mark.parametrize('cfg', _BRANCH_CONFIGS)
def test_marker_search_agrees_with_find_merge_marker(tmp_path: Path, cfg: GitConfig) -> None:
    root = _repo_on(tmp_path, cfg.main_branch)
    branch = f'{cfg.branch_prefix}{TASK_ID}'
    subject = format_merge_subject(branch, cfg.main_branch)
    _land_no_ff(root, branch, cfg.main_branch)
    remerged = _land_no_ff(root, branch, cfg.main_branch, work='rework')
    _land_no_ff(root, f'{cfg.branch_prefix}{TASK_ID}0', cfg.main_branch)
    _commit_with_body(root, 'docs: quote a marker', subject, 'quote.txt')
    commit(root, f'{subject}\tsuffixed', 'tab-suffixed.txt')

    searches = LandedShaSearches.for_task(TASK_ID, cfg)
    oracle = asyncio.run(GitOps(cfg, root).find_merge_marker(branch, gate_on_existing_ref=False))

    assert _run_rendered(root, searches.marker_search) == remerged == oracle


def test_containment_agrees_with_is_ancestor(tmp_path: Path) -> None:
    cfg = GitConfig()
    root = _repo_on(tmp_path, cfg.main_branch)
    branch = f'{cfg.branch_prefix}{TASK_ID}'
    marker = _land_no_ff(root, branch, cfg.main_branch)
    searches = LandedShaSearches.for_task(TASK_ID, cfg)
    git_ops = GitOps(cfg, root)

    def verdicts() -> tuple[str, bool | None]:
        return (
            _run_rendered(root, searches.containment(marker)),
            asyncio.run(git_ops.is_ancestor_checked(branch, marker)),
        )

    assert verdicts() == ('containment rc=0', True)
    git(root, 'checkout', branch)
    commit(root, 'work after the landing', 'after.txt')
    git(root, 'checkout', cfg.main_branch)
    assert verdicts() == ('containment rc=1', False)


def test_ancestry_reports_three_outcomes(tmp_path: Path) -> None:
    cfg = GitConfig()
    root = _repo_on(tmp_path, cfg.main_branch)
    _land_no_ff(root, 'task/49', cfg.main_branch)
    git(root, 'branch', 'task/50')
    git(root, 'checkout', 'task/50')
    commit(root, 'work on task/50', 'fifty.txt')
    git(root, 'checkout', cfg.main_branch)

    merged = LandedShaSearches.for_task('49', cfg).ancestry
    unmerged = LandedShaSearches.for_task('50', cfg).ancestry

    assert _run_rendered(root, merged) == 'ancestry rc=0'
    assert _run_rendered(root, unmerged) == 'ancestry rc=1'
    git(root, 'branch', '-D', 'task/49')
    assert _run_rendered(root, merged) == 'ancestry rc=128'


_CITATION_CASES = [
    pytest.param(
        GitConfig(),
        [
            ('fix: thing (task 49 follow-up)', None),
            ('feat(49, 4862): widened scope', None),
            ('docs: note', 'fix(49): body only'),
        ],
        1,
        id='default-pattern',
    ),
    pytest.param(
        GitConfig(commit_citation_pattern=r'^\[{tid}\]'),
        [
            ('[49] custom', None),
            ('docs: other', '[49] body only'),
        ],
        0,
        id='project-pattern',
    ),
]


@pytest.mark.parametrize(('cfg', 'history', 'expected_index'), _CITATION_CASES)
def test_citation_walk_lists_the_commit_find_task_citation_commit_cites(
    tmp_path: Path,
    cfg: GitConfig,
    history: list[tuple[str, str | None]],
    expected_index: int,
) -> None:
    root = _repo_on(tmp_path, cfg.main_branch)
    shas = [
        commit(root, subject, f'c{i}.txt')
        if body is None
        else _commit_with_body(root, subject, body, f'c{i}.txt')
        for i, (subject, body) in enumerate(history)
    ]

    walk = LandedShaSearches.for_task(TASK_ID, cfg).citation_walk
    assert isinstance(walk, str)
    pattern = task_citation_pattern(
        TASK_ID, cfg.commit_citation_pattern,
        branch_prefix=cfg.branch_prefix, main_branch=cfg.main_branch,
    )
    assert pattern is not None
    cites = re.compile(pattern)
    rows = [line.partition(' ') for line in _run_rendered(root, walk).splitlines()]
    first_citing = next(sha for sha, _, subject in rows if cites.search(subject))
    oracle = asyncio.run(
        GitOps(cfg, root).find_task_citation_commit(
            TASK_ID, pattern_template=cfg.commit_citation_pattern,
        ),
    )

    assert first_citing == oracle == shas[expected_index]


def test_opted_out_project_renders_no_citation_walk() -> None:
    searches = LandedShaSearches.for_task(TASK_ID, GitConfig(commit_citation_pattern=''))

    assert searches.citation_walk is CitationGateUnavailable.OPTED_OUT


_UNRENDERABLE_TEMPLATES = [
    pytest.param(r'^\[{tid}\][0-9]{2}', id='ere-brace-quantifier'),
    pytest.param('{nope}', id='unknown-field'),
    pytest.param('({tid}', id='not-a-python-re'),
]


@pytest.mark.parametrize('template', _UNRENDERABLE_TEMPLATES)
def test_an_unrenderable_project_pattern_is_one_typed_error(template: str) -> None:
    with pytest.raises(CitationPatternInvalid):
        task_citation_pattern(TASK_ID, template, branch_prefix='task/', main_branch='main')


@pytest.mark.parametrize('template', _UNRENDERABLE_TEMPLATES)
def test_an_unrenderable_project_pattern_leaves_the_step_renderable(template: str) -> None:
    searches = LandedShaSearches.for_task(TASK_ID, GitConfig(commit_citation_pattern=template))

    assert searches.citation_walk is CitationGateUnavailable.PATTERN_INVALID
    assert 'commit_citation_pattern' in already_merged_step(searches)


_STEP_CONFIGS = [
    pytest.param(GitConfig(), True, id='defaults'),
    pytest.param(
        GitConfig(main_branch='trunk', branch_prefix='feat/', commit_citation_pattern=r'^\[{tid}\]'),
        True,
        id='trunk-feat-project-pattern',
    ),
    pytest.param(GitConfig(commit_citation_pattern=''), False, id='opted-out'),
]


@pytest.mark.parametrize(('cfg', 'renders_walk'), _STEP_CONFIGS)
def test_already_merged_step_carries_the_rendered_searches(
    cfg: GitConfig, renders_walk: bool,
) -> None:
    searches = LandedShaSearches.for_task(TASK_ID, cfg)
    step = already_merged_step(searches)

    assert isinstance(searches.citation_walk, str) == renders_walk
    assert searches.ancestry in step
    assert searches.marker_search in step
    assert searches.containment('<marker-sha>') in step
    if isinstance(searches.citation_walk, str):
        assert searches.citation_walk in step
