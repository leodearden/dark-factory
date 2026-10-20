"""build_chain refuses a queued link whose branch is stacked on a never-landed base (task 6132).

Chaining such a link would land the base's commits under the link's name, the
mis-attribution task 5618 removed from the sequential admission path.  The
chain is truncated there instead, so the link takes the sequential path, where
``classify_and_merge`` un-stacks it.  These tests drive public seams only
(``GitOps.stack_train_branches``, ``GitOps.unlanded_stack_base``,
``StackBaseLedger.base_of``, ``build_chain``) against real temp git repos.

Base fixture: main has shared.txt and other.txt.  After stacking, main
advances on other.txt, so a stacked base is not on main until a test lands it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT

from orchestrator.branch_stack import StackBaseLedger
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane.liveness import release_chain_build_lane
from orchestrator.merge_lane.types import ChainResult, QueuedBranch
from orchestrator.merge_lane.worker import MergeRequest, build_chain

# ---------------------------------------------------------------------------
# Fixtures (copied from test_merge_unstack_admission.py, per-file convention)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _git(repo, 'init', '-b', 'main')
    await _git(repo, 'config', 'user.email', 'test@test.com')
    await _git(repo, 'config', 'user.name', 'Test')
    await _commit(repo, {'shared.txt': 'v0\n', 'other.txt': 'v0\n'}, 'initial')


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_request(task_id: str, git_ops: GitOps, config: OrchestratorConfig) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=git_ops.worktree_base / task_id,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


async def _commit(cwd: Path, writes: dict[str, str], msg: str) -> str:
    """Write and commit exactly *writes*; a bare ``git add -A`` in the main
    checkout would stage the nested ``.worktrees/*`` checkouts as gitlinks."""
    for name, content in writes.items():
        (cwd / name).write_text(content)
    await _git(cwd, 'add', '--', *writes)
    await _git(cwd, 'commit', '-m', msg)
    return (await _git(cwd, 'rev-parse', 'HEAD')).strip()


async def _member(git_ops: GitOps, name: str, writes: dict[str, str]) -> None:
    """task/<name> from main at git_ops.worktree_base / <name>, one commit."""
    wt = (await git_ops.create_worktree(name)).path
    await _commit(wt, writes, f'{name}: write files')


async def _sha(git_ops: GitOps, ref: str) -> str:
    return (await _git(git_ops.project_root, 'rev-parse', '--verify', ref)).strip()


async def _is_ancestor(git_ops: GitOps, ancestor: str, descendant: str) -> bool:
    rc, _, _ = await _run(
        ['git', 'merge-base', '--is-ancestor', ancestor, descendant],
        cwd=git_ops.project_root,
    )
    assert rc in (0, 1), f'merge-base --is-ancestor {ancestor} {descendant} rc={rc}'
    return rc == 0


def _ledger(git_ops: GitOps) -> StackBaseLedger:
    return StackBaseLedger(git_ops.project_root, _run)


async def _stack_m_on_unlanded_p(git_ops: GitOps) -> str:
    """task/M stacked on task/P, which never lands; main then advances.  Returns main's sha."""
    await _member(git_ops, 'P', {'p_only.txt': 'p\n'})
    await _member(git_ops, 'M', {'m.txt': 'm\n'})
    result = await git_ops.stack_train_branches(['P', 'M'])
    assert result.survivors == ['P', 'M'], f'{result=}'
    return await _commit(git_ops.project_root, {'other.txt': 'main\n'}, 'main advances')


async def _land_p(git_ops: GitOps) -> str:
    await _git(git_ops.project_root, 'merge', '--no-ff', 'task/P', '-m', 'land P')
    return await _sha(git_ops, 'main')


async def _build(
    git_ops: GitOps, config: OrchestratorConfig, task_ids: list[str], head: str,
) -> tuple[ChainResult, list[MergeRequest]]:
    requests = [_make_request(tid, git_ops, config) for tid in task_ids]
    result = await build_chain(
        git_ops, requests, head, cap=len(requests), target_depth=len(requests),
    )
    return result, requests


# ---------------------------------------------------------------------------
# GitOps.unlanded_stack_base — the read-only predicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnlandedStackBase:
    async def test_unstacked_branch_has_no_unlanded_base(self, git_ops: GitOps) -> None:
        await _member(git_ops, 'A', {'a.txt': 'a\n'})
        assert await git_ops.unlanded_stack_base('task/A') is None

    async def test_branch_stacked_on_unlanded_predecessor_names_the_base(
        self, git_ops: GitOps,
    ) -> None:
        await _stack_m_on_unlanded_p(git_ops)
        p_tip = await _sha(git_ops, 'task/P')
        assert await git_ops.unlanded_stack_base('task/M') == p_tip

    async def test_base_that_landed_is_not_unlanded(self, git_ops: GitOps) -> None:
        await _stack_m_on_unlanded_p(git_ops)
        await _land_p(git_ops)
        assert await git_ops.unlanded_stack_base('task/M') is None

    async def test_reading_moves_nothing(self, git_ops: GitOps) -> None:
        await _stack_m_on_unlanded_p(git_ops)
        p_tip = await _sha(git_ops, 'task/P')
        m_tip = await _sha(git_ops, 'task/M')

        await git_ops.unlanded_stack_base('task/M')

        assert await _sha(git_ops, 'task/M') == m_tip
        assert await _ledger(git_ops).base_of('task/M') == p_tip


# ---------------------------------------------------------------------------
# build_chain — truncates at the stacked link
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
class TestBuildChainRefusesStackedLink:
    async def test_link_stacked_on_unlanded_base_truncates_the_chain(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        await _member(git_ops, 'A', {'a.txt': 'a\n'})
        head = await _stack_m_on_unlanded_p(git_ops)
        await _member(git_ops, 'C', {'c.txt': 'c\n'})

        result, requests = await _build(git_ops, config, ['A', 'M', 'C'], head)
        try:
            assert [tid for tid, _ in result.links] == ['A'], f'{result=}'
            assert result.truncated_at == 'M', f'{result=}'
            assert result.truncated_reason == 'stacked_on_unlanded_base', f'{result=}'
            assert not await _is_ancestor(git_ops, await _sha(git_ops, 'task/P'), result.tip)
            assert not await _is_ancestor(git_ops, await _sha(git_ops, 'task/C'), result.tip)
            assert not any(req.result.done() for req in requests)
        finally:
            await release_chain_build_lane(git_ops, result.lane, warm=result.lane_warm)

    async def test_stacked_first_link_yields_an_empty_chain_holding_no_lane(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        head = await _stack_m_on_unlanded_p(git_ops)

        result, _ = await _build(git_ops, config, ['M'], head)

        assert result.links == [], f'{result=}'
        assert result.tip == head, f'{result=}'
        assert result.truncated_at == 'M', f'{result=}'
        assert result.truncated_reason == 'stacked_on_unlanded_base', f'{result=}'
        assert result.lane is None, f'{result=}'
        assert result.lane_warm is False, f'{result=}'

    async def test_chain_builder_leaves_the_stacked_branch_for_the_sequential_path(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        head = await _stack_m_on_unlanded_p(git_ops)
        p_tip = await _sha(git_ops, 'task/P')
        m_tip = await _sha(git_ops, 'task/M')

        result, _ = await _build(git_ops, config, ['M'], head)
        await release_chain_build_lane(git_ops, result.lane, warm=result.lane_warm)

        assert await _sha(git_ops, 'task/M') == m_tip
        assert await _ledger(git_ops).base_of('task/M') == p_tip

    async def test_link_whose_stack_base_landed_still_chains(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        await _stack_m_on_unlanded_p(git_ops)
        head = await _land_p(git_ops)

        result, _ = await _build(git_ops, config, ['M'], head)
        try:
            assert [tid for tid, _ in result.links] == ['M'], f'{result=}'
            assert result.truncated_reason is None, f'{result=}'
        finally:
            await release_chain_build_lane(git_ops, result.lane, warm=result.lane_warm)
