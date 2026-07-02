"""Tests for claim_warm_worktree / release_warm_worktree escalation MCP verbs.

PRD β (task 2011) — thin @mcp.tool() closures over the in-process ``harness``
handle that expose task α's interactive-warm-worktree primitive
(``GitOps.create_interactive_worktree``, task 2010) to out-of-process
interactive callers (/do, /warm, the integration gate).

Isolated from test_server.py because these are real-git integration tests
(worktree creation/removal, branch pruning) incompatible with test_server.py's
in-memory stubs — mirrors test_merge_status_git_authority.py's split.

Shared patterns from the existing test suite:
- Cross-package orchestrator import guard (mirrors test_server.py:31-55 /
  test_merge_status_git_authority.py:31-43)
- ``_call_tool`` helper — FastMCP async unit-test pattern
  (``tool = await server.get_tool(name); await tool.fn(...)``), mirrors
  test_server.py's ``_call_tool`` (line 3661).
- Real-repo fixtures (``_init_repo``/``_add_seed_script``/``_run``,
  ``GitConfig()``+``GitOps(config, repo)``) copied from
  orchestrator/tests/test_interactive_worktree.py.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports.
# Guarded so the rest of the file is still collected when the orchestrator
# package is absent (e.g. in an escalation-only install).
# Mirrors test_server.py lines 31-55 / test_merge_status_git_authority.py.
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import GitConfig  # type: ignore[reportMissingImports]
    from orchestrator.git_ops import (  # type: ignore[reportMissingImports]
        GitOps,
        InteractiveWorktreeLimitError,
        _run,
    )
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    # Satisfy pyright's definite-assignment check — see test_server.py:44-55.
    GitConfig: Any = None
    GitOps: Any = None
    InteractiveWorktreeLimitError: Any = None
    _run: Any = None

# ---------------------------------------------------------------------------
# Repo fixture + seed-stub helper — copied from
# orchestrator/tests/test_interactive_worktree.py (_init_repo/_add_seed_script/
# _commit_file), which this file's fixtures mirror exactly so a committed
# seed-warm-lane.sh stub is available for warm=True assertions.
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _add_seed_script(repo: Path, *, exit_code: int = 0) -> None:
    """Commit a seed-warm-lane.sh stub into repo/scripts/.

    On exit_code == 0: creates <lane>/target/seeded.bin (orchestration-observable
    seededness). On non-zero: exits with that code immediately (no target/
    created) — unused here but kept for parity with the α fixture.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    seed = scripts_dir / 'seed-warm-lane.sh'
    if exit_code == 0:
        seed.write_text(
            '#!/usr/bin/env bash\n'
            'mkdir -p "$2/target"\n'
            'echo "seeded" > "$2/target/seeded.bin"\n'
        )
    else:
        seed.write_text(
            f'#!/usr/bin/env bash\necho "seed failure" >&2\nexit {exit_code}\n'
        )
    seed.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add seed-warm-lane.sh stub'], cwd=repo)


async def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    """Write+commit a file on the repo's current branch; return the new commit SHA."""
    (repo / name).write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', message], cwd=repo)
    rc, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse HEAD failed after committing {name!r}'
    return sha.strip()


@pytest.fixture
def warm_repo(tmp_path: Path) -> Path:
    """Temp git repo with an initial commit + committed seed-warm-lane.sh stub."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    asyncio.run(_add_seed_script(repo))
    return repo


@pytest.fixture
def git_ops(warm_repo: Path) -> GitOps:  # type: ignore[reportInvalidTypeForm]
    return GitOps(GitConfig(), warm_repo)


@pytest.fixture
def harness(git_ops: GitOps) -> types.SimpleNamespace:  # type: ignore[reportInvalidTypeForm]
    return types.SimpleNamespace(git_ops=git_ops)


@pytest.fixture
def server(tmp_path: Path, harness: types.SimpleNamespace) -> Any:
    return create_server(EscalationQueue(tmp_path / 'esc'), harness=harness)


async def _call_tool(server: Any, name: str, **kwargs: Any) -> Any:
    """Invoke an async MCP tool by name (mirrors test_server.py's _call_tool)."""
    tool = await server.get_tool(name)
    return await tool.fn(**kwargs)


# ---------------------------------------------------------------------------
# step-1 (RED) / step-2 (GREEN): claim_warm_worktree happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestClaimWarmWorktreeHappyPath:
    """claim_warm_worktree happy path — maps InteractiveWorktreeInfo 1:1."""

    async def test_claim_creates_warm_worktree(
        self, warm_repo: Path, git_ops: GitOps, server: Any,  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """Fresh claim: {path, branch, warm, base_ref} match the α contract."""
        rc, main_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=warm_repo)
        assert rc == 0, 'setup: rev-parse main failed'
        main_sha = main_sha.strip()

        res = await _call_tool(
            server, 'claim_warm_worktree', slug='demo', project_root=str(warm_repo),
        )

        expected_path = git_ops.worktree_base / '_iact-demo'
        assert res == {
            'path': str(expected_path),
            'branch': 'task/demo',
            'warm': True,
            'base_ref': main_sha,
        }, f'unexpected claim_warm_worktree result: {res}'
        assert isinstance(res['path'], str), 'path must be a str, not a Path'

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=warm_repo,
        )
        assert rc_list == 0
        assert '_iact-demo' in wt_list, f'expected _iact-demo registered, got: {wt_list}'

    async def test_claim_forwards_start_ref(self, warm_repo: Path, server: Any) -> None:
        """start_ref is forwarded verbatim — base_ref reflects the pinned commit."""
        second_sha = await _commit_file(warm_repo, 'second.txt', 'x', 'second commit')

        res = await _call_tool(
            server, 'claim_warm_worktree',
            slug='demo2', project_root=str(warm_repo), start_ref=second_sha,
        )

        assert res['base_ref'] == second_sha, (
            f'expected base_ref == start_ref {second_sha!r}, got {res["base_ref"]!r}'
        )


# ---------------------------------------------------------------------------
# step-3 (RED) / step-4 (GREEN): claim guards + error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestClaimWarmWorktreeGuards:
    """Standalone guard (regression), project_root mismatch, exception mapping."""

    async def test_a_standalone_returns_error(self, tmp_path: Path) -> None:
        """(a) No harness wired → {'error'} — already green from step-2; regression."""
        server = create_server(EscalationQueue(tmp_path / 'esc'))  # no harness
        result = await _call_tool(
            server, 'claim_warm_worktree', slug='x', project_root=str(tmp_path),
        )
        assert 'error' in result, f'expected an error key, got: {result}'

    async def test_b_project_root_mismatch_creates_nothing(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """(b) project_root mismatch → {'error'} mentioning mismatch, nothing created."""
        other = warm_repo.parent / 'not-the-repo'
        other.mkdir()

        result = await _call_tool(
            server, 'claim_warm_worktree',
            slug='mismatch-slug', project_root=str(other),
        )

        assert 'error' in result, f'expected an error key, got: {result}'
        assert 'mismatch' in result['error'].lower(), (
            f"expected 'mismatch' in error message, got: {result['error']!r}"
        )

        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=warm_repo,
        )
        assert '_iact-' not in wt_list, (
            f'no worktree should have been created on mismatch, got: {wt_list}'
        )

    async def test_c_maps_interactive_worktree_limit_error(self, tmp_path: Path) -> None:
        """(c) create_interactive_worktree raising the cap error → {'error', reason}."""
        root = tmp_path / 'root'
        root.mkdir()

        async def _raise_limit(slug: str, *, start_ref: str | None = None) -> Any:
            raise InteractiveWorktreeLimitError('at cap: 2/2 _iact-* worktrees')

        fake_git_ops = types.SimpleNamespace(
            project_root=root, create_interactive_worktree=_raise_limit,
        )
        stub_harness = types.SimpleNamespace(git_ops=fake_git_ops)
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=stub_harness)

        result = await _call_tool(
            server, 'claim_warm_worktree', slug='x', project_root=str(root),
        )

        assert 'error' in result, f'expected an error key, got: {result}'
        assert result.get('reason') == 'interactive_worktree_limit', (
            f'expected reason=interactive_worktree_limit, got: {result}'
        )

    async def test_d_maps_value_error(self, tmp_path: Path) -> None:
        """(d) create_interactive_worktree raising ValueError (bad slug) → {'error', reason}."""
        root = tmp_path / 'root'
        root.mkdir()

        async def _raise_value(slug: str, *, start_ref: str | None = None) -> Any:
            raise ValueError('bad slug')

        fake_git_ops = types.SimpleNamespace(
            project_root=root, create_interactive_worktree=_raise_value,
        )
        stub_harness = types.SimpleNamespace(git_ops=fake_git_ops)
        server = create_server(EscalationQueue(tmp_path / 'esc'), harness=stub_harness)

        result = await _call_tool(
            server, 'claim_warm_worktree', slug='x', project_root=str(root),
        )

        assert 'error' in result, f'expected an error key, got: {result}'
        assert result.get('reason') == 'invalid_slug', (
            f'expected reason=invalid_slug, got: {result}'
        )


# ---------------------------------------------------------------------------
# step-5 (RED) / step-6 (GREEN): release_warm_worktree by path — removal,
# idempotency, standalone + project_root guards.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestReleaseWarmWorktreeRemoval:
    """release_warm_worktree by path: removes, is idempotent, guards standalone/mismatch."""

    async def test_release_by_path_removes_worktree(
        self, warm_repo: Path, server: Any,
    ) -> None:
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='rel1', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'

        res = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(warm_repo),
        )

        assert res['removed'] is True, f'expected removed=True, got: {res}'
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=warm_repo,
        )
        assert rc_list == 0
        assert '_iact-rel1' not in wt_list, (
            f'expected _iact-rel1 gone from worktree list, got: {wt_list}'
        )

    async def test_release_by_path_is_idempotent(
        self, warm_repo: Path, server: Any,
    ) -> None:
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='rel1b', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        first = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(warm_repo),
        )
        assert first['removed'] is True, f'setup: first release failed: {first}'

        second = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(warm_repo),
        )

        assert second['removed'] is False, (
            f'expected removed=False on already-gone path, got: {second}'
        )
        # Regression: an already-gone *absolute path* must still resolve to
        # its canonical path/branch (path-mode), not be treated as one giant
        # branch-mode slug (which would previously yield path/branch values
        # derived from the whole path string).
        assert second['path'] == claim['path'], (
            f"expected path to still resolve to the claimed worktree path, "
            f"got: {second['path']!r} vs claimed {claim['path']!r}"
        )
        assert second['branch'] == 'task/rel1b', (
            f"expected branch derived from the path's basename, got: {second['branch']!r}"
        )

    async def test_release_failure_surfaces_stderr_detail(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """A genuine (non-idempotent) removal failure carries a 'detail' with git's stderr."""
        never_claimed = warm_repo / '_iact-never-claimed'

        result = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=str(never_claimed), project_root=str(warm_repo),
        )

        assert result['removed'] is False, f'expected removed=False, got: {result}'
        assert result.get('detail'), (
            f'expected a non-empty detail explaining the removal failure, got: {result}'
        )

    async def test_release_standalone_returns_error(self, tmp_path: Path) -> None:
        """(i) No harness wired → {'error'}."""
        server = create_server(EscalationQueue(tmp_path / 'esc'))  # no harness
        result = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=str(tmp_path / 'nonexistent'), project_root=str(tmp_path),
        )
        assert 'error' in result, f'expected an error key, got: {result}'

    async def test_release_project_root_mismatch_removes_nothing(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """(ii) project_root mismatch → {'error'} and nothing removed."""
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='rel2', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        other = warm_repo.parent / 'not-the-repo'
        other.mkdir()

        result = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(other),
        )

        assert 'error' in result, f'expected an error key, got: {result}'
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=warm_repo,
        )
        assert '_iact-rel2' in wt_list, (
            f'expected _iact-rel2 still registered after mismatch, got: {wt_list}'
        )


# ---------------------------------------------------------------------------
# step-7 (RED) / step-8 (GREEN): release prunes a merged branch, keeps an
# unmerged one, and accepts branch-name input.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestReleaseWarmWorktreeBranchPrune:
    """Prune-if-merged after removal; branch-mode path_or_branch resolution."""

    async def test_a_merged_branch_is_pruned(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """(a) task/mrg tip == main tip (already an ancestor) → pruned + gone."""
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='mrg', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'

        res = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(warm_repo),
        )

        assert res['branch_pruned'] is True, f'expected branch_pruned=True, got: {res}'
        rc, out, _ = await _run(['git', 'branch', '--list', 'task/mrg'], cwd=warm_repo)
        assert rc == 0
        assert out.strip() == '', f'expected task/mrg branch gone, got: {out!r}'

    async def test_b_unmerged_branch_is_kept(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """(b) task/wip has a commit beyond main → branch_pruned False, kept."""
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='wip', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'
        await _commit_file(Path(claim['path']), 'wip.txt', 'x', 'wip commit')

        res = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch=claim['path'], project_root=str(warm_repo),
        )

        assert res['branch_pruned'] is False, f'expected branch_pruned=False, got: {res}'
        rc, out, _ = await _run(['git', 'branch', '--list', 'task/wip'], cwd=warm_repo)
        assert rc == 0
        assert 'task/wip' in out, f'expected task/wip branch kept, got: {out!r}'

    async def test_c_release_by_branch_name(
        self, warm_repo: Path, server: Any,
    ) -> None:
        """(c) path_or_branch='task/bybr' (branch-mode) resolves + removes."""
        claim = await _call_tool(
            server, 'claim_warm_worktree', slug='bybr', project_root=str(warm_repo),
        )
        assert 'error' not in claim, f'setup: claim failed: {claim}'

        res = await _call_tool(
            server, 'release_warm_worktree',
            path_or_branch='task/bybr', project_root=str(warm_repo),
        )

        assert res['removed'] is True, f'expected removed=True, got: {res}'
        rc_list, wt_list, _ = await _run(
            ['git', 'worktree', 'list', '--porcelain'], cwd=warm_repo,
        )
        assert rc_list == 0
        assert '_iact-bybr' not in wt_list, (
            f'expected _iact-bybr gone from worktree list, got: {wt_list}'
        )
