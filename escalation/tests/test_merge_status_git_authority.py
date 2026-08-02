"""Tests for merge_status Tier-3.5 git-authority resolution.

Isolated from test_server.py because:
- Real-git integration tests need subprocess/worktree fixtures incompatible
  with test_server.py's in-memory stubs.
- Isolating keeps the slower integration test grouped and avoids bloating
  the 3400-line test_server.py.

Shared patterns from the existing test suite:
- Cross-package import guard (mirrors test_server.py lines 30-54)
- _call_merge_status helper (mirrors test_server.py line 2825)
- _stub_git_ops: returns SimpleNamespace with AsyncMock methods for unit tests
- Real-git fixtures (git_repo/_init_repo/orch_config/git_ops) modeled on
  test_workflow_status_on_resume.py:47-80 and test_git_ops.py
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports.
# Mirrors test_server.py lines 30-54 exactly.
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import (  # type: ignore[reportMissingImports]
        GitConfig,
        OrchestratorConfig,
    )
    from orchestrator.git_ops import GitOps, _run  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    GitConfig: Any = None  # type: ignore[assignment,misc]
    OrchestratorConfig: Any = None  # type: ignore[assignment,misc]
    GitOps: Any = None  # type: ignore[assignment,misc]
    _run: Any = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_merge_status(server, **kwargs) -> dict:
    """Invoke the merge_status MCP tool (async tool)."""
    tool = await server.get_tool('merge_status')
    return await tool.fn(**kwargs)


def _stub_git_ops(**overrides) -> types.SimpleNamespace:
    """Return a SimpleNamespace stub for git_ops unit tests.

    Pass keyword args matching method names; each value must be an async
    callable (e.g. AsyncMock).  Unspecified methods default to AsyncMock()
    returning None / False so callers only specify the values they care about.

    Example::

        stub = _stub_git_ops(
            resolve_branch_sha=AsyncMock(return_value='a' * 40),
            is_ancestor=AsyncMock(return_value=True),
        )
        # stub.find_merge_marker is an AsyncMock(return_value=None) by default

    ``find_task_citation_commit`` / ``commit_effect_present_in_main`` are the
    two sub-methods ``validate_landing_evidence`` calls (task 3103's citation
    gate).  They MUST be present by default: without them the SimpleNamespace
    raises AttributeError inside the Tier-3.5 fire-safe wrapper, which
    swallows it and degrades every ancestor/marker-arm test to a misleading
    ``unknown`` that looks like a passing guard.  The defaults are the
    conservative pair — no citation found (so the gate rejects unless a test
    opts in) and effect present (so the FIX 1' guard is not what a test
    accidentally trips over).
    """
    stub = types.SimpleNamespace(
        resolve_branch_sha=AsyncMock(return_value=None),
        is_ancestor=AsyncMock(return_value=False),
        find_merge_marker=AsyncMock(return_value=None),
        find_task_citation_commit=AsyncMock(return_value=None),
        commit_effect_present_in_main=AsyncMock(return_value=True),
    )
    for name, fn in overrides.items():
        setattr(stub, name, fn)
    return stub


def _make_config(
    tmp_path: Path, *, commit_citation_pattern: str | None = None,
) -> OrchestratorConfig:  # type: ignore[reportInvalidTypeForm]
    """Build the standard unit-test OrchestratorConfig rooted at *tmp_path*.

    Wraps the GitConfig literal this module repeated inline.  ``None`` (the
    default) means "use the built-in DEFAULT_COMMIT_CITATION_PATTERN" and is
    NOT the opt-out; ``''`` is the documented per-project opt-out that
    disables the citation check entirely (config.py ``commit_citation_pattern``).
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            commit_citation_pattern=commit_citation_pattern,
        ),
    )


def _stub_harness(
    git_ops, *, metadata: dict[str, Any] | None = None, get_task_raises: bool = False,
) -> types.SimpleNamespace:
    """Return the standard merge_status harness stub wired to *git_ops*.

    By default the stub has NO ``scheduler`` attribute at all — which is the
    fail-soft path task 3103's ``_git_authority_task_metadata`` helper is
    built for (metadata unavailable → skip the degeneracy check → still apply
    the citation gate), and the shape every pre-existing test in this module
    already uses.

    Pass ``metadata=`` to attach a ``.scheduler`` whose ``get_task`` resolves
    to ``{'metadata': metadata}``, or ``get_task_raises=True`` to attach one
    whose ``get_task`` raises — the scheduler-fault path.
    """
    harness = types.SimpleNamespace(
        _merge_worker=None,
        _terminal_retention=None,
        git_ops=git_ops,
    )
    if get_task_raises:
        harness.scheduler = types.SimpleNamespace(
            get_task=AsyncMock(side_effect=RuntimeError('scheduler unreachable')),
        )
    elif metadata is not None:
        harness.scheduler = types.SimpleNamespace(
            get_task=AsyncMock(return_value={'metadata': metadata}),
        )
    return harness


# ---------------------------------------------------------------------------
# Real-git fixtures — modeled on test_workflow_status_on_resume.py:47-80
# and test_git_ops.py.  Only used by integration tests guarded with
# @pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, ...).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def orch_config(git_repo: Path) -> OrchestratorConfig:  # type: ignore[reportInvalidTypeForm]
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(orch_config: OrchestratorConfig) -> GitOps:  # type: ignore[reportInvalidTypeForm]
    return GitOps(orch_config.git, orch_config.project_root)


# ---------------------------------------------------------------------------
# Unit tests — in-memory stubs, no real git required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatusGitAuthority:
    """Unit tests for the git-authority Tier-3.5 inside merge_status."""

    # ── step-1: is_ancestor live-branch path ─────────────────────────────────

    async def test_live_branch_on_main_returns_done(self, tmp_path: Path) -> None:
        """When tip is not None and is_ancestor(tip, 'main') → state='done'/found_on_main.

        Verifies:
        - state == 'done', kind == 'found_on_main', generation == 1
        - merge_sha == the citation commit discovered on main (task 3103; this
          assertion used to pin ``merge_sha == tip``, the documented
          ``_found_on_main_response`` wart that the citation gate retires)
        - the (tip, 'main') ancestry call is made
        - find_merge_marker was NOT called (cheaper-common-path ordering: skip
          the find_merge_marker scan when the branch ref is still live)
        """
        tip = 'a' * 40
        main_sha = 'm' * 40  # distinct from tip so the tip != main_tip guard passes
        citation = 'c' * 40
        rsb = AsyncMock(side_effect=lambda b: tip if b == 'task/123' else main_sha)
        ia = AsyncMock(return_value=True)
        fmm = AsyncMock(return_value=None)
        stub_git = _stub_git_ops(
            resolve_branch_sha=rsb,
            is_ancestor=ia,
            find_merge_marker=fmm,
            find_task_citation_commit=AsyncMock(return_value=citation),
            commit_effect_present_in_main=AsyncMock(return_value=True),
        )
        stub_harness = _stub_harness(stub_git)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = _make_config(tmp_path)
        # NO event_store — durable tiers miss, so Tier-3.5 must fire
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='123')

        assert result.get('state') == 'done', f'Expected state=done, got: {result}'
        assert result.get('kind') == 'found_on_main', f'Expected kind=found_on_main, got: {result}'
        assert result.get('merge_sha') == citation, (
            f'Expected merge_sha={citation!r} (the citation commit on main), got: {result}'
        )
        assert result.get('generation') == 1, f'Expected generation=1, got: {result}'

        # Verify routing: the (tip, 'main') ancestry call is made.  Not
        # assert_called_ONCE_with: validate_landing_evidence legitimately makes
        # a second is_ancestor(citation, branch) call to classify the
        # effect-present anchor.
        assert (tip, 'main') in [c.args for c in ia.call_args_list], (
            f'Expected an is_ancestor(tip, main) call, got: {ia.call_args_list}'
        )
        # find_merge_marker must NOT be called when tip is present
        fmm.assert_not_called()

    # ── step-3: deleted-branch find_merge_marker path (canonical 4352 shape) ──

    async def test_deleted_branch_find_merge_marker_returns_done(
        self, tmp_path: Path
    ) -> None:
        """When branch ref is gone (tip=None), find_merge_marker finds the merge commit.

        This is the canonical 4352 lost-record shape: task merged to main,
        branch+worktree deleted, retention ring/event store record lost —
        but the merge commit is findable on main via git log.

        Verifies:
        - state == 'done', kind == 'found_on_main', merge_sha == marker, generation == 1
        - find_merge_marker was called with full ref 'task/456'
        - is_ancestor is NOT called (tip is None, cheaper-common-path ordering)
        """
        marker = 'b' * 40
        rsb = AsyncMock(return_value=None)   # branch ref gone
        ia = AsyncMock(return_value=False)
        fmm = AsyncMock(return_value=marker)
        stub_git = _stub_git_ops(
            resolve_branch_sha=rsb,
            is_ancestor=ia,
            find_merge_marker=fmm,
        )
        stub_harness = _stub_harness(stub_git)
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = _make_config(tmp_path)
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='456')

        assert result.get('state') == 'done', f'Expected state=done, got: {result}'
        assert result.get('kind') == 'found_on_main', f'Expected kind=found_on_main, got: {result}'
        assert result.get('merge_sha') == marker, f'Expected merge_sha={marker!r}, got: {result}'
        assert result.get('generation') == 1, f'Expected generation=1, got: {result}'

        # find_merge_marker called with full ref 'task/456'
        fmm.assert_called_once_with('task/456')
        # is_ancestor must NOT be called when tip is None
        ia.assert_not_called()

    # ── step-5: fire-safety + negatives + guards ──────────────────────────────

    async def _make_server_with_git_ops(
        self, tmp_path: Path, stub_git
    ) -> Any:
        """Helper: create a server with git_ops wired, no event_store."""
        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = _stub_harness(stub_git)
        config = _make_config(tmp_path)
        return create_server(esc_queue, harness=stub_harness, orch_config=config)

    async def test_fire_safe_resolve_branch_sha_raises_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(a) fire-safe: resolve_branch_sha raising → state='unknown', no raise."""
        rsb = AsyncMock(side_effect=RuntimeError('git exploded'))
        stub_git = _stub_git_ops(resolve_branch_sha=rsb)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T')

        assert result.get('state') == 'unknown', f'Expected unknown, got: {result}'
        assert 'hint' in result, f'Expected hint key in unknown response: {result}'

    async def test_fire_safe_is_ancestor_raises_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(a) fire-safe: is_ancestor raising → state='unknown', no raise."""
        tip = 'c' * 40
        rsb = AsyncMock(return_value=tip)
        ia = AsyncMock(side_effect=RuntimeError('git subprocess died'))
        stub_git = _stub_git_ops(resolve_branch_sha=rsb, is_ancestor=ia)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T')

        assert result.get('state') == 'unknown', f'Expected unknown, got: {result}'
        assert 'hint' in result, f'Expected hint key in unknown response: {result}'

    async def test_genuine_miss_branch_present_not_on_main_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(b) genuine miss: tip present, is_ancestor=False, no marker → unknown.

        A branch that exists but hasn't landed on main yet must NOT
        false-positive as 'done'.
        """
        tip = 'd' * 40
        rsb = AsyncMock(return_value=tip)
        ia = AsyncMock(return_value=False)
        fmm = AsyncMock(return_value=None)
        stub_git = _stub_git_ops(resolve_branch_sha=rsb, is_ancestor=ia,
                                  find_merge_marker=fmm)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T-pending')

        assert result.get('state') == 'unknown', (
            f'Branch not on main must stay unknown, got: {result}'
        )
        # find_merge_marker must NOT be called when tip is present (cheaper path)
        fmm.assert_not_called()

    async def test_guard_orch_config_absent_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(c) guard: git_ops present but no orch_config → straight to honest unknown."""
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(return_value='e' * 40),
            is_ancestor=AsyncMock(return_value=True),
        )
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=stub_git,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        # No orch_config passed — Tier-3.5 must be skipped
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, task_id='T-noconfig')

        assert result.get('state') == 'unknown', (
            f'No orch_config must yield unknown, got: {result}'
        )

    async def test_guard_git_ops_absent_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(d) guard: harness without git_ops attr → straight to honest unknown."""
        # Harness with NO git_ops attribute at all
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='T-nogit')

        assert result.get('state') == 'unknown', (
            f'No git_ops must yield unknown, got: {result}'
        )

    # ── suggestion-1: branch-at-main-HEAD no-op guard ────────────────────────

    async def test_branch_at_main_head_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """Branch tip == main tip (no extra commits) must return unknown, not done.

        A commit is its own ancestor, so is_ancestor(tip, main) would return
        True even when the branch has no extra commits and nothing has been
        merged — a false-positive 'done'.

        The tip != main_tip guard prevents this: when the branch sits at
        exactly main's HEAD, Tier-3.5 falls through to the honest Tier-4
        unknown.  is_ancestor must NOT be called (Python short-circuit).
        """
        tip = 'f' * 40  # branch and main both at the same SHA
        rsb = AsyncMock(return_value=tip)   # returns same SHA for ALL resolve calls
        ia = AsyncMock(return_value=True)   # would satisfy is_ancestor if called
        fmm = AsyncMock(return_value=None)
        stub_git = _stub_git_ops(
            resolve_branch_sha=rsb,
            is_ancestor=ia,
            find_merge_marker=fmm,
        )
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='same-as-main')

        assert result.get('state') == 'unknown', (
            f'Branch at main HEAD must return unknown (not false-positive done): {result}'
        )
        # is_ancestor must NOT be called — tip==main_tip guard short-circuits first
        ia.assert_not_called()
        # find_merge_marker must NOT be called — tip is not None so deleted-branch
        # path is not entered
        fmm.assert_not_called()

    # ── task 3103: ancestor-arm degeneracy guard ─────────────────────────────

    async def test_degenerate_ancestor_branch_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """A branch parked at an OLD main commit must return unknown, not done.

        The live 3024/3031 shape (53 of the 64 reify census candidates): a
        warm-lane CoW seed that faulted, or a task that never committed, sits
        at a commit that IS an ancestor of main and is NOT main's tip — so
        both conjuncts of the pre-existing `tip != main_tip and is_ancestor`
        guard pass and the tier answers a confident `done` against a commit
        containing none of the task's work.

        branch_tip_sha == branch_base_sha proves zero commits were ever
        pushed, so the honest answer is Tier-4 unknown.
        """
        tip = 'a' * 40
        main_sha = 'm' * 40
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/3024' else main_sha
            ),
            is_ancestor=AsyncMock(return_value=True),
        )
        # branch_base_sha == tip → the branch never advanced past creation
        stub_harness = _stub_harness(stub_git, metadata={'branch_base_sha': tip})
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue, harness=stub_harness, orch_config=_make_config(tmp_path),
        )

        result = await _call_merge_status(server, task_id='3024')

        assert result.get('state') == 'unknown', (
            f'Degenerate branch must NOT resolve as done, got: {result}'
        )
        assert 'hint' in result, f'Expected hint key in unknown response: {result}'
        # No terminal-shaped keys may leak into the honest unknown
        assert 'merge_sha' not in result, f'merge_sha leaked into unknown: {result}'
        assert 'kind' not in result, f'kind leaked into unknown: {result}'

    async def test_degenerate_branch_with_citation_on_main_still_unknown(
        self, tmp_path: Path
    ) -> None:
        """Ordering pin: the degeneracy guard runs BEFORE the citation gate.

        The reify-5493 case that triggered the investigation: a degenerate
        branch whose task nonetheless HAS a citing commit on main (someone
        landed a `docs(5493): ...` commit directly).  A citation-first — or
        citation-only — design accepts here and still answers a wrong `done`.

        Asserting find_task_citation_commit is never called is the only
        assertion that would catch a refactor reordering the two gates.
        """
        tip = 'a' * 40
        main_sha = 'm' * 40
        citation = 'c' * 40
        ftcc = AsyncMock(return_value=citation)
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/5493' else main_sha
            ),
            is_ancestor=AsyncMock(return_value=True),
            find_task_citation_commit=ftcc,
            commit_effect_present_in_main=AsyncMock(return_value=True),
        )
        stub_harness = _stub_harness(stub_git, metadata={'branch_base_sha': tip})
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue, harness=stub_harness, orch_config=_make_config(tmp_path),
        )

        result = await _call_merge_status(server, task_id='5493')

        assert result.get('state') == 'unknown', (
            f'Degenerate branch must stay unknown even with a citation on '
            f'main, got: {result}'
        )
        ftcc.assert_not_called()

    async def test_degenerate_guard_consults_task_metadata_by_bare_id(
        self, tmp_path: Path
    ) -> None:
        """The metadata lookup uses the BARE task id, not the branch ref.

        Entered via the already-prefixed ``branch='task/3024'`` form (the
        shape the live repro used), the bare id must be derived by stripping
        the configured prefix — re-prefixing would look up a task that does
        not exist and silently lose the guard.
        """
        tip = 'a' * 40
        main_sha = 'm' * 40
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == 'task/3024' else main_sha
            ),
            is_ancestor=AsyncMock(return_value=True),
        )
        stub_harness = _stub_harness(stub_git, metadata={'branch_base_sha': tip})
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue, harness=stub_harness, orch_config=_make_config(tmp_path),
        )

        result = await _call_merge_status(server, branch='task/3024')

        assert result.get('state') == 'unknown', (
            f'Degenerate branch via branch= form must be unknown, got: {result}'
        )
        stub_harness.scheduler.get_task.assert_awaited_once_with('3024')

    # ── task 3103: ancestor-arm citation gate ────────────────────────────────

    def _ancestor_arm_server(
        self,
        tmp_path: Path,
        tid: str,
        *,
        tip: str,
        citation: str | None = None,
        effect_present: bool = True,
        metadata: dict[str, Any] | None = None,
        get_task_raises: bool = False,
        commit_citation_pattern: str | None = None,
    ) -> tuple[Any, types.SimpleNamespace]:
        """Build a server whose branch is a NON-degenerate ancestor of main.

        Returns ``(server, stub_git)`` so callers can assert on the stubs.
        """
        main_sha = 'm' * 40
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: tip if b == f'task/{tid}' else main_sha
            ),
            is_ancestor=AsyncMock(return_value=True),
            find_task_citation_commit=AsyncMock(return_value=citation),
            commit_effect_present_in_main=AsyncMock(return_value=effect_present),
        )
        stub_harness = _stub_harness(
            stub_git, metadata=metadata, get_task_raises=get_task_raises,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue,
            harness=stub_harness,
            orch_config=_make_config(
                tmp_path, commit_citation_pattern=commit_citation_pattern,
            ),
        )
        return server, stub_git

    async def test_reseeded_ancestor_branch_without_citation_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """A re-seeded branch with no citing commit on main must be unknown.

        The 11-of-64 class the degeneracy predicate MISSES: a warm lane
        re-seeded the branch ref to a NEWER main commit after creation, so
        ``tip != branch_base_sha`` and the branch reads as non-degenerate —
        yet it still carries zero of the task's own work.  Only the citation
        gate catches this, which is why neither guard ships alone.
        """
        tip = 'a' * 40
        server, stub_git = self._ancestor_arm_server(
            tmp_path, '3031', tip=tip, citation=None,
            metadata={'branch_base_sha': 'b' * 40},   # != tip → non-degenerate
        )

        result = await _call_merge_status(server, task_id='3031')

        assert result.get('state') == 'unknown', (
            f'Uncited ancestor branch must be unknown, got: {result}'
        )
        stub_git.find_task_citation_commit.assert_awaited_once()

    async def test_cited_ancestor_branch_returns_done_with_evidence_sha(
        self, tmp_path: Path
    ) -> None:
        """TRUE POSITIVE: a cited, effect-present landing still resolves done.

        merge_sha must be the CITATION commit found on main, not the branch
        tip — this pins the fix to the documented ``_found_on_main_response``
        wart (for a --no-ff merge the branch tip is a different commit from
        the one on main).
        """
        tip = 'a' * 40
        citation = 'c' * 40
        server, _ = self._ancestor_arm_server(
            tmp_path, '900', tip=tip, citation=citation, effect_present=True,
            metadata={'branch_base_sha': 'b' * 40},
        )

        result = await _call_merge_status(server, task_id='900')

        assert result.get('state') == 'done', f'Expected done, got: {result}'
        assert result.get('kind') == 'found_on_main', f'Expected found_on_main: {result}'
        assert result.get('merge_sha') == citation, (
            f'merge_sha must be the citation commit on main ({citation!r}), '
            f'not the branch tip ({tip!r}); got: {result.get("merge_sha")!r}'
        )

    async def test_cited_ancestor_branch_effect_absent_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """FIX 1' (task-1175 shape): a reverted landing must not read as done."""
        server, _ = self._ancestor_arm_server(
            tmp_path, '901', tip='a' * 40, citation='c' * 40, effect_present=False,
            metadata={'branch_base_sha': 'b' * 40},
        )

        result = await _call_merge_status(server, task_id='901')

        assert result.get('state') == 'unknown', (
            f'A citation whose effect is absent at main HEAD must be unknown, '
            f'got: {result}'
        )

    async def test_absent_branch_base_sha_without_citation_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """Metadata {} fails OPEN out of degeneracy — the citation gate decides."""
        server, _ = self._ancestor_arm_server(
            tmp_path, '902', tip='a' * 40, citation=None, metadata={},
        )

        result = await _call_merge_status(server, task_id='902')

        assert result.get('state') == 'unknown', (
            f'No branch_base_sha and no citation must be unknown, got: {result}'
        )

    async def test_absent_branch_base_sha_with_citation_returns_done(
        self, tmp_path: Path
    ) -> None:
        """Pre-#1226 backward compat: no base sha, but a real citation → done.

        The deliberate fail-open direction — a task filed before
        branch_base_sha was recorded must not be hard-failed when its landing
        is positively attributable on main.
        """
        citation = 'c' * 40
        server, _ = self._ancestor_arm_server(
            tmp_path, '903', tip='a' * 40, citation=citation, effect_present=True,
            metadata={},
        )

        result = await _call_merge_status(server, task_id='903')

        assert result.get('state') == 'done', f'Expected done, got: {result}'
        assert result.get('merge_sha') == citation, (
            f'Expected merge_sha={citation!r}, got: {result.get("merge_sha")!r}'
        )

    async def test_citation_pattern_disabled_still_returns_done(
        self, tmp_path: Path
    ) -> None:
        """commit_citation_pattern='' is the documented per-project opt-out.

        find_task_citation_commit honours '' by returning None for
        everything, so running the gate would reject unconditionally and turn
        Tier 3.5 into dead code for projects without citation conventions.
        The gate is skipped entirely; the degeneracy guard still applies.
        """
        tip = 'a' * 40
        server, stub_git = self._ancestor_arm_server(
            tmp_path, '904', tip=tip, citation=None,
            metadata={'branch_base_sha': 'b' * 40},
            commit_citation_pattern='',
        )

        result = await _call_merge_status(server, task_id='904')

        assert result.get('state') == 'done', (
            f'Citation-disabled projects must keep pre-fix behaviour, got: {result}'
        )
        assert result.get('merge_sha') == tip, (
            f'Expected merge_sha={tip!r} (branch tip) when the citation gate is '
            f'disabled, got: {result.get("merge_sha")!r}'
        )
        stub_git.find_task_citation_commit.assert_not_called()

    async def test_citation_pattern_none_uses_default(
        self, tmp_path: Path
    ) -> None:
        """None means "use the built-in default" and is NOT the opt-out."""
        server, stub_git = self._ancestor_arm_server(
            tmp_path, '905', tip='a' * 40, citation=None,
            metadata={'branch_base_sha': 'b' * 40},
            commit_citation_pattern=None,
        )

        await _call_merge_status(server, task_id='905')

        stub_git.find_task_citation_commit.assert_awaited_once_with(
            '905', pattern_template=None,
        )

    @pytest.mark.parametrize(
        ('citation', 'expected_state'),
        [('c' * 40, 'done'), (None, 'unknown')],
        ids=['cited', 'uncited'],
    )
    async def test_get_task_raises_still_applies_citation_gate(
        self, tmp_path: Path, citation: str | None, expected_state: str
    ) -> None:
        """A scheduler fault degrades ONE guard, never the whole probe.

        The metadata lookup's own try/except is nested inside the Tier-3.5
        fire-safe wrapper precisely so a raising get_task cannot swallow the
        citation gate along with the degeneracy check.
        """
        server, _ = self._ancestor_arm_server(
            tmp_path, '906', tip='a' * 40, citation=citation, effect_present=True,
            get_task_raises=True,
        )

        result = await _call_merge_status(server, task_id='906')

        assert isinstance(result, dict), 'merge_status must not raise'
        assert result.get('state') == expected_state, (
            f'Expected {expected_state} with a raising scheduler, got: {result}'
        )


# ---------------------------------------------------------------------------
# Real-git integration test — the canonical 4352 lost-record shape end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatusGitAuthorityIntegration:
    """End-to-end integration test using real git operations.

    Exercises the canonical 4352 shape:
        land branch to main → advance main → delete branch + worktree →
        no ring/event-store record → merge_status must return done/found_on_main.

    Real git subprocess is required to validate the find_merge_marker code
    path that the SimpleNamespace stubs cannot exercise.
    """

    async def test_4352_lost_record_returns_done_found_on_main(
        self, tmp_path: Path, git_ops: GitOps, orch_config: OrchestratorConfig  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """After merge+delete with no ring/event-store record, merge_status returns done.

        Sequence:
        1. Create worktree for task '770', commit a file.
        2. merge_to_main → advance_main (branch lands on main).
        3. cleanup_merge_worktree + cleanup_worktree (branch+worktree deleted).
        4. Build server with NO event_store, stub_harness with real git_ops.
        5. Call merge_status(task_id='770').
        6. Assert: state='done', kind='found_on_main', merge_sha==result.merge_commit,
           len(merge_sha)==40.
        """
        tid = '770'

        # --- Step 1: create worktree and commit ---
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        # --- Step 2: merge to main ---
        merge_result = await git_ops.merge_to_main(wt_info.path, tid)
        assert merge_result.success, f'merge_to_main failed: {merge_result}'
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        adv = await git_ops.advance_main(merge_result.merge_commit)
        assert adv.result == 'advanced'

        expected_sha = merge_result.merge_commit

        # --- Step 3: delete branch + worktree (4352 shape) ---
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Verify branch is gone (confirms the find_merge_marker path will fire)
        rc, sha_out, _ = await _run(
            ['git', 'rev-parse', '--verify', f'task/{tid}'],
            cwd=git_ops.project_root,
        )
        assert rc != 0, f'Branch should be gone (non-zero rc) but rc={rc}, sha={sha_out!r}'

        # --- Step 4: server with NO event_store (ring/event-store miss) ---
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=git_ops,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, harness=stub_harness, orch_config=orch_config)

        # --- Step 5 & 6: call merge_status and assert ---
        result = await _call_merge_status(server, task_id=tid)

        assert result.get('state') == 'done', (
            f'Expected done/found_on_main after 4352 shape, got: {result}'
        )
        assert result.get('kind') == 'found_on_main', (
            f'Expected kind=found_on_main, got: {result}'
        )
        assert result.get('merge_sha') == expected_sha, (
            f'Expected merge_sha={expected_sha!r}, got: {result.get("merge_sha")!r}'
        )
        assert len(result['merge_sha']) == 40, (
            f'merge_sha must be a 40-char SHA, got: {result["merge_sha"]!r}'
        )

    async def test_degenerate_branch_parked_on_old_main_returns_unknown(
        self, tmp_path: Path, git_ops: GitOps, orch_config: OrchestratorConfig  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """End-to-end, no stubs: the user-observable false-positive signal.

        Reproduces the live shape behind ~71 non-terminal tasks: a branch
        created at some main commit that never received a commit, while main
        moved on.  The branch IS an ancestor of main and is NOT at main's tip
        — exactly the census preconditions — so before task 3103 merge_status
        answered done/found_on_main against a foreign commit.

        The advancing commit on main deliberately does NOT cite the task, so
        the citation gate is not what produces the unknown here; the
        degeneracy guard is.
        """
        tid = '3024'
        branch = f'task/{tid}'

        # --- Branch created at current main, zero commits pushed to it ---
        rc, base_raw, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert rc == 0
        branch_base_sha = base_raw.strip()
        rc, _, err = await _run(
            ['git', 'branch', branch, branch_base_sha], cwd=git_ops.project_root,
        )
        assert rc == 0, f'branch create failed: {err!r}'

        # --- main advances with an unrelated, non-citing commit ---
        (git_ops.project_root / 'unrelated.txt').write_text('someone else\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        rc, _, err = await _run(
            ['git', 'commit', '-m', 'chore: unrelated work by another task'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'main advance failed: {err!r}'

        # --- The census preconditions the false positive rode on ---
        assert await git_ops.is_ancestor(branch, 'main'), (
            'precondition: the parked branch must be an ancestor of main'
        )
        rc, main_tip_raw, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert rc == 0
        assert main_tip_raw.strip() != branch_base_sha, (
            'precondition: branch tip must differ from main tip'
        )

        stub_harness = _stub_harness(
            git_ops, metadata={'branch_base_sha': branch_base_sha},
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue, harness=stub_harness, orch_config=orch_config,
        )

        result = await _call_merge_status(server, task_id=tid)

        assert result.get('state') == 'unknown', (
            f'A branch parked on an old main commit must return unknown, '
            f'got: {result}'
        )

    async def test_reseeded_uncited_ancestor_branch_returns_unknown(
        self, tmp_path: Path, git_ops: GitOps, orch_config: OrchestratorConfig  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """End-to-end: a re-seeded branch the degeneracy predicate MISSES.

        A warm lane reset the branch ref FORWARD to a newer main commit after
        creation, so ``tip != branch_base_sha`` and the branch reads as
        non-degenerate — yet it still carries zero of the task's own work and
        no commit on main cites it.  Only the citation gate catches this
        class, which is why both guards ship together.
        """
        tid = '3031'
        branch = f'task/{tid}'

        rc, base_raw, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert rc == 0
        branch_base_sha = base_raw.strip()
        rc, _, err = await _run(
            ['git', 'branch', branch, branch_base_sha], cwd=git_ops.project_root,
        )
        assert rc == 0, f'branch create failed: {err!r}'

        # --- main advances with an unrelated, non-citing commit ---
        (git_ops.project_root / 'unrelated.txt').write_text('someone else\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        rc, _, err = await _run(
            ['git', 'commit', '-m', 'chore: unrelated work by another task'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'main advance failed: {err!r}'
        rc, new_main_raw, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert rc == 0
        new_main = new_main_raw.strip()

        # --- warm-lane re-seed: reset the branch ref forward to newer main ---
        rc, _, err = await _run(
            ['git', 'branch', '-f', branch, new_main], cwd=git_ops.project_root,
        )
        assert rc == 0, f'branch re-seed failed: {err!r}'

        # main advances once more so tip != main_tip still holds
        (git_ops.project_root / 'unrelated2.txt').write_text('and again\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        rc, _, err = await _run(
            ['git', 'commit', '-m', 'chore: more unrelated work'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'second main advance failed: {err!r}'

        # --- preconditions: non-degenerate, yet an ancestor of main ---
        assert await git_ops.resolve_branch_sha(branch) != branch_base_sha, (
            'precondition: the re-seeded tip must differ from branch_base_sha '
            '(so the degeneracy predicate misses)'
        )
        assert await git_ops.is_ancestor(branch, 'main'), (
            'precondition: the re-seeded branch must be an ancestor of main'
        )

        stub_harness = _stub_harness(
            git_ops, metadata={'branch_base_sha': branch_base_sha},
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(
            esc_queue, harness=stub_harness, orch_config=orch_config,
        )

        result = await _call_merge_status(server, task_id=tid)

        assert result.get('state') == 'unknown', (
            f'A re-seeded, uncited ancestor branch must return unknown, '
            f'got: {result}'
        )

    async def test_live_branch_merge_sha_is_the_citation_commit_on_main(
        self, tmp_path: Path, git_ops: GitOps, orch_config: OrchestratorConfig  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """Live-branch path: merge_sha is the citation commit ON MAIN.

        Inverts the pre-3103 pin (``merge_sha == branch_tip``), which recorded
        the documented ``_found_on_main_response`` wart: for a ``--no-ff``
        merge the branch tip is NOT a commit on main's first-parent chain, so
        provenance stamped from it pointed at the wrong commit.

        The citation gate now returns ``verdict.evidence_sha``.
        ``merge_to_main`` writes the canonical ``Merge task/<id> into main``
        subject, which ``DEFAULT_COMMIT_CITATION_PATTERN`` matches, so the
        evidence sha IS the merge commit — the honest provenance anchor the
        runbooks stamp.

        Sequence:
        1. Create worktree for task '771', commit a file.
        2. Capture branch tip SHA before merging.
        3. merge_to_main → advance_main.
        4. Do NOT clean up branch — keep it alive (live-branch path fires).
        5. Build server with NO event_store.
        6. Call merge_status(task_id='771').
        7. Assert: merge_sha == the merge commit on main (NOT the branch tip).
        """
        tid = '771'

        # --- Step 1: create worktree and commit ---
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        # --- Step 2: capture branch tip before merge ---
        rc, branch_tip_raw, _ = await _run(
            ['git', 'rev-parse', f'task/{tid}'],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'Expected rc=0 resolving branch tip, got rc={rc}'
        branch_tip = branch_tip_raw.strip()
        assert len(branch_tip) == 40, f'Expected 40-char SHA, got {branch_tip!r}'

        # --- Step 3: merge to main (--no-ff creates a distinct merge commit) ---
        merge_result = await git_ops.merge_to_main(wt_info.path, tid)
        assert merge_result.success, f'merge_to_main failed: {merge_result}'
        assert merge_result.merge_commit is not None
        # For --no-ff the merge commit is distinct from the branch tip
        assert merge_result.merge_commit != branch_tip, (
            'merge_to_main did a fast-forward — branch_tip == merge_commit; '
            'this test requires --no-ff to demonstrate the semantic distinction'
        )
        adv = await git_ops.advance_main(merge_result.merge_commit)
        assert adv.result == 'advanced'

        # --- Step 4: intentionally keep branch alive → is_ancestor path fires ---
        # (no cleanup_worktree / cleanup_merge_worktree)

        # --- Step 5: server with NO event_store ---
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=git_ops,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, harness=stub_harness, orch_config=orch_config)

        # --- Steps 6 & 7: call merge_status and assert merge_sha == merge_commit ---
        result = await _call_merge_status(server, task_id=tid)

        assert result.get('state') == 'done', (
            f'Expected done/found_on_main for live branch on main, got: {result}'
        )
        assert result.get('kind') == 'found_on_main', (
            f'Expected kind=found_on_main, got: {result}'
        )
        # Live-branch path returns the citation commit found on main
        assert result.get('merge_sha') == merge_result.merge_commit, (
            f'Live-branch path must return the citation commit on main. '
            f'Expected merge_sha={merge_result.merge_commit!r}, '
            f'branch_tip={branch_tip!r}, '
            f'got merge_sha={result.get("merge_sha")!r}'
        )
        # Explicitly pin the inversion: the branch tip is NOT the answer any
        # more (--no-ff creates a distinct merge commit on main, and that
        # commit is the honest provenance anchor)
        assert result['merge_sha'] != branch_tip, (
            'merge_sha must be the merge commit on main, not the branch tip'
        )
