"""Real-git end-to-end tests for the already-landed dispatch gate's shared
landing-evidence helper (task 2678).

Every other test covering ``validate_landing_evidence`` and its five call
sites (``test_landing_evidence.py``, ``test_harness_already_landed_gate_wiring.py``,
``test_reconcile_stranded.py``, ``test_merge_queue_coalesce.py``) drives them
with a ``MagicMock`` ``git_ops`` whose sub-methods are individually stubbed.
This file is the ONE user-observable, real-git boundary check: a genuine
temporary git repository, a REAL ``GitOps`` rooted at it (only
``McpLifecycle`` / ``Scheduler`` / ``BriefingAssembler`` are patched out —
mirrors every other file's ``_build_harness`` helper, which never patches
``GitOps``), and ``Harness._already_landed_dispatch_gate`` driven directly.

Covers the PRD boundary-test sketch (``plans/found-on-main-provenance-integrity-prd.md``)
scenarios #5 and #6, plus a control:

  #5  A no-ff merge marker survives as an ancestor of main forever, but a
      LATER commit on main reverts exactly the deliverable it introduced
      (the task-1175 shape).  The branch ref is already deleted (post-merge
      cleanup), so only the CANDIDATE-mode merge-marker path can attribute
      this landing; ``commit_effect_present_in_main``'s real second-parent
      check must reject it — no ``mark_done``, one ``provenance_unattributed``
      escalation (reason ``effect_absent``).
  ctrl Same shape but the deliverable is NOT reverted — the effect-present
      guard accepts, the gate marks the task done anchored on the marker
      sha, and no escalation is filed.
  #6  A branch's changed files coincidentally match main's independent
      content (the content-equivalence fallback engages) but no commit on
      main cites the task — DISCOVERY mode rejects with ``no_citation``.
      The deleted ``citation or get_main_sha()`` fallback must not
      resurface: no fabricated anchor, no ``mark_done``, one escalation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.git_ops import _run
from orchestrator.harness import Harness

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Real git repo fixture (mirrors test_git_ops.py's git_repo/_setup_repo).
# ---------------------------------------------------------------------------

@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit on main."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ---------------------------------------------------------------------------
# Harness construction — real GitOps, stub scheduler + escalation queue.
# ---------------------------------------------------------------------------

def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out.

    Mirrors test_harness_already_landed_gate_wiring.py's ``_build_harness``
    exactly — only ``McpLifecycle`` / ``Scheduler`` / ``BriefingAssembler``
    are patched, so ``Harness.__init__`` still builds a genuine
    ``GitOps(config.git, config.project_root, ...)``.  Callers point
    ``mock_orch_config.project_root`` at a real git repo BEFORE calling this
    so every git_ops call (``resolve_branch_sha``, ``is_ancestor``,
    ``find_merge_marker``, ``commit_effect_present_in_main``,
    ``branch_content_in_main``, ``find_task_citation_commit``) runs a real
    git subprocess against genuine repository state instead of a mock.
    """
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


def _wire_gate_harness(mock_orch_config, repo: Path, *, task_id: str) -> Harness:
    """Build a Harness with real GitOps rooted at *repo*, a stub scheduler
    recording ``mark_done``, and a recording (MagicMock) escalation queue —
    ready to drive ``_already_landed_dispatch_gate`` end-to-end.

    ``metadata`` carries no ``branch_base_sha`` — the marker-path stale
    check (``_is_valid_sha_40`` guard) is a no-op without one, which is
    correct here: these scenarios are all a FIRST incarnation's landing,
    never a re-created branch under the same task id.
    """
    mock_orch_config.project_root = repo
    h = _build_harness(mock_orch_config)
    h.scheduler.get_task = AsyncMock(
        return_value={'id': task_id, 'metadata': {}},
    )
    h.scheduler.mark_done = AsyncMock()
    h._escalation_queue = MagicMock()
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    # Explicit (task 3534): the gate's veto reads the ANY-LEVEL pending rows,
    # so "no open escalations" must be stated, not inherited from
    # MagicMock.__iter__'s empty default.
    h._escalation_queue.get_by_task = MagicMock(return_value=[])
    h._escalation_queue.make_id = MagicMock(return_value=f'esc-{task_id}-1')
    return h


# ---------------------------------------------------------------------------
# Real-git scenario builders.
# ---------------------------------------------------------------------------

async def _land_via_merge_marker(
    repo: Path, task_id: str, *, revert_deliverable: bool,
) -> str:
    """Build the task-1175 merge-marker shape: ``task/{task_id}`` is merged
    ``--no-ff`` into main, then its branch ref is deleted (as
    ``cleanup_worktree`` would after ``advance_main`` but before
    ``set_task_status('done')`` lands) — so only the merge marker, not a
    live branch ref, can attribute this landing.

    When *revert_deliverable* is True, a LATER commit on main removes the
    deliverable the merge introduced (PRD boundary-test #5, the task-1175
    reverted-merge shape): the merge commit remains an ancestor of main
    forever, but its own second-parent content is no longer present at
    HEAD.  When False, the deliverable is left intact (the control case).

    Returns the merge marker's commit sha.
    """
    branch = f'task/{task_id}'
    rc, _, err = await _run(['git', 'checkout', '-b', branch], cwd=repo)
    assert rc == 0, f'checkout {branch} failed: {err}'
    (repo / 'deliverable.py').write_text('deliverable\n')
    rc, _, err = await _run(['git', 'add', 'deliverable.py'], cwd=repo)
    assert rc == 0, f'git add failed: {err}'
    rc, _, err = await _run(
        ['git', 'commit', '-m', f'impl({task_id}): add deliverable'], cwd=repo,
    )
    assert rc == 0, f'deliverable commit failed: {err}'

    rc, _, err = await _run(['git', 'checkout', 'main'], cwd=repo)
    assert rc == 0, f'checkout main failed: {err}'
    rc, _, err = await _run(
        ['git', 'merge', '--no-ff', branch, '-m', f'Merge {branch} into main'],
        cwd=repo,
    )
    assert rc == 0, f'merge failed: {err}'
    rc, marker_sha, err = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse failed: {err}'
    marker_sha = marker_sha.strip()

    rc, _, err = await _run(['git', 'branch', '-D', branch], cwd=repo)
    assert rc == 0, f'branch -D failed: {err}'

    if revert_deliverable:
        (repo / 'deliverable.py').unlink()
        rc, _, err = await _run(['git', 'add', '-A'], cwd=repo)
        assert rc == 0, f'git add -A failed: {err}'
        rc, _, err = await _run(
            ['git', 'commit', '-m', 'revert deliverable'], cwd=repo,
        )
        assert rc == 0, f'revert commit failed: {err}'

    return marker_sha


async def _land_content_equivalent_without_citation(repo: Path, task_id: str) -> None:
    """Build a branch whose changed files coincidentally match main's own
    INDEPENDENT content (the content-equivalence fallback), with NO commit
    anywhere citing *task_id* — PRD boundary-test #6 (no attributable
    citation -> no stamp).  The branch is left live (not merged, not
    deleted) so the ancestry and marker paths both fall through to the
    content-equivalence fallback.
    """
    branch = f'task/{task_id}'
    rc, _, err = await _run(['git', 'checkout', '-b', branch], cwd=repo)
    assert rc == 0, f'checkout {branch} failed: {err}'
    (repo / 'shared.txt').write_text('coincidental content\n')
    rc, _, err = await _run(['git', 'add', 'shared.txt'], cwd=repo)
    assert rc == 0, f'git add failed: {err}'
    rc, _, err = await _run(
        ['git', 'commit', '-m', 'wip: add shared.txt on branch'], cwd=repo,
    )
    assert rc == 0, f'branch commit failed: {err}'

    rc, _, err = await _run(['git', 'checkout', 'main'], cwd=repo)
    assert rc == 0, f'checkout main failed: {err}'
    (repo / 'shared.txt').write_text('coincidental content\n')
    rc, _, err = await _run(['git', 'add', 'shared.txt'], cwd=repo)
    assert rc == 0, f'git add failed: {err}'
    rc, _, err = await _run(
        ['git', 'commit', '-m', 'chore: unrelated shared.txt on main'], cwd=repo,
    )
    assert rc == 0, f'main commit failed: {err}'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlreadyLandedGateRealGitBoundaries:
    """Real-git end-to-end coverage for PRD boundary-tests #5 and #6."""

    async def test_reverted_merge_marker_escalates_no_mark_done(
        self, mock_orch_config, git_repo,
    ) -> None:
        """PRD boundary-test #5 / the task-1175 shape, end-to-end: a merge
        marker on main whose SECOND-PARENT (branch) content was later
        reverted by a further commit on main.  The branch ref is deleted
        post-merge, so only the merge-marker CANDIDATE-mode path can
        attribute this landing; ``validate_landing_evidence``'s FIX 1'
        effect-present guard must reject it against REAL git state — no
        ``mark_done``, and exactly one ``provenance_unattributed``
        escalation carrying the branch, the marker sha, and reason
        ``effect_absent``.
        """
        task_id = '1175'
        branch = f'task/{task_id}'
        marker_sha = await _land_via_merge_marker(
            git_repo, task_id, revert_deliverable=True,
        )
        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)

        result = await h._already_landed_dispatch_gate(task_id)

        assert result is False
        cast(AsyncMock, h.scheduler.mark_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == task_id
        assert branch in esc.detail
        assert marker_sha in esc.detail
        assert 'effect_absent' in esc.detail

    async def test_intact_merge_marker_marks_done_no_escalation(
        self, mock_orch_config, git_repo,
    ) -> None:
        """Control for boundary-test #5: the deliverable is NOT reverted,
        so the real ``commit_effect_present_in_main`` check accepts — the
        gate marks the task done anchored on the marker sha, with no
        escalation filed.
        """
        task_id = '1176'
        marker_sha = await _land_via_merge_marker(
            git_repo, task_id, revert_deliverable=False,
        )
        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)

        result = await h._already_landed_dispatch_gate(task_id)

        assert result is True
        cast(AsyncMock, h.scheduler.mark_done).assert_awaited_once()
        call = cast(AsyncMock, h.scheduler.mark_done).await_args
        assert call is not None
        assert call.args[0] == task_id
        assert call.kwargs['kind'] == 'found_on_main'
        assert call.kwargs['sha'] == marker_sha
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_content_equivalent_no_citation_escalates_no_mark_done(
        self, mock_orch_config, git_repo,
    ) -> None:
        """PRD boundary-test #6, end-to-end: branch content coincidentally
        matches main (content-equivalence fallback engages) but no commit
        on main cites the task — DISCOVERY mode rejects with reason
        ``no_citation`` against REAL git state.  The gate must not
        fabricate an anchor from main HEAD (the deleted
        ``or get_main_sha()`` fallback); it escalates instead of stamping
        done.
        """
        task_id = '77'
        branch = f'task/{task_id}'
        await _land_content_equivalent_without_citation(git_repo, task_id)
        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)

        result = await h._already_landed_dispatch_gate(task_id)

        assert result is False
        cast(AsyncMock, h.scheduler.mark_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert esc.task_id == task_id
        assert branch in esc.detail
        assert 'no_citation' in esc.detail
