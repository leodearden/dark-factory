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
    # Explicit for the same reason (task 4499): the filer's auto-dismiss guard
    # reads a TERMINAL record on this citation, so "nothing was previously
    # adjudicated" must be stated, not inherited from MagicMock's truthy
    # auto-child — which would silently suppress every filing below.
    h._escalation_queue.find_terminal_by_citation = MagicMock(return_value=None)
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
        # Task 3116 part A, end-to-end.  The gate is still SAFE here, and the
        # three assertions below pin what an operator can ACT on: the
        # machine-readable reason code (``effect_absent``), the labelled
        # ``diverged paths`` block — whose header is the structural evidence
        # that the paths are rendered under their own label rather than buried
        # in the ``probe: {...}`` dict repr — and the real deliverable path
        # this scenario's fixture created, which the probe had to compute.
        # Each fails if the step-4 enrichment or the step-6 rendering
        # regresses.
        #
        # The operator PROSE around them is deliberately not asserted: it is
        # not a contract, part (b) moved the semantics underneath it (the
        # check no longer decides on byte-identity), and pinning the wording
        # here would let the assertion outlive a real regression while
        # blocking the corrective edit at its source.
        assert 'effect_absent' in esc.detail
        assert 'diverged paths' in esc.detail
        assert 'deliverable.py' in esc.detail

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


async def _commit_all(repo: Path, message: str) -> str:
    """Stage everything (including deletions) and commit, returning the sha."""
    rc, _, err = await _run(['git', 'add', '-A'], cwd=repo)
    assert rc == 0, f'git add -A failed: {err}'
    rc, _, err = await _run(['git', 'commit', '-m', message], cwd=repo)
    assert rc == 0, f'commit {message!r} failed: {err}'
    rc, sha, err = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse failed: {err}'
    return sha.strip()


def _lines(prefix: str, count: int) -> str:
    """*count* unique, non-blank lines.

    Uniqueness is load-bearing: survival is measured by line-SET membership,
    so a duplicated or short-common line can be "present" at main by pure
    coincidence and a fixture that tripped over it would measure the
    coincidence instead of the deliverable.
    """
    return ''.join(f'{prefix}_{i:05d} = {i}\n' for i in range(count))


async def _land_branch_marker(
    repo: Path, task_id: str, mutate, main_edit=None,
) -> str:
    """Land ``task/{task_id}`` via a ``Merge task/N into main`` no-ff marker.

    *mutate* is called with *repo* on the branch and may write OR delete files
    (staging is ``git add -A``, so a deletion is a first-class deliverable —
    that is the vacuous shape task 3116 b3 exists for).  The branch ref is
    deleted after the merge, exactly as ``cleanup_worktree`` does before
    ``set_task_status('done')`` lands, so only the marker can attribute the
    landing.  Returns the marker sha.

    *main_edit*, when given, is called with *repo* back on main AFTER the
    branch commit and BEFORE the merge, and its result is committed on main.
    ORDER IS THE WHOLE POINT: the branch must fork BEFORE that edit, so
    ``merge-base(parent1, parent2)`` precedes it and the ``--no-ff`` merge
    genuinely auto-integrates two independent edits.  Making the edit first
    and forking afterwards produces a fixture where main HEAD's tree for the
    touched paths is byte-identical to parent2's — no divergence at all — so
    even the retired byte-identity predicate would accept it and the test
    would discriminate nothing.
    """
    branch = f'task/{task_id}'
    rc, _, err = await _run(['git', 'checkout', '-b', branch], cwd=repo)
    assert rc == 0, f'checkout {branch} failed: {err}'
    mutate(repo)
    await _commit_all(repo, f'impl({task_id}): branch work')
    rc, _, err = await _run(['git', 'checkout', 'main'], cwd=repo)
    assert rc == 0, f'checkout main failed: {err}'
    if main_edit is not None:
        main_edit(repo)
        await _commit_all(repo, "chore: an unrelated task's edit to a hot file")
    rc, _, err = await _run(
        ['git', 'merge', '--no-ff', branch, '-m', f'Merge {branch} into main'],
        cwd=repo,
    )
    assert rc == 0, f'merge failed: {err}'
    rc, marker_sha, err = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse failed: {err}'
    rc, _, err = await _run(['git', 'branch', '-D', branch], cwd=repo)
    assert rc == 0, f'branch -D failed: {err}'
    return marker_sha.strip()


class TestAlreadyLandedGateSurvivalSemantics:
    """End-to-end: the false-positive class part (b) removes (task 3116).

    Every scenario here is a CLEAN LANDING that the pre-task byte-identity
    check rejected.  The cost of each rejection was not a cheap re-check: the
    task was left pending, DISPATCHED to an agent on the next tick for a full
    plan/verify/review cycle, and a spurious ``task_failure`` escalation was
    filed — and because byte-identity, once broken, is never restored, the
    condition is ABSORBING, so it repeated every tick.  Three live tasks
    (3653, 3640, 3717) burned days and ~5.80 USD this way.

    These drive the REAL gate over REAL git state, which is the only place the
    whole chain — marker discovery, CANDIDATE-mode validation, survival
    measurement, escalation filing — is exercised together.
    """

    async def test_cotouched_hot_file_no_longer_reads_as_a_revert(
        self, mock_orch_config, git_repo,
    ) -> None:
        """(a) THE INSTANCE-2 SHAPE: the branch touches its own deliverable
        AND a shared hot file that main independently edited at a distant,
        non-conflicting line before the merge.

        This landing is CLEAN — the deliverable is present at main and so is
        every line the branch added to the shared file.  Byte-identity said
        otherwise purely because main carries an unrelated neighbour's edit to
        a co-touched file, and that false positive is what cost a full
        spurious dispatch.  The gate must now accept: mark_done awaited, no
        escalation.

        The fixture ORDER is what makes this shape real, and is asserted
        below rather than assumed.  The branch forks from the seeded manifest
        FIRST and APPENDS its entry to the file as read from disk; only then
        does the neighbour prepend its line on main.  So the fork point
        precedes the neighbour commit, ``git merge --no-ff`` auto-integrates
        two genuinely independent edits, and main HEAD carries BOTH — while
        parent2's pre-merge blob carries only the branch's.  That is exactly
        the divergence byte-identity rejected and survival accepts: the probe
        must still NAME ``shared.manifest`` as diverged while reporting
        ``present``.  ``test_git_ops.py``'s
        ``test_merge_names_only_the_co_touched_hot_file`` is the unit-level
        template for the same shape.
        """
        task_id = '3640'
        (git_repo / 'shared.manifest').write_text(_lines('shared', 40))
        await _commit_all(git_repo, 'seed the shared hot file')

        def _branch_work(repo: Path) -> None:
            (repo / 'deliverable.py').write_text(_lines('deliv', 12))
            manifest = repo / 'shared.manifest'
            manifest.write_text(manifest.read_text() + _lines('branch_entry', 3))

        def _neighbour_edit(repo: Path) -> None:
            manifest = repo / 'shared.manifest'
            manifest.write_text(_lines('neighbour', 1) + manifest.read_text())

        marker_sha = await _land_branch_marker(
            git_repo, task_id, _branch_work, main_edit=_neighbour_edit,
        )

        merged_manifest = (git_repo / 'shared.manifest').read_text()
        assert 'neighbour_00000' in merged_manifest, (
            "main HEAD must carry the neighbour's independent edit"
        )
        assert 'branch_entry_00000' in merged_manifest, (
            "main HEAD must carry the branch's own added lines"
        )

        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)
        probe = await h.git_ops.describe_commit_effect_in_main(marker_sha)
        assert 'shared.manifest' in probe.diverged_paths, (
            'the co-touched hot file must diverge from parent2 — that '
            'divergence IS the false positive byte-identity rejected'
        )
        assert 'deliverable.py' not in probe.diverged_paths
        assert probe.present is True, 'survival must accept what byte-identity rejected'

        result = await h._already_landed_dispatch_gate(task_id)

        assert result is True, 'a clean landing beside a co-touched hot file'
        cast(AsyncMock, h.scheduler.mark_done).assert_awaited_once()
        call = cast(AsyncMock, h.scheduler.mark_done).await_args
        assert call is not None
        assert call.kwargs['sha'] == marker_sha
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_later_additive_evolution_no_longer_reads_as_a_revert(
        self, mock_orch_config, git_repo,
    ) -> None:
        """(b) THE TASK-3653 SHAPE: the merge lands, then main ADDS further
        lines to the same file without disturbing any the branch added.

        Byte-identity is broken; survival is total.  This is the exact live
        shape that re-dispatched task 3653 and left it blocked four days.
        """
        task_id = '3653'

        def _branch_work(repo: Path) -> None:
            (repo / 'deliverable.py').write_text(_lines('deliv', 12))

        marker_sha = await _land_branch_marker(git_repo, task_id, _branch_work)
        (git_repo / 'deliverable.py').write_text(
            _lines('deliv', 12) + _lines('later_unrelated', 8),
        )
        await _commit_all(git_repo, 'feat: a later task appends to the same file')

        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)
        result = await h._already_landed_dispatch_gate(task_id)

        assert result is True
        cast(AsyncMock, h.scheduler.mark_done).assert_awaited_once()
        call = cast(AsyncMock, h.scheduler.mark_done).await_args
        assert call is not None
        assert call.kwargs['sha'] == marker_sha
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_deletion_deliverable_is_accepted_when_it_holds(
        self, mock_orch_config, git_repo,
    ) -> None:
        """(d) A deliverable that is a FILE DELETION adds no lines at all, so
        an added-lines-survive test is trivially true for it.

        Reached through the REAL gate, not just the unit tests: the vacuous
        arm must decide it by the mechanism that applies — is the file still
        absent at main — and accept.
        """
        task_id = '3717'
        (git_repo / 'obsolete.py').write_text(_lines('obsolete', 20))
        await _commit_all(git_repo, 'seed the file this task will delete')

        marker_sha = await _land_branch_marker(
            git_repo, task_id, lambda repo: (repo / 'obsolete.py').unlink(),
        )

        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)
        result = await h._already_landed_dispatch_gate(task_id)

        assert result is True
        cast(AsyncMock, h.scheduler.mark_done).assert_awaited_once()
        call = cast(AsyncMock, h.scheduler.mark_done).await_args
        assert call is not None
        assert call.kwargs['sha'] == marker_sha
        cast(MagicMock, h._escalation_queue).submit.assert_not_called()

    async def test_resurrected_deletion_is_still_caught_as_a_revert(
        self, mock_orch_config, git_repo,
    ) -> None:
        """(d), the safety half: putting back the file the deliverable removed
        IS a genuine revert, and a zero-added-lines branch must not be waved
        through on the vacuous path.

        This is the task-1175 clobber in deletion shape — the exact failure
        that would make part (b) "a green light that proves nothing".
        """
        task_id = '3718'
        (git_repo / 'obsolete.py').write_text(_lines('obsolete', 20))
        await _commit_all(git_repo, 'seed the file this task will delete')
        await _land_branch_marker(
            git_repo, task_id, lambda repo: (repo / 'obsolete.py').unlink(),
        )
        (git_repo / 'obsolete.py').write_text(_lines('obsolete', 20))
        await _commit_all(git_repo, 'resurrect the deleted file on main')

        h = _wire_gate_harness(mock_orch_config, git_repo, task_id=task_id)
        result = await h._already_landed_dispatch_gate(task_id)

        assert result is False
        cast(AsyncMock, h.scheduler.mark_done).assert_not_awaited()
        cast(MagicMock, h._escalation_queue).submit.assert_called_once()
        esc = cast(MagicMock, h._escalation_queue).submit.call_args[0][0]
        assert esc.category == 'provenance_unattributed'
        assert 'vacuous_effect_absent' in esc.detail
