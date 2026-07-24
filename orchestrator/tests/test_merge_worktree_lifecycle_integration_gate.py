"""Merge-worktree lifecycle integrity zeta done-gate: restart-simulation
boundary suite (PRD Sec.9 rows 1-9).

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task zeta (the B+H
done-gate).  All five prerequisite legs are LANDED and BEHAVIOUR-FROZEN for
this batch:

  alpha (2924) -- GitOps.remove_merge_worktree_guarded (git_ops.py:8239):
      lease-enforced removal primitive; outcome vocabulary 'removed' /
      'skipped_lease_held' / 'skipped_persistent' / 'not_present' / 'failed'.
  beta  (2925) -- classify_worktree_entry (git_ops.py:467) + the C2 namespace
      guard in Harness._recover_crashed_tasks (harness.py:2842-2859): the
      crash-recovery sweep SKIPS+REPORTS `_merge-*`/infra bands instead of
      force-removing them (the 2026-07-22 task/5326 incident).
  gamma (2926) -- recover_pending_merges' registry-gated per-branch collapse
      (merge_queue_store.py:433-537): a branch with N surviving journal
      entries enqueues exactly ONE winner (descendant-most snapshot tip);
      every loser attaches as a peer waiter whose future mirrors the
      winner's terminal outcome.
  delta (2927) -- coalesce_or_enqueue_merge_request's duplicate_in_verify
      reject (merge_queue.py:2991/4293): a newer SHA submitted while the
      earlier SHA is IN VERIFY is structurally REJECTed, not coalesced or
      replaced.
  epsilon (2928) -- retire_cancelled_merge_request (see
      test_merge_cancel_retire.py): a merge_cancel FULLY retires the
      cancelled entry (registry slot + worktree + sticky retention) before
      returning, so an immediate resubmit gets a fresh, uncorrupted slot.

This file is the ONE NEW test file the done-gate adds -- a TEST-ONLY
COMPOSITION gate exercising all five legs, alone and together, across
Sec.9 boundary rows 1-9:

  1. A live-leased persistent `_merge-verify` survives a concurrent
     crash-recovery sweep (C2 skip-by-name; C1 futureproofs the lease).
  2. A live-leased ephemeral `_merge-<uuid>` survives the same sweep.
  3. A DEAD-holder ephemeral fails OPEN (guarded removal succeeds); a
     LIVE-held ephemeral is skipped with exactly one WARNING naming the
     holder pgid + reason.
  4. Non-merge infra bands (`.reseed-trash`, `_mainprobe-x`, ...) are left
     to their owner by the SAME sweep that cleans a task-shaped planless
     dir (the positive control proving the sweep is not inert).
  5. (capstone) the live verify observes its own worktree intact across the
     concurrent sweep -- see the capstone class docstring for the exact
     zero-ENOENT causal-proxy chain; this row has no standalone test.
  6. Two journal entries for one branch with the SAME snapshot tip
     collapse to ONE enqueued winner; the loser's future mirrors the
     winner's terminal outcome (OBSERVED, not inferred).
  7. Two journal entries for one branch with ancestor/descendant tips
     collapse to the DESCENDANT, order-independently.
  8. A newer SHA submitted while the branch's earlier SHA is IN VERIFY is
     structurally REJECTed (`duplicate_in_verify`) -- the live entry is
     left undisturbed.
  9. A cancelled merge is FULLY retired (slot + worktree + sticky) before
     an immediate resubmit, which gets a genuinely fresh entry rather than
     coalescing onto the retired corpse.

Row 10 (the C4 concurrent-local-verify serial-lane telemetry tripwire) is
OUT OF SCOPE for this gate -- it belongs to task eta (a separate rider
leaf, PRD Sec.8/Sec.9), so it is not exercised here.

Concurrency model -- READ THIS BEFORE editing test bodies
-----------------------------------------------------------
Harness.run()'s two startup recovery entry points, `_recover_pending_merges`
(step 1c0a, harness.py:1881) and `_recover_crashed_tasks` (step 2c,
harness.py:2010), are SEQUENTIAL awaits -- NOT gathered/parallelized. The
2026-07-22 task/5326 incident's concurrency was the pre-launched merge-worker
BACKGROUND TASK (step 1b, `_start_merge_worker` -> create_task) draining the
re-enqueued `_merge_queue` WHILE the crash-recovery sweep scanned worktrees.
The capstone below reproduces this exactly: it starts a REAL merge worker
with a gated `run_scoped_verification` (holding a verify live in its own
`_merge-<hash>` tree) BEFORE awaiting the sweep. Do NOT add an assertion
that `_recover_pending_merges`/`_recover_crashed_tasks` run concurrently
with EACH OTHER -- that would fail against current code and misrepresent
the design (PRD D6: no startup reordering; C1+C2 make the ordering
irrelevant for this class of bug).

'Zero ENOENT' is a failure MODE, not a matchable token
----------------------------------------------------------
The incident's `Error: ENOENT ... uv_cwd` signature appears only in PRD
prose -- it is not a FailureCategory, not an EventType, and is not asserted
anywhere in the tree via string match. It is proved via CAUSAL PROXIES
instead:
  (a) every `_merge-*`/infra tree survives the concurrent sweep
      (`.exists()`), paired with a POSITIVE CONTROL (the task-shaped '999'
      dir the SAME sweep cleans) so a survives-because-the-sweep-is-inert
      false pass is impossible;
  (b) the gated verify runner asserts its OWN `_merge-<hash>` cwd worktree
      exists at entry AND at completion -- a tree yanked mid-verify would
      fail this assertion, directly modelling the incident;
  (c) the recovered merge reaches `outcome.status == 'done'` and its
      branch lands on main ('merge finalizes');
  (d) zero spurious `verify_cross_check_mismatch` L1 escalations are filed
      (the incident's clobbered-worktree false-FAIL signature).

SCOPE -- TEST-ONLY / BEHAVIOUR-FROZEN
----------------------------------------
Every production surface exercised below (alpha-epsilon, tasks 2924-2928)
already SHIPPED and is frozen for this batch. This is a COMPOSITION gate:
it wires already-landed callables together and asserts their combined
behaviour. If a scenario surfaces a GENUINE production defect, ESCALATE
(category='design_concern' or 'scope_violation') rather than editing
production here -- editing frozen production from this task would widen
the concurrency lock on the hottest files in the repo (harness.py,
merge_queue.py) and conflict with the frozen seam the prerequisite tasks
already landed.

STALE-OFFSET WARNING
------------------------
Every `:NNNN` line citation above (and in inline comments below) can drift
as the modules it cites are edited by unrelated work. Always locate
symbols BY NAME (grep/search), never trust a line offset.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import GitOps
from orchestrator.verify_cancel import release_merge_verify_flock, remove_lock_holder_pgid

# ---------------------------------------------------------------------------
# TestDeleterFace -- PRD Sec.9 rows 1-4
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeleterFace:
    """PRD Sec.9 rows 1-4: the crash-recovery sweep's deleter face.

    (a) rows 1, 2, 4 + positive control -- Harness._recover_crashed_tasks()
    must SKIP+REPORT every `_merge-*`/infra tree (never remove one
    directly; that is the merge reaper's job) while still cleaning a
    task-shaped planless dir in the SAME pass (the positive control
    proving the sweep is not inert). Ported from
    test_crash_recovery.py::TestRecoverCrashedTasksC2Namespace.

    (b) row 3 -- GitOps.remove_merge_worktree_guarded's dead-holder
    fail-open contrasted with row 1/2's live-held skip, on REAL ephemeral
    git worktrees (remove_merge_worktree_guarded's 'removed'/'failed'
    outcomes are only meaningful against a real registered worktree; a
    plain ``mkdir()``'d directory always returns 'failed'). Ported from
    test_remove_merge_worktree_guarded.py's live-held-skip / dead-holder
    fail-open pair.
    """

    async def test_merge_and_infra_trees_survive_sweep_task_shaped_cleaned(
        self,
        mock_orch_config: MagicMock,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Rows 1, 2, 4 + positive control: protected trees survive the
        sweep with an explicit INFO skip/report line each; the task-shaped
        planless dir is the ONLY cleanup_worktree call."""
        harness = _build_recovery_harness(mock_orch_config, git_repo)
        base = harness.git_ops.worktree_base

        merge_verify = base / '_merge-verify'
        fd_verify = _plant_leased_tree(base, merge_verify)
        merge_uuid = base / '_merge-ba97f10a'
        fd_uuid = _plant_leased_tree(base, merge_uuid)

        infra_dirs = {
            name: _plant_infra_dir(base, name)
            for name in (
                '.reseed-trash', '_mainprobe-x', '.lane-state',
                '.task-meta', '_offline-deep',
            )
        }

        wt_task = _plant_task_dir(base, '999')

        try:
            with caplog.at_level(logging.INFO, logger='orchestrator.harness'):
                await harness._recover_crashed_tasks()
        finally:
            release_merge_verify_flock(fd_verify)
            release_merge_verify_flock(fd_uuid)
            remove_lock_holder_pgid(base)

        # Positive control: the ONLY cleanup_worktree call is the
        # task-shaped planless dir -- any merge/infra cleanup call would
        # push the count past one (the 5326 "Cleaned up worktree
        # _merge-verify" regression).
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt_task, '999')  # type: ignore[attr-defined]

        cleaned_paths = {
            c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list  # type: ignore[attr-defined]
        }
        protected = {merge_verify, merge_uuid, *infra_dirs.values()}
        assert cleaned_paths.isdisjoint(protected), (
            f'C2 violated -- sweep cleaned protected entries: '
            f'{cleaned_paths & protected}'
        )
        for d in protected:
            assert d.exists(), f'{d.name} must survive the recovery sweep'

        # Skip disposition OBSERVED (not silence): every protected entry is
        # named in an explicit INFO record.
        info_messages = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.INFO
        ]
        for name in ('_merge-verify', '_merge-ba97f10a', '_mainprobe-x',
                     '_offline-deep', '.reseed-trash', '.lane-state',
                     '.task-meta'):
            assert any(name in m for m in info_messages), (
                f'missing explicit skip/report line naming {name}'
            )

    async def test_dead_holder_fails_open_live_holder_skips(
        self,
        git_ops: GitOps,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Row 3: a dead/stale lease holder never wedges removal (fail
        open), contrasted with a genuinely live holder (skip, single
        WARNING naming pgid + reason), on real ephemeral merge worktrees."""
        dead_wt = await _make_ephemeral_worktree(git_ops)
        live_wt = await _make_ephemeral_worktree(git_ops)
        base = git_ops.worktree_base

        _plant_dead_holder_tree(base, dead_wt)

        outcome_dead = await git_ops.remove_merge_worktree_guarded(dead_wt, reason='reaper')
        assert outcome_dead == 'removed', (
            'a stale holder-pgid record with no live flock must fail OPEN'
        )
        assert not dead_wt.exists()

        fd_live = _plant_leased_tree(base, live_wt)
        try:
            with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
                outcome_live = await git_ops.remove_merge_worktree_guarded(live_wt, reason='reaper')

            assert outcome_live == 'skipped_lease_held', (
                'a LIVE lease holder must skip removal, never force through'
            )
            assert live_wt.exists(), 'a live lease holder must leave the tree intact'

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, (
                f'expected exactly one WARNING, got {len(warnings)}: '
                f'{[r.getMessage() for r in warnings]}'
            )
            message = warnings[0].getMessage()
            assert str(os.getpgrp()) in message, message
            assert 'reaper' in message, message
        finally:
            release_merge_verify_flock(fd_live)
            remove_lock_holder_pgid(base)
