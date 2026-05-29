# RCA — `recon_stale_run` escalation fires while Stage 2 is still actually running

**Date:** 2026-05-28
**Symptom queue:** `data/reconciliation/escalations` (escalation MCP port 8103)
**Affected category:** `recon_stale_run`
**Same dedupe fingerprint across repeats:** `06f933d8c157f5df3e94510251cc76d03776fe5ceaaf1f112f33f93c26449f95`

## TL;DR

The reaper's stale-run recovery path **unconditionally deletes the project
lock** whenever it recovers any stale run, including legitimate orphan runs
from a previous process. When the *current live* full cycle is running in the
same project, that release strips the live cycle of its lock. On a subsequent
reaper tick (or even within the same `_recover_stale_runs` pass), the live
run's `lock_holder == run.instance_id` guard then fails (because
`lock_holder` is now `None`), so the live, still-actually-running cycle is
misclassified as stale, marked `failed`, has its drained events restored, and
an `recon_stale_run` escalation is filed. The live cycle then finishes its
work, overwrites the journal status back to `completed`, but the spurious
escalation has already been queued.

A secondary effect: remediation runs (`run_type='remediation'`) **do not call
`mark_run_active`** — they inherit the parent full cycle's lock. So once the
lock has been deleted by the bug above, the remediation pass runs the rest of
the cycle with no lock at all, and it will also be falsely reaped at the
600 s mark.

This entirely explains the user's three-event cadence today
(13:54:00, 13:57:42, 14:14:28 UTC):

| time (UTC) | run id  | run_type    | iid          | what happened |
|------------|---------|-------------|--------------|---------------|
| 13:54:00   | c51466ed | remediation | dbf8dddd (dead) | **Legitimate** stale recovery of an orphan from a previous process. Side-effect: lock row for `dark_factory` deleted. |
| 13:57:42   | 50dfff5c | full        | c3787f0e (live) | **False positive.** Live cycle has been running 604 s; lock was deleted 3 m ago by the previous step, so `lock_holder is None`. Reaper marks the live run `failed`. The live cycle later finishes and overwrites the row back to `completed`. |
| 14:14:28   | 6e29076d | remediation | c3787f0e (live) | **False positive.** The remediation pass that ran after 50dfff5c never re-acquires a lock; the deleted lock is still gone. Same misclassification. |

After the project loop's `finally` block finally runs at 14:18:13, the next
cycle calls `mark_run_active` again and the lock is re-created with the
correct instance_id — and from that point onward the long-running 97b49a64
(29:58 minutes!) is correctly skipped by the reaper. The escalations stop
exactly when a `mark_run_active` re-creates the lock row.

## File-and-line walkthrough

### Where the warning is emitted

`fused-memory/src/fused_memory/reconciliation/harness.py:511-555`
(`ReconciliationHarness._recover_stale_runs`):

```python
async def _recover_stale_runs(self) -> None:
    cutoff = self.config.stale_run_recovery_seconds            # 600 s
    stale_runs = await self.journal.get_stale_runs(cutoff)
    for run in stale_runs:
        lock_holder = await self.buffer.get_lock_holder_instance_id(run.project_id)
        if (
            lock_holder is not None
            and run.instance_id is not None
            and lock_holder == run.instance_id
        ):
            continue
        logger.warning(
            f'Recovering stale run {run.id} for {run.project_id} '
            f'(started {run.started_at.isoformat()}, lock expired)'
        )
        run.stage_reports['_error'] = {...}
        await self.journal.update_run_stage_reports(run.id, run.stage_reports)
        await self.journal.complete_run(run.id, 'failed')
        restored = await self.buffer.restore_drained(run.project_id)
        if restored:
            logger.info(f'Restored {restored} drained events for stale run {run.id}')
        await self.buffer.mark_run_complete(run.project_id)        # ← lock DELETE
        await self._replay_deferred_writes(run.project_id)
        self._escalate('recon_stale_run', run.id, f'Run stale (>{cutoff}s, lock expired), recovered')
```

Called from the harness's outer management loop on every 5 s iteration
(harness.py:777).

### Where the lock TTL is set

`fused-memory/src/fused_memory/config/schema.py:354-364`:
- `stale_lock_seconds: float = 7200.0` — heartbeat-staleness cutoff for the
  lock row (`reconciliation_locks`)
- `stale_run_recovery_seconds: int = 600` — age cutoff for runs in the journal

The 12× asymmetry between these two is intentional: the reaper is supposed to
respond to long-running rows quickly via the *lock-holder identity check*,
not via lock TTL expiry. Lock TTL is the last-resort cleanup for genuinely
dead processes that never released their lock.

### The faulty release

`fused-memory/src/fused_memory/reconciliation/event_buffer.py:625-631`:

```python
async def mark_run_complete(self, project_id: str) -> None:
    """Release the reconciliation lock."""
    async with self._txn() as db:
        await db.execute(
            'DELETE FROM reconciliation_locks WHERE project_id = ?',
            (project_id,),
        )
```

The DELETE has **no `instance_id` predicate**. Combined with the unconditional
call from `_recover_stale_runs`, this is the cross-instance lock theft.

### Why the heartbeat doesn't repair the damage

`fused-memory/src/fused_memory/reconciliation/event_buffer.py:633-640`:

```python
async def heartbeat(self, project_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    async with self._txn() as db:
        await db.execute(
            'UPDATE reconciliation_locks SET heartbeat_at = ? WHERE project_id = ? AND instance_id = ?',
            (now, project_id, self.instance_id),
        )
```

After the reaper's DELETE, the row is gone. `UPDATE … WHERE` no-ops with zero
rowcount; the heartbeat never reinserts. `mark_run_active` is the only
inserter, and it is only called at the *start* of a project_loop iteration,
not from within a running cycle.

### Why the remediation pass is also vulnerable

`fused-memory/src/fused_memory/reconciliation/harness.py:1079-1081` and
`1351-1392`: `_maybe_remediate` → `_run_remediation_pass` is invoked inside
`run_full_cycle` *after* `complete_run('completed')` for the parent. It
creates a new journal row (`run_type='remediation'`, `instance_id=self.buffer.instance_id`)
via `journal.start_run`, but it **never calls `mark_run_active`** — it relies
on the parent's lock for ~10–20 more minutes of work. Once the parent's lock
has been deleted by the reaper, this entire window is unprotected.

### Confirmed in the live database

I queried `data/reconciliation/reconciliation.db` directly. The data matches
this trace exactly:

```
start: 13:43:58Z end: 13:54:00Z status: failed     iid: dbf8dddd run: c51466ed proj: dark_factory  (StaleRunRecovery, real orphan)
start: 13:47:38Z end: 14:04:21Z status: completed  iid: c3787f0e run: 50dfff5c proj: dark_factory  (live full cycle, status was 'failed' between 13:57:42 and 14:04:21 then overwritten back to 'completed')
start: 14:04:23Z end: 14:18:13Z status: completed  iid: c3787f0e run: 6e29076d proj: dark_factory  (live remediation, falsely reaped 14:14:28)
start: 14:18:21Z end: 14:48:19Z status: completed  iid: c3787f0e run: 97b49a64 proj: dark_factory  (NEW cycle that re-acquired the lock — never reaped despite running 29:58)
```

The 50dfff5c run shows `status='completed'` in the journal even though the
reaper marked it `failed`: the live cycle's later `complete_run('completed')`
call overwrites the status field. The `_error` is also gone because
`update_run_stage_reports` writes the live in-memory `run.stage_reports` dict
(which never contained `_error`) after the reaper's update.

This race between the live cycle and the reaper means the journal looks
fine in retrospect, while the escalation queue is stuffed with phantom
stale-run records.

### Targeted-recon side-issue (minor, related)

`fused-memory/src/fused_memory/reconciliation/targeted.py:139-148`:
`TargetedReconciler.reconcile_task` constructs a `ReconciliationRun` without
setting `instance_id`. Result: every targeted-recon row in the journal has
`instance_id IS NULL`. (Confirmed: 6699 targeted rows in the DB, all with
NULL instance_id.) These complete in seconds so they don't *themselves*
trigger stale-run alarms, but the NULL-instance bucket is what the
"pre-migration runs are recovered unconditionally" branch was designed for;
the population is now a mix of legacy migration debris and currently-active
targeted-recon code. Not load-bearing for this bug; flagged for hygiene.

## Why the same-fingerprint dedup is not collapsing repeats

This is separate from the false-positive emission but worth answering. The
recon dedup config is in `harness.py:76-87`:

```python
_RECON_DEDUP_CONFIG = (
    dataclasses.replace(
        DedupeConfig.for_recon(),
        infra_dedupe_categories=(
            'recon_integrity_issue',
            'recon_failure',
            'recon_stale_run',
            'recon_backlog_overflow',
        ),
    )
    if HAS_ESCALATION else None
)
```

`DedupeConfig.for_recon()` (escalation/src/escalation/dedupe.py:160) uses an
**unbounded** window (`float('inf')`) and `key_fn=content_fingerprint_key`.
So matching fingerprints should fold across any time span.

The catch is in `find_dedupe_parent`
(escalation/src/escalation/dedupe.py:281):

```python
for parent in queue.get_pending():
    ...
```

Dedup only folds against **pending** parents. Once the watcher (port 8103
session) closes a `recon_stale_run` via `resolve_issue`, that escalation is
no longer in `get_pending()`. The next emission has nothing to fold into and
submits as a fresh `queued` record. `dedupe_count` therefore stays 0 across
repeats whenever the watcher has been closing things on its old cadence —
which the user explicitly noted ("8103 historical pile (already cleaned)").

This is **by design** for `infra_dedupe`. It is NOT the same code path as the
A7a/b "infra noise across resolutions" issue the user is tracking separately:
the recon path here uses the unbounded window and the fingerprint key, but it
still gates on `pending` because `get_pending()` is the only source list.
A7a/b would have to keep an archive index to fold against resolved parents.
That is a deliberate larger design choice and out of scope for this RCA.

In other words: the dedup machinery is doing exactly what it was specified to
do; the *primary* fix is to stop emitting false positives. Dedup would not
hide the recon harness's incorrect behaviour anyway — every closed escalation
gets a fresh re-fire on the next bogus trigger.

## Why the reaper marks a run stale at the same instant Stage 2 completes

It does not happen at the same instant *causally* — it happens at the same
instant *because the reaper tick had been waiting for the 600 s threshold
and Stage 2 happened to complete in the same 5 s reaper sleep window*.
Stage 1 ≈ 4 min + Stage 2 ≈ 5 min totals ≈ 9 min, which is statistically
right next to the 600 s threshold. So:

- The reaper fires every 5 s.
- The live cycle has been running ~601-605 s (just past threshold).
- The lock_holder check is fooled (lock_holder is None because the previous
  orphan recovery deleted it).
- The reaper marks the run failed, restores drained events, files the
  escalation.
- Within the same few seconds, Stage 2 hits its `await stage.run(...)`
  return, the loop falls out, and `journal.complete_run(run_id, 'completed')`
  runs.

The collision is a consequence of the cycle length being narrowly above the
recovery cutoff combined with the lock having been wrongly released. It is
not a TOCTOU between Stage 2 and the journal write; it is the reaper acting
on the still-running row before the live cycle's own commit.

## Proposed fix

**File:** `fused-memory/src/fused_memory/reconciliation/harness.py`
**Function:** `_recover_stale_runs` (line 511)
**Change kind:** behavioural — remove the unconditional `mark_run_complete`
call from the reaper path, replace with a guarded ownership-aware release.

### Change sketch

Remove the unconditional `await self.buffer.mark_run_complete(run.project_id)`
on line 553 and replace with a conditional release that only fires when the
project lock actually belongs to the run being recovered. If the lock is held
by a live instance (whose instance_id differs from the orphan's), leave the
lock alone — the live instance's project_loop is responsible for releasing
it. If there's no lock at all (lock_holder is None), there's nothing to do.

```python
# Replace harness.py:553
# Before:
#     await self.buffer.mark_run_complete(run.project_id)
#
# After:
# Only release the project's lock when it actually belongs to the run
# we're reaping. A lock held by a different (live) instance, or no lock
# at all, must NOT be touched — releasing it would strip the live
# cycle's lock and cause the reaper to misclassify it as stale on the
# next tick. (Locks that genuinely outlive their owner are cleaned up
# by the heartbeat-staleness sweep in event_buffer.get_lock_holder_instance_id
# / mark_run_active at stale_lock_seconds = 7200 s.)
if (
    lock_holder is not None
    and run.instance_id is not None
    and lock_holder == run.instance_id
):
    # Unreachable: this case is filtered out by the `continue` above.
    pass
elif lock_holder is not None and run.instance_id is None:
    # Pre-migration NULL-instance orphan whose lock is held by SOMEONE.
    # We can't prove ownership; leave the lock to the stale-lock sweep.
    pass
# else: lock_holder is None — nothing to release.
```

Equivalent simpler form: **just delete the `mark_run_complete` call**. The
lock-staleness sweep already covers the case where a dead instance left its
own lock behind: `get_lock_holder_instance_id` and `mark_run_active` both
DELETE rows whose `heartbeat_at` is older than `stale_lock_seconds`. So a
genuinely orphan lock from a dead instance is cleaned up the next time
*any* code path runs that helper — within minutes of any reconciliation
activity.

### Optional secondary hardening

1. **Tighten `mark_run_complete` itself** to filter by instance_id (defense in
   depth, callable-by-callable). Signature change to
   `mark_run_complete(project_id, instance_id=None)`; when `instance_id` is
   passed, add `AND instance_id = ?` to the DELETE. Then update both the
   `_project_loop` finally callers (line 869, 912) to pass
   `self.buffer.instance_id`. This is the right long-term shape regardless
   of the recovery-path fix.

2. **Restrict `restore_drained` to the run's drained events** (currently
   restores all drained events project-wide). Today this restores the *live*
   cycle's drained events too, which causes the next cycle to reprocess
   them — duplicate work but not incorrect. A per-run claim_id on drained
   rows would let `restore_drained` filter precisely. Not strictly required
   for the symptom; defer unless this becomes a real cost driver.

3. **Set `instance_id` on targeted-recon runs**
   (`reconciliation/targeted.py:139-148`). Single-line change:
   `instance_id=self.buffer.instance_id` in the `ReconciliationRun`
   constructor. Not load-bearing for the bug, hygiene only.

4. **Bump `stale_run_recovery_seconds`** above the realistic upper bound of a
   full cycle (Stage 1+2 has measured at ≈9 minutes today; long
   remediation passes push parent+remediation past 30 minutes). E.g. 1800 s
   would put the alarm well outside normal cycle duration. This is a
   palliative — the lock-holder check is still the right primary guard —
   but it shrinks the false-positive window when the primary guard is
   broken or when an actually-dead orphan leaves uncleaned debris. Treat as
   a tuning question, not a fix.

### Suggested new test (regression guard)

In `fused-memory/tests/test_harness.py`, alongside
`test_recover_stale_runs_recovers_when_different_instance_holds_lock`
(line 3255):

```python
@pytest.mark.asyncio
async def test_recover_stale_runs_does_not_release_live_lock(
    journal, event_buffer, mock_memory_service
):
    """Recovering an orphan must not delete a lock held by the live instance.

    Regression for the false-positive recon_stale_run cascade observed
    2026-05-28 (rca: plans/recon-stale-recovery-rca.md): when the reaper
    recovers an orphan from a dead instance, calling mark_run_complete
    stripped the lock from the current cycle on the same project, causing
    subsequent reaper iterations to misclassify the live run as stale.
    """
    harness = _make_test_harness(journal, event_buffer, mock_memory_service)
    project_id = 'test-project'
    cutoff = harness.config.stale_run_recovery_seconds

    # Orphan from a dead instance.
    orphan = ReconciliationRun(
        id='run-orphan-dead', project_id=project_id, run_type=RunType.full,
        trigger_reason='unit-test',
        started_at=datetime.now(UTC) - timedelta(seconds=cutoff * 2),
        status=RunStatus.running, instance_id='dead-instance',
    )
    await journal.start_run(orphan)

    # Live instance holds the lock for its own current cycle.
    assert await event_buffer.mark_run_active(project_id) is True
    live_iid = event_buffer.instance_id

    await harness._recover_stale_runs()

    # Lock must still be held by the live instance.
    assert await event_buffer.get_lock_holder_instance_id(project_id) == live_iid
```

This test will fail today against `mark_run_complete`'s unconditional DELETE,
and pass after either remediation above.

## Confidence

- **Root cause (lock-deletion on orphan recovery breaking live lock):** very
  high. Confirmed by source trace + live DB inspection matching the user's
  three-event cadence exactly, and matching the precise transition from
  false-positives to silence at 14:18:21 when the next cycle's
  `mark_run_active` re-creates the lock.
- **Dedup explanation (pending-only fold + watcher closure):** very high.
  Direct read of `find_dedupe_parent`.
- **Fix correctness (drop the `mark_run_complete` call):** high. The
  alternative (heartbeat-staleness sweep) already covers the orphan-lock
  case; nothing in the code relies on the reaper releasing the lock
  synchronously. Verified by checking every caller of `mark_run_complete`
  and every reader of `reconciliation_locks`.
