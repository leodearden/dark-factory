# PRD: Stranding remediation + scheduler-pause ergonomics

**Date:** 2026-07-22 · **Status:** approved for decomposition · **Approach: B**
(self-contained harness/ops mechanisms; the load-bearing merge-lane work lives
in `plans/merge-verdict-integrity-prd.md`, which is the high-stakes sibling).
**Provenance:** reify task-5260 stranding RCA — the *second half* of the
incident: after the phantom merge-block (sibling PRD), the task sat `pending`
7+ hours with verified-green work in lane `_lane-9` because (a) the
stranded-blocked remediation re-pended it into a scheduler that had been
paused for 26 minutes (`ewa_trip_29.1818`), (b) the `scheduler_paused` L1 sat
un-triaged ~9h through an AFK window, and (c) the EWA storm itself was fed by
warm-lane pool saturation (41/48 lanes `assigned`, some stale since 07-09; 22+
`warm_lane_pool_exhausted` retry-cap escalations in 2.5h). Cite by symbol;
refs as-of DF `main 08925d962e`.

## 0. Consumer + user-observable surface (G1)

- **α (merge-queue-direct remediation)** — consumer: the harness
  stranded-blocked reaper's resolution path (the same reconcile that filed
  `esc-5260-9`). Operator-ratified direction (2026-07-22): a stranded task
  whose lane holds verified-green, work-complete state is submitted **directly
  to the merge queue** (which runs fine under scheduler pause — proven by the
  manual `mr-973ad563`), not re-pended for a re-dispatch it may never get.
  Escalate born-at-L2 if the merge/verify durably fails.
- **β (pause-aware remediation + pause visibility)** — consumer: escalation
  resolution paths that flip a task toward dispatch, the AFK digest reader
  (the operator), and the auto-watcher.
- **γ (WIP-preserving lane reclamation)** — consumer: the warm-lane pool
  allocator (`GitOps._acquire_warm_lane_impl`) — reclaimed lanes end pool
  exhaustion, which was the storm source behind the EWA trip.
- **δ (EWA trip storm classification)** — consumer: the digest reader deciding
  "resume vs investigate" from the pause reason alone.

## 1. Sketch of approach

Four small, independent leaves in `orchestrator/` (+ digest templates). No new
subsystems; each mechanism extends an existing reconcile/digest/lifecycle path.

## 2. Resolved design decisions

1. **α is deterministic — no LLM.** Verified-green shape :=
   lane `LaneRecord.state == assigned` for the task ∧ lane branch tip ==
   the task's last `workflow_verify.passed=true` `tip_sha` ∧ plan.json
   all-steps-done (`task_runtime._derive_phase == 'DONE'`). On match: submit
   `merge_request` (task_id + `task/<id>` branch), auto-resolve the
   stranded-blocked escalation recording the action; the merge queue's own
   full verify remains the gate (a stale-green branch simply fails there →
   born-at-L2 `stranded_merge_failed`, task stays blocked, worktree/branch
   preserved). Non-matching stranded tasks keep today's re-pend path.
2. **α never bypasses**: no direct-to-main, no verify skip — direct *submission*
   only; all sibling-PRD invariants apply to the resulting merge.
3. **β re-pend visibility**: any resolution path that flips a task to
   `pending`/dispatchable while the scheduler is paused appends a loud marker
   to the resolution text, emits `repend_while_paused`, and the digest gains a
   standing "paused with N re-pended tasks waiting" line. `scheduler_paused`
   escalations pending > 2h are re-surfaced in every digest until resolved.
   Resolution stays human-reserved (no auto-resume — an EWA trip can be a real
   quality signal; classification is δ's job).
4. **γ first pass is conservative**: reclaim only lanes whose assigned task is
   TERMINAL (`done`/`cancelled`) — release via the existing lifecycle
   (branches already survive release per the branch-lifecycle-decouple PRD;
   assert branch ref exists before release). Lanes assigned to live tasks
   (`pending`/`in-progress`/`blocked`) are **never** auto-reclaimed — lane-9
   held 5260's good work precisely while `pending`; blind idle-age reclaim
   would have destroyed the work the incident was about. Non-terminal stale
   assignments (> `lane_stale_report_days`, default 7) get a digest census
   line, human decides.
5. **δ is annotation-only**: the pause reason and digest carry the trip
   window's dominant escalation category share (e.g.
   `ewa_trip_29.18 · storm: 22/29 retry_cap_exhausted(warm_lane_pool)`) so an
   infra-starvation storm is distinguishable from a quality collapse at a
   glance. No behavior change to the trip itself.

## 3. Pre-conditions (G3 — verified on main)

Stranded-blocked reaper + escalation path exist (`esc-5260-9`,
`agent_role=harness-stranded-blocked-reaper`); `LaneRecord`/`.lane-state`
durable records exist; `task_runtime._derive_phase` exists; `merge_request`
accepts operator submissions while the scheduler is paused (empirical:
`mr-973ad563` verified + landed under an active pause); digest writer +
`DigestInputs` exist; EWA trip writes the digest before pausing
(`_maybe_write_digest` step 14); branch-lifecycle decoupling landed
(`plans/warm-lane-branch-lifecycle-decouple-prd.md`).

## 4. Out of scope

- Anything gating verdict adoption or verify fidelity (sibling PRD).
- Auto-resuming a paused scheduler; changing EWA thresholds.
- Aggressive lane reclaim of live-task lanes; lane pool re-sizing.
- The reify-side kloc-guard fixes (reify micro-PRD).

## 5. Cross-PRD seams (G4)

| Seam | Owner |
|---|---|
| Merge submission path used by α | sibling PRD owns verdict integrity; α is a plain client of `merge_request` |
| Lane lifecycle transitions used by γ | `warm-lane-branch-lifecycle-decouple` PRD owns branch survival; γ asserts it, never re-implements |
| Digest content (β, δ) | this PRD; digest *format* conventions unchanged |

## 6. Decomposition plan (leaf → user-observable signal)

- **α — merge-queue-direct stranded remediation.** Signal: harness test
  reproducing the 5260 end-state (blocked task, no open esc, lane assigned
  with green-verified DONE plan) shows the reaper submitting `merge_request`
  (event `merge_queued` with `source=stranded-reaper` detail) instead of
  re-pending; forced merge-verify failure files born-at-L2
  `stranded_merge_failed` and preserves branch + lane; the happy path lands
  and the task reaches `done` with normal provenance.
- **β — pause-aware re-pend + pause visibility.** Signal: resolving a
  stranded esc with `resume` while paused produces a resolution text
  containing the paused-marker, a `repend_while_paused` event, and the next
  digest lists the task under "paused, waiting"; a `scheduler_paused` esc
  pending > 2h reappears in the digest.
- **γ — terminal-task lane reclamation + stale census.** Signal: a lane
  assigned to a `done` task is released on the next reclaim pass (lane-state
  file transitions `assigned→released`, branch ref still resolvable), pool
  free-count rises; a 10-day-stale lane on a `blocked` task is NOT touched and
  appears in the digest census line.
- **δ — EWA trip storm annotation.** Signal: a synthetic escalation storm of
  `retry_cap_exhausted` trips the EWA and the pause reason string + digest
  carry the category-share annotation.

Dependencies: none between leaves; all four independent.

## 7. Open (tactical) questions

- α: whether the reaper submits immediately on detection or after one
  confirmation tick (race with a concurrent legitimate dispatch) —
  implementer decides with a lock-or-recheck.
- γ: reclaim-pass cadence (piggyback the existing warm-lane GC pass vs own
  timer) — implementer's call.
