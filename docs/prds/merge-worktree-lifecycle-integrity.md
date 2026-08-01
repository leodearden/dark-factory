# PRD: Merge-worktree lifecycle integrity — request identity + lease-enforced removal

**Project:** dark-factory. **Status:** active, 2026-07-22. **Approach:** B+H (contract + two-way boundary tests).
**Sibling program:** reify `docs/prds/merge-gate-health.md` (W1 restart-collateral, DF tasks 2685/2828/2873 — verifier-side protections, all landed). This PRD is the deleter-side + request-identity completion of that program. No W-item of merge-gate-health claims this territory (checked 2026-07-22).

## 1. Goal (G1 consumer + user-observable surface)

No merge verify ever loses its working tree to another actor in the same system, and no branch ever runs two concurrent verifies of the same work item. Observable surfaces:

- `merge_request` MCP callers (humans via /merge-queue, escalation watchers, workflow path) get an **informative structured rejection** when submitting a newer SHA for a branch whose earlier SHA is already in verify — instead of a silent second verify.
- Orchestrator restart with an in-flight merge produces **one** resumed verify per branch and **zero** `ENOENT`-mid-verify failures (the 2026-07-22 task/5326 signature: `Error: ENOENT ... uv_cwd`, `unknown_test_failure`, spurious `verify_cross_check_mismatch`).
- The resource audit / merge reaper remains the sole disposer of dead `_merge-*` trees; live-held trees are never removed (journal shows explicit `skip: lease held` lines instead of `Cleaned up worktree _merge-verify`).

## 2. Background — incident lineage (why this PRD exists)

Three incidents of one class, fixed piecemeal (full RCA: 2026-07-22 /deb session, journal-attributed):

1. **Pre-2315:** persistent `_merge-verify` clobbered by reset/rmtree during live verify → `MergeVerifyLeaseHeld` guard (git_ops.py:996) — protects `reset_persistent_merge_worktree` only.
2. **2026-07-20, merge_sha 83336a32:** ephemeral `_merge-<hash>` deleted mid-compile by reseed/reclaim → task 2873 lane-lock: cross-check trust-anchor now *holds* `merge_verify_lease(lane_dir=merge_wt)` (merge_queue.py:2244), contended fail-safe landed (merge_queue.py:2268).
3. **2026-07-22, task/5326 merge 952cc6b3:** two *deleters that don't check the lease* struck during a restart:
   - `harness._recover_crashed_tasks` (harness.py:2576, "no plan — cleaning up" at :2957) → `cleanup_worktree` (git_ops.py:10143) `git worktree remove --force`d the **persistent `_merge-verify`** 21s after the same process dispatched a verify into it — and also removed readopted `_merge-ba97f10a`, and attempted `.reseed-trash`, `.merge_verify_pgids`, `_mainprobe-*`, `_mainsweep-*`, `_offline-deep` (most survived only because their git metadata was invalid).
   - `cleanup_merge_worktree` (git_ops.py:7986) removed ephemeral `_merge-a083d4e9` 1.2s after the *duplicate* merge request's verify failed — while the second verify was live inside it.
   - Enabler: `recover_pending_merges` (harness.py:1697) rehydrated **two** journal entries for branch task/5326 (same merge SHA) and enqueued both — bypassing the `InFlightMergeRegistry` coalesce gate (task 1604) that guards live enqueues. Two concurrent local verifies also violate the serial-lane assumption `enforce_persistent_worktree_serial_lane` rests on (shared `target/` safety).

Pattern: verifiers **hold** the lease (2685/2830/2822/2873); only *some* deleters **check** it. Each incident added one actor-pair guard. This PRD replaces per-path patches with two invariants enforced at chokepoints.

## 3. Sketch of approach

**Thrust A — request identity (dedupe/coalesce/reject/cancel).** One live work item per branch, SHA-sensitive, enforced at *both* entry points (submit + journal recovery) via the existing `InFlightMergeRegistry`.

**Thrust B — worktree lifecycle (lease-enforced removal).** One guarded-removal primitive; every deleter of a `_merge-*` tree routes through it; the generic crash-recovery sweep never disposes of infra-namespace entries at all.

## 4. Contract (H)

### C1 — Removal invariant
> **No actor removes a `_merge-*` working tree without holding that tree's lease.**

- New primitive `GitOps.remove_merge_worktree_guarded(path, *, reason) -> RemovalOutcome` (`removed | skipped_lease_held | skipped_persistent | not_present | failed`):
  - **Atomically try-acquires** the tree's lease (`merge_verify_lease(lane_dir=path)` lock file, non-blocking) — *acquire-then-remove*, never check-then-remove (the incident's sweep had a 23s TOCTOU).
  - Contention ⇒ `skipped_lease_held` + one WARNING journal line naming holder + reason. **Skip, never defer/retry**: removing a live-held tree is never correct; true leaks are collected later by the reaper (lease detection is fail-open on dead/stale holders, so a crashed holder never wedges removal).
  - Persistent paths (`_merge-verify`, `_offline-deep`) ⇒ `skipped_persistent` (subsumes the existing `cleanup_merge_worktree` no-op).
- **All** deleters route through it: `cleanup_merge_worktree` (git_ops.py:7986), the harness sweep's merge-entry disposition, `reap_orphaned_merge_worktrees`, the resource-audit auto-remediation that task 2922 will add.

### C2 — Namespace invariant
> **`_`-prefixed and `.`-prefixed entries under `worktree_base` are infra-owned. The generic task-worktree machinery (crash-recovery sweep, orphan reaper) never applies the "no plan ⇒ clean up" heuristic to them.**

Positive match: only entries whose name is task-id-shaped (or an adoptable lane, already special-cased) are task-worktree candidates. `_merge-*` entries are *reported* to the merge reaper's disposition (readopt / age-grace reap via C1); `_mainprobe-*`, `_mainsweep-*`, `_iact-*`, `_spec-*`, `_offline-deep`, `.reseed-trash`, `.merge_verify_pgids`, `.lane-state`, `.task-meta` are left to their owners. Mirrors reify's warm-lane-gc `PROTECT_GLOB` convention ("`_merge-*` MUST stay first"). Kills the exclusion whack-a-mole (LANE_STATE/TASK_META were already excluded one-by-one; the incident showed the list is not maintainable).

### C3 — Request-identity semantics (per branch)

| Incoming submission vs existing live entry | Disposition |
|---|---|
| Same branch, **same SHA** | **Coalesce**: attach caller to existing entry's outcome (sticky-result compatible). Never a second work item — regardless of verify state. |
| Same branch, **new SHA**, existing **queued** (verify not started) | **Replace** iff new SHA is a descendant of (or equal to) the old; drop old entry, clean its scratch via C1. If ancestry says the "new" SHA is an ancestor (stale retry), coalesce-to-existing instead. Diverged (force-push): replace, log divergence. |
| Same branch, **new SHA**, existing **in verify** | **Reject** with structured error: `{code: duplicate_in_verify, existing_mr, existing_sha, verify_age_secs, hint: "merge_cancel then resubmit"}`. |
| `merge_cancel` | Returns only after the entry is **fully retired**: dequeued/aborted, worktrees released via C1, **sticky per-task result cleared** — so an immediate resubmit can never coalesce onto the corpse (fixes the known cancel+resubmit race). |

- Enforced at **both** entry points: `merge_request` (MCP + workflow path — already registry-gated per 1604, gains SHA-sensitivity) and `recover_pending_merges` (currently bypasses the registry entirely; collapse duplicates per branch **before** enqueue — at recovery nothing is in verify, so it is always coalesce/replace, never reject).
- Verify-started check is **atomic** with the replace/reject decision (same lock scope as the registry mutation) — otherwise replacement races dispatch and reproduces the mid-verify teardown.

### C4 — Serial-lane tripwire (rider)
Dispatching a second concurrent *local* merge verify while `_MERGE_AHEAD_BOUND`-derived per-host in-flight is 1 logs a WARNING + emits a telemetry event (no hard block). Cheap detection for any future identity leak; would have flagged 5326 at 12:10:26.

## 5. Resolved design decisions

- **D1** Same-SHA duplicates coalesce (never reject) — protects innocent re-polls after timeout.
- **D2** "Later wins" is ancestry-aware, submission-order as tiebreak on divergence.
- **D3** Post-verify-start newer SHA rejects with structured payload; explicit `merge_cancel` + resubmit is the sanctioned override (making C3's cancel-retirement fix a hard prerequisite of documenting that override).
- **D4** Deleters **skip** on lease contention (leave for reaper); no retry loops.
- **D5** Recovery dedupe reuses `InFlightMergeRegistry` (1604) — no second registry.
- **D6** **No startup reordering.** `_recover_crashed_tasks` still runs after merge resume; C1+C2 make the ordering irrelevant for this class. Reordering the startup sequence has independent blast radius (step-2e comment's "no live workflow" assumption is wrong today; fixing the *assumption* rather than the order).
- **D7** Namespace invariant (C2) over per-name exclusion lists.
- **D8** 2922 (teardown half-completion + junit husk) stays separate with a dependency edge onto the C1 primitive — its "verified-dead, no live holders" auto-remediation check *is* C1's contention probe.
- **D9** The zeta done-gate's per-row classes duplicate their upstream unit tests, and that is **accepted** (adjudicated task 3153, 2026-07-30, from task 2929 review suggestion #1 / esc-2929-2). The alternative — deleting the ports, or thinning them to composition assertions that call the upstream helpers — was rejected on measured evidence: (i) the ports are redundant with upstream but **not** with the capstone (`TestFiveThreeTwoSixReplayGate`), which asserts neither the rows-1/2/4 INFO skip-reporting nor the rows-6/7 peer-future mirror and order-independence, and covers rows 8 and 9 not at all — so deletion would reduce the gate's Sec.9 row coverage, defeating its purpose as one executable Sec.9 row matrix; (ii) "call the upstream helpers" means cross-test-module imports of private fixtures, which collides with the per-file duplication convention pervasive and explicit throughout `orchestrator/tests`, trading a documented explicit cost for an undocumented implicit coupling; (iii) task 2929's plan steps 01-10 each mandate a named per-row class and are each done against a landed commit. The accepted cost is bounded and itemised per row in the gate module's PROVENANCE table (`orchestrator/tests/test_merge_worktree_lifecycle_integration_gate.py`). **Reopen** if the capstone is ever extended to assert the rows-1/2/4 INFO skip-reporting and the rows-6/7 peer-mirror + order-independence deltas (rows 1,2,4,6,7 then become genuinely redundant *within* this file and should be deleted), or if a production change to the alpha–epsilon legs is observed to land in one file and not the other (drift realised, not merely risked).
- G7 waiver: no-lockstep-duplication — the zeta done-gate's per-row classes restate assertions their upstream unit tests already make (see the PROVENANCE table in `orchestrator/tests/test_merge_worktree_lifecycle_integration_gate.py` for the measured per-row duplicate-vs-unique split). Extraction is unavailable: the shared-helper alternative is a cross-test-module import of private test fixtures, which the repo's per-file duplication convention rejects, and the two sites are not required to agree byte-for-byte — they independently assert the same production contract. Drift control is the PROVENANCE split plus the upstream node ids it names. See D9.

## 6. Pre-conditions / substrate (G3 — verified 2026-07-22 against live source)

| Assumed capability | Evidence |
|---|---|
| Per-lane lease incl. ephemerals | `merge_verify_lease(lane_dir=merge_wt)` call, merge_queue.py:2244-2248 (task 2873, landed incl. contended fail-safe :2268) |
| Fail-closed reset guard + fail-open holder detection | `MergeVerifyLeaseHeld` git_ops.py:996-1013 |
| Coalesce registry | `InFlightMergeRegistry` (task 1604, done) |
| Merge reaper w/ readoption + age grace | `reap_orphaned_merge_worktrees` (task 2060, done; harness.py:8367) |
| Journal recovery path | `recover_pending_merges` (harness.py:8331, run() step 1c0a :1697) |
| Sweep + chokepoints to modify | harness.py:2576-2961; git_ops.py:7986, 10143 |
| Incident ground truth | `journalctl --user` 2026-07-22 11:10Z: `Recovery: worktree _merge-verify has no plan — cleaning up` (12:10:46.9), `Cleaned up merge worktree _merge-a083d4e9` (12:10:56.5), `recovered mr-7e689858` + `mr-29dfdbc2` both `task/5326` (12:10:22.9) |

## 7. Cross-PRD / cross-task seams (G4)

| Seam | Owner |
|---|---|
| Verifier-side lease holding (done: 2685/2830/2822/2873) | merge-gate-health W1 — untouched; this PRD consumes the lease primitive |
| Deleter-side enforcement + request identity | **this PRD** |
| Dead-tree auto-remediation + junit-husk guard | task 2922 (dep edge → C1 primitive leaf; 2922's remediator calls `remove_merge_worktree_guarded`) |
| Cancel+resubmit as sanctioned override | this PRD (C3); obsoletes reify memory `merge-cancel-race` guidance once landed |
| reify warm-lane GC scripts | already correct (`PROTECT_GLOB` protects `_merge-*`); out of scope |

## 8. Decomposition plan (Greek letters → task ids at decompose)

- **α — C1 primitive**: `remove_merge_worktree_guarded` in git_ops + route `cleanup_merge_worktree` through it. Signal: unit test — removal during a held lease returns `skipped_lease_held`, tree intact, WARNING logged; uncontended removal proceeds; dead-holder lease does not block. *(No prereqs.)*
- **β — sweep/namespace (C2)**: `_recover_crashed_tasks` + orphan reaper apply positive task-shape match; `_merge-*` routed to merge-reaper disposition via α; other infra namespaces untouched. Signal: harness test — planted `_merge-verify` (live lease), `_merge-<uuid>` (live lease), `.reseed-trash`, `_mainprobe-x` all survive a recovery sweep; a task-id-shaped planless dir is still cleaned. *(Prereq: α.)*
- **γ — recovery dedupe (C3 recovery path)**: `recover_pending_merges` collapses per-branch through `InFlightMergeRegistry` before enqueue. Signal: journal fixture with two entries for one branch (same SHA; and new-SHA variant) recovers to exactly one enqueued mr; both pollers resolve. *(No prereqs; shares C3 table with δ.)*
- **δ — submit-path SHA semantics (C3)**: registry gains same-SHA coalesce / ancestry replace / in-verify reject with structured payload. Signal: MCP-level test observes the documented rejection payload (negative assertion: the reject actually fires) and the replace path drops the queued entry + cleans its scratch. *(Prereq: α for the scratch cleanup path.)*
- **ε — cancel retirement (C3)**: `merge_cancel` retires fully before returning; sticky result cleared. Signal: cancel-then-immediate-resubmit test — resubmission gets a fresh entry, lands the new SHA, never sees the stale result. *(Prereq: δ.)*
- **ζ — 5326-replay integration gate (H)**: end-to-end restart simulation — in-flight merge, duplicated journal entries, live verify planted in `_merge-verify` + an ephemeral, then full startup (merge resume + crash-recovery sweep concurrently). Asserts: one verify per branch, zero ENOENT, both worktrees intact until their verifies complete, merge finalizes. Signal: the boundary-test suite passes; this is the PRD's done gate. *(Prereqs: α, β, γ, δ, ε.)*
- **η — C4 tripwire (rider, small)**: concurrent-local-verify telemetry warning. Signal: unit test triggers two local dispatches, observes the event. *(Prereq: γ.)*
- **Out-of-batch dep**: `add_dependency(2922 → α)`.

## 9. Boundary-test sketch (H — both faces of the seam)

| # | Face | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | deleter | sweep meets live-leased `_merge-verify` | verify holds lease | tree intact; `skipped_lease_held` logged |
| 2 | deleter | sweep meets live-leased ephemeral `_merge-<uuid>` | trust-anchor-style holder | tree intact |
| 3 | deleter | reaper meets dead-holder `_merge-<uuid>` past grace | stale lock file, no live pgid | removed (fail-open) |
| 4 | deleter | sweep meets `.reseed-trash` / `_mainprobe-*` | no plan.json | untouched (C2) |
| 5 | verifier | verify runs to completion while sweep executes concurrently | 5326 timing | verify exit ≠ ENOENT; result recorded |
| 6 | identity | journal holds 2 entries, same branch+SHA | restart | 1 enqueued mr; both requesters resolve |
| 7 | identity | journal holds 2 entries, same branch, SHA₂ descendant | restart | SHA₂ enqueued, SHA₁ dropped+cleaned |
| 8 | identity | submit SHA₂ while SHA₁ in verify | live verify | structured reject observed; SHA₁ verify undisturbed |
| 9 | identity | cancel during verify, resubmit immediately | live verify | retire completes first; fresh entry; no stale sticky result |
| 10 | serial | two local dispatches attempted | bound=1 | C4 event emitted |

## 10. Out of scope

- 2922's junit-husk writer guard and dead-tree remediation internals (dep-linked, separately owned).
- Startup-sequence reordering (D6).
- reify-side GC/sweep scripts (already lease/protect-correct).
- Remote-runner allocation and cross-check verdict policy (merge-gate-health W2 territory).
- Generalizing C1 locking to non-merge infra namespaces (their owners have their own locks; C2 merely stops *this* subsystem deleting them).

## 11. Open questions (tactical)

1. **Reject error-code string + payload field names** — align with existing MCP error envelope conventions. Decide in δ.
2. **`RemovalOutcome` representation** (enum vs str literals) — match git_ops house style. Decide in α.
3. **Registry keying** — `branch.bare_id` vs full ref for the recovery path; confirm 1604's existing key. Decide in γ.
4. **C4 event name** (`merge_serial_lane_violation`?) — match EventType naming. Decide in η.
