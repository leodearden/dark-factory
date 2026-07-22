# Capability manifest — merge-worktree lifecycle integrity

PRD: `docs/prds/merge-worktree-lifecycle-integrity.md` (commit d0ae78bbe7).
Mechanizes G3 + G6 per leaf (skills/prd `references/gates.md` → Capability
Manifest). All substrate evidence re-verified 2026-07-22 against live source
at decompose time. Machine-readable twin:
`merge-worktree-lifecycle-integrity.capability-manifest.yaml` (same stem;
task_ids stamped by `commit_planning`).

Negative-assertion discipline (G6 branch 4): every reject/skip signal below
must be **observed to fire** (the outcome value / WARNING line / structured
payload appears), never inferred from silence or from a tree merely still
existing. Each such test needs the paired positive control so a trivially
inert implementation cannot pass.

## α — C1 guarded-removal primitive

- `lease-primitive-consumable-for-try-acquire` → capability→producer
  (wired) — `merge_verify_lease(lane_dir=merge_wt)` held in production at
  merge_queue.py:2244-2248 (task 2873, incl. contended fail-safe :2268);
  fail-closed guard + fail-open holder detection `MergeVerifyLeaseHeld`
  git_ops.py:996-1013. **PASS**
- `guarded-removal-primitive-exists-and-routed` → producer: α itself;
  `cleanup_merge_worktree` (git_ops.py:7986) routes through it. Name
  `remove_merge_worktree_guarded` is contract-fixed by PRD C1 (Open Q2
  covers only the RemovalOutcome representation, not the fn name or the
  outcome token vocabulary). delivered_check: grep
  `def remove_merge_worktree_guarded` present. **PASS**
- `lease-held-skip-observed-to-fire` → rejection-mechanism built+bound by
  α: test holds a live lease, invokes removal, observes
  `skipped_lease_held` + one WARNING naming holder + reason; uncontended
  removal proceeds (positive control). delivered_check: grep
  `skipped_lease_held|SKIPPED_LEASE_HELD` present (token contract-fixed by
  C1; alternation covers enum-member casing). **PASS**
- `dead-holder-fail-open` → holder-liveness detection exists
  (git_ops.py:996-1013 fail-open on dead/stale holders); α's test binds
  "dead-holder lease does not block removal". delivered_check: manual —
  liveness-probe quality is a test property, not greppable. **PASS**

## β — C2 sweep/namespace invariant

- `sweep-chokepoints-exist` → capability→producer (wired) —
  `_recover_crashed_tasks` harness.py:2576 ("no plan — cleaning up"
  :2957) → `cleanup_worktree` git_ops.py:10143; orphan reaper adjacent.
  **PASS**
- `infra-namespace-survival-observed` → negative assertion bound by β's
  harness test: planted `_merge-verify` (live lease), `_merge-<uuid>`
  (live lease), `.reseed-trash`, `_mainprobe-x` all survive a recovery
  sweep with the skip/report disposition **observed** (explicit journal
  disposition, not silence); positive control: a task-id-shaped planless
  dir is still cleaned (an inert sweep fails the control).
  delivered_check: manual — the positive-match predicate literal is
  implementation-chosen; bound by β's test + control. **PASS**
- `merge-entries-routed-via-alpha` → DAG-direction: α is upstream (dep
  wired at decompose); `_merge-*` disposition calls α's primitive.
  delivered_check: manual. **PASS**

## γ — C3 recovery-path dedupe

- `coalesce-registry-reuse` → capability→producer (wired) —
  `InFlightMergeRegistry` (task 1604, done) imported + consumed in
  production merge path (merge_queue.py:143, :3590); D5 forbids a second
  registry. **PASS**
- `recovery-collapse-observed` → producer: γ. Journal fixture with two
  entries for one branch (same-SHA, and new-SHA-descendant variant)
  recovers to exactly **one** enqueued mr; both pollers resolve (coalesce
  attach observed, not inferred). Substrate: `recover_pending_merges`
  harness.py:8331, invoked at run() step 1c0a harness.py:1697.
  delivered_check: manual — the wiring site (harness vs merge_queue
  helper) is implementation-chosen; registry keying is Open Q3. **PASS**

## δ — C3 submit-path SHA semantics

- `registry-gate-at-submit-exists` → capability→producer (wired) —
  `merge_request` already registry-gated per 1604; δ adds
  SHA-sensitivity. **PASS**
- `structured-reject-observed-to-fire` → rejection-mechanism built+bound
  by δ (G6 branch 4): MCP-level test submits a newer SHA while the old
  SHA is in verify and **observes** the structured rejection payload fire
  (code + existing_mr + existing_sha + verify_age_secs + hint), with the
  SHA₁ verify undisturbed; observed absence of the payload ⇒ FAIL.
  delivered_check: manual — the code string / field names are Open Q1
  (aligned to MCP error-envelope conventions in δ); no stable grep token
  at authoring time. **PASS**
- `replace-path-scratch-cleanup-via-alpha` → DAG-direction: α upstream
  (dep wired); the dropped queued entry's scratch is cleaned via C1.
  delivered_check: manual. **PASS**
- `decision-atomic-with-registry-mutation` → INV-3: verify-started check
  shares the lock scope of the replace/reject mutation (PRD C3);
  race-freedom bound by δ's test. delivered_check: manual. **PASS**

## ε — C3 cancel retirement

- `merge-cancel-endpoint-exists` → capability→producer (wired) —
  `merge_cancel` is a live escalation-server MCP tool (verified in the
  live tool registry 2026-07-22). **PASS**
- `full-retirement-before-return-observed` → producer: ε; δ upstream.
  Cancel-then-immediate-resubmit test: resubmission gets a **fresh**
  entry (observed as a new mr identity), lands the new SHA, and never
  coalesces onto the retired entry's sticky result — fresh-entry evidence
  observed, not inferred from lack of failure. delivered_check: manual —
  sticky-result clearing is registry-internal state. **PASS**

## ζ — 5326-replay integration gate (H; PRD done-gate)

- `all-legs-produced-upstream` → DAG-direction: α, β, γ, δ, ε all
  upstream deps (wired at decompose); no capability is owned downstream.
  **PASS**
- `boundary-matrix-green-e2e` → §9 rows 1–9 exercised: restart with
  in-flight merge + duplicated journal entries + live verify planted in
  `_merge-verify` and an ephemeral; asserts one verify per branch, zero
  ENOENT, both trees intact until their verifies complete, merge
  finalizes. Startup substrate: merge resume harness.py:1697 + sweep
  harness.py:2576-2961 run concurrently today (incident ground truth,
  PRD §6). delivered_check: manual — the task IS the check suite; its
  signal is the green boundary matrix. **PASS**

## η — C4 serial-lane tripwire

- `serial-lane-substrate-exists` → capability→producer (wired) —
  `enforce_persistent_worktree_serial_lane` merge_liveness.py:723;
  `_MERGE_AHEAD_BOUND` lives module-level in merge_queue (reach-back
  documented merge_liveness.py:763-765). **PASS**
- `tripwire-event-observed` → producer: η. Test triggers two concurrent
  local dispatches at bound=1 and **observes** the WARNING + telemetry
  event fire (no hard block). delivered_check: manual — event name is
  Open Q4 (EventType naming decided in η). **PASS**

## G7 note (design invariants — no waivers)

No invariant hits; batch filed with no `g7_waivers`. INV-4 note for the
record: C1's lease-contention **skip** is the PRD's one fail-soft path; its
storm escape is the existing resource-audit L2 (`merge_resource_leak`,
task 1994/2060 family) which flags any `_merge-*` tree persisting past
grace — a tree skipped repeatedly surfaces there, and dead-holder
fail-open (α) prevents the wedge case. C4 (η) is itself a tripwire, and
δ's reject is caller-visible by construction.
