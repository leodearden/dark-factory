# PRD: Merge-verdict integrity — no unverified or ill-founded verdict ever gates a landing

**Date:** 2026-07-22 · **Status:** approved for decomposition · **Approach: B + H**
(contract + two-way boundary tests — this PRD sits on the load-bearing path to
`main`; the invariants below earn the full contract treatment).
**Provenance:** reify task-5260 stranding RCA (2026-07-21/22, this repo's
interactive sessions; memory `project_stranded_verified_green_5260_rca_2026_07_21`),
two commissioned forensic investigations (laptop fast-PASS forensics; steward
requeue-economics mining, 184 pairs Jun 12–Jul 21), and direct event-store +
laptop-side evidence. Cite code by symbol; line refs are as-of DF `main`
`08925d962e` and drift.

## 0. Consumer + user-observable surface (G1)

Every mechanism below is consumed by the existing merge pipeline itself —
`SpeculativeMergeWorker` dispatch/finalize, `VerifyRunnerPool`/`RemoteRunner`,
steward requeue handling — and surfaces to the operator through `runs.db`
events, the merge-queue heartbeat, and escalations. User-observable outcome:
**a red tree can no longer land on a target project's `main` via a verify that
ran nothing, ran stale gate code, or verified a tree whose base chain had
already died** — and a green task can no longer be blocked by a verdict from a
dead speculative base (the reify-5260 phantom block).

Incident anchors (all verified against `runs.db` + git forensics):
- `966f23a6` (reify, 07-20 04:41Z): guard-red tree landed; sole verdict = 7.9s
  laptop **trivial pass** (no command ran; laptop DF checkout frozen at
  `bb834dd42a` 06-11, predating the task-1774 pipeline-guard consult).
- `83336a32` (reify, 07-20 17:45Z): red tree landed via **0ms local trivial
  pass** (ENOENT-clobbered worktree → empty `existing_files` → guards fail
  open). The 2822 cross-check diverged 3 seconds *after* CAS-advance.
- Task 5260 (reify, 07-20 09:06Z): 43-min FAIL verdict against speculative
  train tip `4728cb00` **adopted 50+ minutes after that base was known-dead**
  (head 5213 failed; successors 5299/5232 were re-merged; 5260's edge dangled)
  → task blocked on a phantom failure, stranded ~30h.
- Steward same-tip requeues: 34 determinate pairs, 38% win rate, 1.24
  burned-min per productive-min (vs 0.34 changed-branch); wins are verify-env
  flake re-rolls (10/13 identical base), not "main healed"; 16/18 signature-known
  losses refailed on a *different* test. 5213's blind same-tip requeue became a
  train head and poisoned three successors.

## 1. Contract (the invariants — G5 §A)

- **INV-1 (evidence-backed verdicts).** A merge-role verify PASS is adoptable
  only if the verify *executed the project's merge gate* on the exact tree it
  vouches for. A verify that resolves to "nothing to run" (no source files, no
  module configs, empty `existing_files`, empty command set) is **not** a PASS:
  for merge role it escalates to the full gate, and if no full-gate command
  exists it FAILS loud. Fail-open trivial passes are abolished for merge role
  on every host.
- **INV-2 (contract currency).** A remote runner's verdict is adoptable only if
  the runner executed current gate logic. Operationally: RemoteRunner
  auto-syncs the remote DF checkout (fail-closed) before dispatch when stale
  (design decision §3.1).
- **INV-3 (chain-intact adoption).** A verdict (PASS or FAIL) may be adopted
  iff, at adoption time, its verified tree ≡ `current-main ∘ S` where `S` is an
  ordered sequence of still-live pipeline items — none failed, ejected,
  superseded, or re-merged since verify dispatch. A broken-chain verdict is
  **void** (event `verdict_voided`, named broken link), and the item is
  re-merged + re-verified; a FAIL verdict from a dead base never blocks a task.
  This deliberately generalizes two-layer §5.3 from dispatch-time to
  adoption-time, and is **verify-depth-agnostic**: a deep multi-merge PASS with
  an intact chain vouches for its whole prefix (the variable-depth
  probe/deep-frontier work is unimpeded — the rule is chain-intact, NOT
  `base == main`).
- **INV-4 (re-rolls are never load-bearing for others).** An identical-tip
  merge retry (same `merge_sha^2` as the failed attempt) is capped at 1, never
  forms or joins a train, and is never selected as a speculation base —
  successors must not stack on a re-roll.
- **INV-5 (detectors must be able to fire).** Fidelity detectors (drift check,
  off-lane cold detective, cross-check) run on persisted cadence state, sample
  the populations that can actually diverge (remote verdicts, trivial-pass
  lands), and a concluded divergence halts the queue + escalates born-at-L2 —
  a suspected-red land is a "human must look now" condition.

## 2. Sketch of approach

Six leaves (α–ζ), all in `orchestrator/` (one package), each independently
landable; α+β close the live red-main holes, γ closes the phantom-block hole,
δ/ε/ζ harden the surround. Reify-side sibling work (structured offender lines,
branch-scope guard trigger) is a separate reify micro-PRD; ε consumes its
output opportunistically (absence of structured lines degrades to no-op, so
there is no hard cross-project dependency).

## 3. Resolved design decisions

1. **INV-2 shape = auto-sync at dispatch** (operator-ratified 2026-07-22, over
   spec-carried decisions and version-handshake): RemoteRunner checks remote
   DF HEAD vs local DF HEAD (cheap ssh `git rev-parse`), and when stale runs
   `git pull --ff-only` + `uv sync` on the remote checkout, serialized
   per-runner, never while that runner has a verify in flight. Sync failure ⇒
   fail-closed: runner benched (existing quarantine path) + `runner_stale`
   event; dispatch falls back local. Remote *project* checkout main ref is
   mirror-semantics: force-update + event when the best-effort push is non-FF
   (today it silently fails rc=1 forever).
2. **INV-1 scope**: generalizes task 2838 (config-only → full gate) and the
   task-1774 guard: for `DF_VERIFY_ROLE=merge`, *any* trivial-pass resolution
   (not just guard-matched diffs) forces the full gate; empty-`existing_files`
   (clobbered worktree) is treated as evidence-absence ⇒ full gate, never pass.
   The 2823 cache-only main-red gate stays as an extra layer but is no longer
   load-bearing.
3. **INV-3 enforcement points**: (a) verify dispatch — re-check chain validity
   at host-acquisition time (5260's verify would have been discarded before
   burning 43 min; the existing dispatch-time gate keys on a global
   `_has_inflight_verify` flag and never re-checks); (b) `_finalize_inflight`
   adoption — the invariant proper. The head-failure cascade fix (dangling
   successor edge after predecessor re-merge) remains as the prompt-invalidation
   optimization; INV-3 is the correctness backstop if the cascade misses again.
4. **Cross-check stays a trailing detector** (not a pre-adoption gate): gating
   every remote-sole verdict on a local cross-check would double the cost of
   Lever C. Instead: concluded divergence ⇒ immediate merge-queue halt +
   born-at-L2 (suspected red main), and with INV-1/INV-2 in force the
   trivial/stale classes that produced the 3-seconds-too-late case are gone.
5. **INV-4 numbers from evidence, as config**: `same_tip_retry_cap: 1`
   (existing cap ≈3 permitted the observed 2–3-deep thrash chains),
   `same_tip_reroll_idle_slot_only: true`. Rationale recorded: re-roll wins are
   flake re-rolls (38%), so one re-roll captures most of the win mass; as the
   CPU-contention deflake work lands, flake rate — and hence re-roll value —
   drops further; the cap can tighten to 0-with-escalation later.
6. **Foreign-drift disposition (ε)**: when a merge-verify failure carries
   structured offender lines (opt-in grammar, e.g. reify's
   `HARNESS_KLOC_CAP FAIL … file=<path>`), intersect offender paths with the
   task's own three-dot diff; empty intersection ⇒ new disposition
   `foreign_drift` (never `branch_bug`, never auto-requeue) with steward
   guidance naming the foreign paths. No structured lines ⇒ existing behavior
   unchanged.
7. **Event vocabulary**: new event types `verdict_voided`, `runner_stale`,
   `runner_synced`, `trivial_pass_escalated`; existing dashboards/digest read
   them for free via the event census.

## 4. Pre-conditions (G3 — substrate verified on `main 08925d962e`)

All touch points confirmed present by direct read during the forensics:
`RemoteRunner.run_merge_verify` (verify_runner.py), `build_merge_verify_spec`
(empty-spec behavior confirmed), `run_scoped_verification` trivial-pass branch +
task-1774 guard consult + task-2838 config-only escalation (verify.py),
task-2822 cross-check + task-2823 main-red gate (merge_queue.py), frozen-prefix
`two_layer_invariants()` reporting, `_finalize_inflight` / `advance_main` CAS,
head-failure cascade + `speculative_discard` emission, `HostAllocator`
prefer-local/overflow-remote, `ProbePlacement` (relabel-only — exonerated),
`merge_drift.py` in-memory `_drift_land_count`, `merge_shadow.py`
warm-capture-gated sampling. No new substrate is assumed anywhere.

## 5. Out of scope

- Reify-side guard changes (offender-line emission, branch-scope trigger) — the
  sibling reify micro-PRD.
- Stranding remediation, scheduler-pause ergonomics, lane reclamation, EWA trip
  classification — `plans/stranding-remediation-scheduler-ergonomics-prd.md`.
- The merge-skew classifier reference frame — DF task 2869 (in-progress).
- Variable-depth probe placement policy — the deep-verify campaign owns it;
  see seam table.
- Reverting or pausing the speculation probe (`probe_fraction 0.5`) — the
  probe is exonerated (relabel-only) and the campaign continues per standing
  operator directive.

## 6. Cross-PRD relationship + seam owners (G4)

| Seam | Owner | Other party |
|---|---|---|
| Verdict adoption validity (INV-3) | **this PRD** | deep-verify/probe campaign (task 2359 lineage) owns *placement policy*; deeper placement raises void-probability — measured, not fought |
| Remote runner fidelity (INV-1/2) | **this PRD** (amends Lever C's enforcement) | `plans/merge-throughput-multihost-verify-prd.md` owns the runner architecture |
| Structured offender-line grammar | **reify guard** (rule (c) grammar already exists) | DF ε parses defensively; absence ⇒ no-op |
| Disposition taxonomy (`foreign_drift`) | **this PRD** | task 2869 (reference frame) — independent, composable |
| Train formation exclusions | **this PRD** (INV-4 extends to speculation-base selection) | task 1720's coalescer confidence-gate covers GroupMergeRequest formation only |

## 7. Two-way boundary tests (G5 §B sketch)

Pipeline→runner: a merge-role spec that resolves to no commands must come back
as full-gate-run-or-FAIL, never PASS (both LocalRunner and a fake RemoteRunner).
Runner→pipeline: a stale-HEAD fake remote must be benched fail-closed with
`runner_stale`, and a synced one accepted. Chain→adoption: predecessor dies
(fail AND re-merge variants) while successor is (a) mid-verify, (b) **built,
awaiting host** — the untested 5260 state — verdict voided both directions
(PASS and FAIL variants), item re-merged, no task blocked. Re-roll→train: a
same-tip retry is never chosen as spec_base and never coalesced.

## 8. Decomposition plan (one leaf per bullet; signal = user-observable)

- **α — INV-1 fail-closed trivial pass (merge role).** Signal: an infra-only
  (no-`.rs`/`.py`) diff merge-verify runs the full gate (event shows real
  duration/commands), and a clobbered-worktree empty-`existing_files` verify
  FAILS loud instead of 0ms-passing; `trivial_pass_escalated` event emitted.
- **β — INV-2 RemoteRunner auto-sync, fail-closed + mirror-push.** Signal:
  with a deliberately-lagged remote checkout, next dispatch emits
  `runner_stale`→`runner_synced` and the remote executes current-HEAD gate
  code (probe: guard symbol version echoed in verify env/log); sync failure
  benches the runner with the existing quarantine event; non-FF project push
  force-updates + events instead of silent rc=1.
- **γ — INV-3 chain-intact at dispatch + adoption (+ cascade dangling-edge
  fix).** Signal: reproduce the 5260 topology in the boundary harness (head
  fails while successor is built-awaiting-host) — successor's verdict is
  voided (`verdict_voided` names the dead link), it re-merges against real
  main, and the task is never marked blocked by the phantom FAIL; two-layer
  invariant report shows zero violations during the scenario.
- **δ — INV-5 detector hardening.** Signal: drift-check counter survives an
  orchestrator restart (persisted; census shows a first-ever drift event in a
  soak with n≥20 lands); shadow/cold detective samples a trivial-pass and a
  remote-verdict land (coarse suite-level compare when no per-test map);
  concluded cross-check divergence halts the queue + files born-at-L2 before
  any further adoption.
- **ε — foreign-drift disposition.** Signal: a fabricated verify failure whose
  structured offender lines name only files outside the task's diff produces
  disposition `foreign_drift` (event + escalation text names the foreign
  paths), and no auto-requeue fires; the same failure without structured lines
  behaves exactly as today.
- **ζ — INV-4 same-tip re-roll policy.** Signal: in the harness, a same-tip
  requeue after a merge-verify FAIL is accepted once (cap), runs solo
  (never selected as spec_base, never coalesced), and a second same-tip
  requeue is refused into blocked+proposal; config knobs visible in
  `routing_decision`-style event detail.

Dependency shape: α, β, δ, ε, ζ independent; γ independent of all (touches
dispatch/finalize only). No cross-project hard deps.

## 9. Open (tactical) questions

- β: sync cadence guard (HEAD-compare per dispatch vs TTL) — architect's call;
  either satisfies INV-2.
- δ: coarse-compare threshold for map-less lands (suite exit vs per-command) —
  implementer's call after reading merge_shadow.
- ζ: whether `same_tip_reroll_idle_slot_only` defaults on at first land or
  after one soak window — flag to operator at review.
