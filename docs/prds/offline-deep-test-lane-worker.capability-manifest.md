# Capability manifest — `offline-deep-test-lane-worker` (Part B)

Mechanizes G3 + G6 per leaf for `docs/prds/offline-deep-test-lane-worker.md`. Each binding ties a
leaf's asserted capability to **evidence** (grep/command/file). Any **FAIL** binding blocks queueing
until resolved. Verified against dark-factory `main @ 0f9aa4a186` and reify `main @ 0113758b11`,
2026-07-01.

**Domain notes.** Part B is **orchestrator control-flow / config-deploy wiring**, not DSL or
numeric-result production. The reify numeric-floor and field-population sentinels are **N/A by
construction** (Part B asserts no numeric bound and reads no result field — the one numeric floor in
this effort is Part A's A3 gate smoke). The live G6 risks here are **anti-orphan wiring** (every
mechanism reaches a consumer on main), the **dedup negative-assertion** (a "no duplicate task" claim
must be backed by a real fingerprint-dedup mechanism), and the **cross-project flip edge** (the `→A4`
consumer must be a real `add_dependency`, not prose).

---

## β1 — `on_post_merge` trigger fan-out

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | The fan-out seam is live on main: `merge_queue.py:10569-10580` invokes `self._on_merge_landed(task_id, base_sha, head_sha)` **fail-open** at the landing moment; `harness.py:4953` wires `on_merge_landed=self._note_merge_all`; `harness.py:4973 _note_merge_all` fans out to `coord.note_merge(...)` (`harness.py:5005`). Adding a notifiee is symmetric with the existing coordinator loop. |
| **Anti-orphan / wired** | **PASS** | Consumed by β2 (the worker's `dirty` flag). Not a producer with no consumer. |
| **Negative-assertion (fail-open)** | **PASS (by ζ/B1)** | The claim "a raising notifiee never fails a merge" binds to the existing fail-open swallow (`merge_queue.py:10580` "on_merge_landed hook raised … ignoring (fail-open)"); ζ/B1 injects a raising notifiee and asserts the merge still lands. |
| Numeric floor / field-population | **N/A** | Trigger plumbing asserts no numeric bound and reads no result field. |

## β2 — Singleton lane worker (single-flight / coalesce / always-from-head)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | `git rev-parse main` snapshot + subprocess invocation of Part A's `run-offline-deep.sh` are standard; the worker is launched by `harness.py` (which already owns the merge worker + coordinators). Lockfile-singleton is a local addition. Reuses `workflow.py:402 compute_preexisting_main_break_fingerprint` (present). |
| **Anti-orphan / wired** | **PASS** | Consumes β1 (trigger) and δ (warm worktree); its runs feed β3 + operator logs. A real in-process caller, not a dangling module. |
| **Runtime entry (cross-project) exists** | **PASS (precondition-tracked)** | The worker's subprocess entry `scripts/run-offline-deep.sh` + `DF_VERIFY_ROLE=offline` are **Part A deliverables (A5/A2)**, decompose-ready but not yet on reify `main`. Bound as a cross-project precondition dep (ζ → `reify:A5`/`reify:A2`) wired at decompose — a named upstream producer in the transitive closure, not an absent capability. |
| **Negative-assertion (single-flight)** | **PASS (by β2/ζ-B2)** | "Exactly one run; coalesced re-run at newest head; second instance refuses" binds to the lockfile + dirty-flag loop; ζ/B2 observes exactly one coalesced re-run under a 3-advance burst. |
| Numeric floor / field-population | **N/A** | Control-flow; no numeric bound, no result field. |

## β3 — Dedup'd fix-task spawn + staged escalation

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | Fix-task filing: `workflow.py:8166 _post_submit_tasks` POSTs `submit_task` (used by `_spawn_main_health_fix_task`, `workflow.py:5444`). Escalation: orchestrator builds `escalation.models.Escalation` and enqueues (`from escalation.models import Escalation` throughout `harness.py`/`workflow.py`/`deterministic_runner.py`). Fingerprint model: `workflow.py:402 compute_preexisting_main_break_fingerprint` (present; consumed at `workflow.py:4715`). |
| **DAG-direction (anti-inversion)** | **PASS** | The failing-test-set key is a **re-keying** of an existing upstream helper (DB3), not a dependency on a downstream task. |
| **Negative-assertion (dedup, the load-bearing claim)** | **PASS (by β3/ζ-B5, mechanism real)** | "Same failing set on a later advance spawns no duplicate" binds to a real dedup mechanism (`open_fix_tasks[fingerprint]` update-or-file, C3), **not** a promise. G6-branch-4 satisfied: ζ/B5 authors the repeat-red and observes the existing task updated + zero new task/escalation. The `main_sha`-keying trap (DB3) is flagged so decompose does not re-introduce the flood. |
| **Negative-assertion (flake filter)** | **PASS (by ζ-B6)** | "Fail-then-pass ⇒ no task" binds to the confirmation re-run (C3); ζ/B6 observes the "intermittent nondeterminism" log and zero task. |
| Numeric floor / field-population | **N/A** | Failing-test IDs are opaque strings; no numeric bound, no result field sampled. |

## δ — Second persistent warm worktree `_offline-deep`

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | Phase-1 machinery is live on main: `git_ops.py:117 PERSISTENT_MERGE_WORKTREE_NAME='_merge-verify'` (task 1692, create-once-at-fixed-path), `WarmLanePool` (`warm_lane_pool.py`), warm-lane knobs (`config.py:864 warm_lane_pool`, `warm_lane_pool_size`, `warm_lane_base_target_dir`, `warm_lane_disk_guard`). warmer-builds Phase 1 is LIVE (reify task ε #4663). |
| **Extent (anti-`producer-extent-short`)** | **PASS with noted generalization** | The fixed-name constant `PERSISTENT_MERGE_WORKTREE_NAME` is single-valued today; the 2nd instantiation parameterizes it (a **new** fixed name `_offline-deep`) — a bounded generalization of existing machinery, **not** a fiction. δ owns exactly this extent; no downstream task is relied on. |
| **Anti-orphan / wired** | **PASS** | Consumed by β2 (runs the suite in it). |
| **Invariant binding (C5 / warm-lane §11)** | **PASS (by ζ-B8)** | "single-consumer of its own `target/`, never shared/overlaid, self-warming" binds to warm-lane §11; ζ/B8 observes run-2 fingerprint-pass timing and that the merge lane's `target/` is untouched. |
| Numeric floor / field-population | **N/A** | Worktree/build machinery; no numeric bound, no result field. |

## ζ — Lane-live integration gate

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | Pure integration of β1/β2/β3/δ against a live reify checkout with Part A landed; no new production substrate. |
| **Executable integration (anti-tabulation, C-as-integration-gate)** | **PASS — this leaf's whole job** | The §Boundary-test sketch (B1–B8) is **run**, not tabulated: it stands up the real trigger→worker→warm-worktree→failure-handling chain and asserts the observed behavior, including the load-bearing **never-a-gate** invariant (B3) and dedup (B5). Blocks the batch if any boundary scenario fails. |
| **Cross-project precondition wired** | **PASS (tracked)** | ζ deps cross-project on `reify:A5`/`reify:A2` (runner + role on reify `main`) — wired at decompose or left as a documented follow-up if Part A ids are not yet filed (§Sequencing). |
| Numeric floor / field-population | **N/A** | Integration assertions are structural (run started, queue unblocked, task/escalation present). |

## ε1 — Flip deploy script

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | The flip target is real: `/home/leo/src/reify/orchestrator.yaml:148 verify_env:` exists (reify-repo-tracked; `git ls-files` confirms), and the DF orchestrator deep-merges the per-project config file (`config.py:122-125`). Setting `REIFY_GATE_EXCLUDE_HEAVY: "1"` there is a config edit, no reify code change (the knob is *read* by Part A's A4 in `verify.sh`). |
| **Anti-orphan / wired** | **PASS** | Consumed by ε2 (the deterministic task's `before_done` references it). The committed-executable-script prerequisite for a deterministic task is exactly why ε1 is a separate upstream leaf (CLAUDE.md: `before_done.script` "must exist & be executable" at `submit_task`). |
| Numeric floor / field-population | **N/A** | Config edit; no numeric bound, no result field. |

## ε2 — `flip-gate-exclude-heavy` (deterministic config deploy)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS** | `task_kind='deterministic'` + `metadata.before_done` is a first-class `submit_task` capability (CLAUDE.md "Deterministic task kind"; `deterministic_runner.py` present, imports `escalation.models`). The "auto-deploy" preset (`before_done` present, `always_escalates=false`) is documented and validated. |
| **Anti-orphan / wired (the flip's consumer)** | **PASS** | The consumer of the `1` value is Part A's A4 knob in reify `verify.sh`; ε2 → `reify:A4` is a real cross-project `add_dependency` edge (§6/§Sequencing), not prose. The seam has a named owner (dark-factory pulls; reify owns the seam) — not an unclaimed orphan. |
| **DAG-direction (anti-inversion)** | **PASS** | ε2 depends on ζ (local, upstream) **and** `reify:A4` (cross-project, upstream) — both are upstream producers, never downstream. The ordering invariant (C6) is enforced by these two edges, so the flip cannot fire before the knob exists and the lane is live. |
| **Negative-assertion (reversible + gated)** | **PASS (by ζ-B9)** | "Gate flips to `not (heavy)` only after both edges; reverting restores the full gate" binds to A4's strict-`1` semantics (validated in Part A's manifest) + the dependency gate; ζ/B9 asserts the flip did not fire early and reverts cleanly. |
| Numeric floor / field-population | **N/A** | Config deploy; no numeric bound, no result field. |

---

## Summary

| Leaf | Blocking verdict |
|---|---|
| β1 trigger fan-out | **PASS** (fail-open seam live at `merge_queue.py:10575`) |
| β2 singleton worker | **PASS** (runtime entry = Part A A5/A2, cross-project precondition tracked) |
| β3 dedup fix-spawn + escalation | **PASS** (real fingerprint-dedup mechanism; `main_sha`-key trap flagged) |
| δ 2nd warm worktree | **PASS** (Phase-1 machinery live; bounded name-parameterization generalization) |
| ζ lane-live integration gate | **PASS** (executable boundary tests; never-a-gate invariant asserted) |
| ε1 flip deploy script | **PASS** (flip target `reify/orchestrator.yaml:148 verify_env` real) |
| ε2 deterministic flip deploy | **PASS** (cross-project `→A4` edge = real `add_dependency`, upstream) |

**No FAIL bindings.** The batch is clear to queue once the decompose session (a) confirms these
bindings executably per-leaf and (b) wires the two cross-project edges — **ε2 → `reify:A4`** (the flip)
and **ζ → `reify:A5`/`reify:A2`** (the runner/role precondition) — or records them as documented
follow-ups if Part A's task ids are not yet filed (§Sequencing). The single load-bearing G6 risk (the
dedup "no duplicate" negative-assertion) is neutralized by keying on the failing-test-set signature, not
`main_sha` (DB3); the manifest flags the `main_sha`-key flood trap so decompose does not re-introduce it.
The hard "**never a gate**" invariant (D1/§11) is bound executably to ζ's B3 boundary scenario, not left
as a tabulated promise.
