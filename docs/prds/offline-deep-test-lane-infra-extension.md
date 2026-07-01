# PRD — Offline deep-test lane, infra-residue extension (dark-factory)

**Status:** draft (2026-07-01) — for `/prd` check + decompose.
**Slug:** `offline-deep-test-lane-infra-extension`
**Extends:** `docs/prds/offline-deep-test-lane-worker.md` (Part B — the *numeric* cold-lane worker, filed as DF **#1951–1957**).
**Consumes (reify Part A primitives):** `/home/leo/src/reify/docs/prds/run-all-host-infra-partition.md` (reify **#4921–4929**, leaves H1–H9).

This extension makes the **same** post-merge cold lane Part B built also run reify's **host-global-unsafe infra-test
residue** — reusing Part B's trigger / single-flight worker / dedup-fix-task / warm-worktree engine wholesale, adding
only one worker invocation, a lane-live gate, and a flip. It is the DF-side consumer of reify's off-by-default infra
seam — the exact numeric ζ/ε pattern (DF #1955/#1956/#1957) applied to the infra residue.

---

## 0. Goal & user-observable surface (G1)

**Goal.** Run reify's `run_all.sh` **host-exclusive** bucket (the ~4 files that do real cgroup burn / real cgroup
delegation / real reflink-FS + cargo) **post-merge, off the verify hot path**, on the existing single-flight
always-from-head cadence, with autonomous failure handling — then flip reify's off-by-default
`REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` knob so that residue stops running in every per-task/per-merge verify. The gate
gets faster; coverage of the residue **moves** to the idle cold lane (never lost). **Never a gate** (inherited C7).

**Mechanisms introduced, each with a named consumer (no orphan producers):**

| Mechanism | Consumer |
|---|---|
| Worker infra invocation — β2 also runs reify `run_all --scope host-infra` (reify **H9 #4929**) at snapshot head | reify's host-exclusive infra tests get executed off-gate; failures feed β3's dedup |
| Infra lane-live integration gate (ζ-analog) | the infra flip (IE4) depends on it green |
| Infra flip deploy script + deterministic auto-deploy | the reify merge/task gate — pulls reify's `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` (H3 **#4925**) to `1`; user surface = a faster hot-path `run_all` |

**User-observable surface when this lands:**
- On each `main` advance the worker runs **both** `run-offline-deep.sh` (numeric, Part B) **and**
  `run_all --scope host-infra` (infra) at the same snapshot head; a run record shows both sub-runs' head SHA + pass/fail.
- A **confirmed** red host-exclusive infra test produces a **normal `pending` fix task** (failing test-file names +
  suspect range in `metadata`) **and** an `escalate_info` — via β3's existing failing-test-set dedup — **never** a
  merge block, **never** the B3 red-main path.
- Merges keep flowing during an in-flight infra run (never-a-gate).
- Once live, the flip lands and reify `DF_VERIFY_ROLE=merge ./scripts/verify.sh` (its `run_all` step) runs the full
  suite **minus** the host-exclusive set — the `=== Summary ===` count drops by exactly the host-exclusive count;
  reversible in one config line.

## 1. Premise (G6)

reify's `run-all-host-infra-partition` PRD (#4921–4929) classifies the `run_all.sh` residue as host-global-unsafe
(false-REDs or generates host load under concurrent verify — the class that has ambushed unrelated green tasks'
merge gates, e.g. ROW4-1 #4656, host-baked-cap #4901). reify ships the primitives this extension consumes:
**H1 #4921** (classification manifest + drift-guard), **H2 #4924** (concurrent hermetic pool), **H3 #4925**
(off-by-default `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` seam), **H8 #4923** (Lane-X host-exclusive flock primitive),
**H9 #4929** (`run_all --scope host-infra` runner, which **self-acquires** the H8 flock). This PRD adds **no new
numeric premise** — its leaf signals are structural (run records, task-tree/escalation appearance, `Summary`-count
diff). The generic cold-lane engine (single-flight, from-head, dedup, never-a-gate) is already proven by Part B.

## 2. Ratified decisions

- **IE-D1 — reuse Part B's engine, NOT a new lane.** β2's loop invokes reify H9 at the same run-start snapshot head,
  in the same `on_post_merge` cadence; infra failures route through β3's **failing-test-set fingerprint** dedup (the
  fingerprint is content-agnostic — infra test-file names fingerprint exactly like numeric test IDs). One coalesced
  cadence, two sub-runs (numeric + infra).
- **IE-D2 — reify H9 self-acquires the H8 flock**, so the worker just invokes H9; **no separate DF flock invocation
  is required.** (This subsumes the reify PRD's deferred "Part B flock-invocation → H8" edge — H8 is invoked
  transitively via H9. A single coarse host lock wrapping numeric+infra together is an *optional* decompose choice,
  not required — the numeric run and the infra run are already serialized within β2's single-flight loop.)
- **IE-D3 — the flip mirrors ε1/ε2** (a committed deploy script THEN a deterministic auto-deploy task — because
  fused-memory's deterministic guard validates `before_done.script` existence + executability at `submit_task` time,
  so the script must land first). Sets `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA:"1"` in reify `orchestrator.yaml verify_env`;
  deps `reify:4925` (H3) + the infra-lane-live gate.
- **IE-D4 — where the infra run executes.** Default: **reuse δ's `_offline-deep` worktree** (its solver+eval warm
  `target/` is simply unused by the infra `.sh` / synth-workspace-cargo tests; what the infra set needs is the
  checkout + a **cgroup-delegated + reflink-capable host**, which the cold-lane box provides). A lighter dedicated
  context is the alternative — revisit only if reusing `_offline-deep` violates its single-consumer-of-`target/`
  invariant (it should not: the infra tests never touch `_offline-deep/target/`). *(decompose Q — IE1.)*
- **IE-D5 — never a gate (inherit C7).** The infra run is out-of-band, idle-class, single-flight; a red infra result
  files a normal queued fix task and never blocks/halts/delays the merge queue.

## 3. Pre-conditions / substrate (G3)

- **Part B engine:** β1 **#1951** (trigger, done), δ **#1952** (`_offline-deep`, done), β2 **#1953** (worker,
  in-progress), β3 **#1954** (dedup fix-task, pending). IE1 depends on β2 + β3.
- **reify Part A infra primitives (filed, landing):** H9 **#4929** (`run_all --scope host-infra` runner — the worker's
  runtime entry; must be on reify `main` to run) → cross-project dep for IE1/IE2; H1 **#4921** (classification —
  defines the host-exclusive set the runner executes); H3 **#4925** (the flip knob) → cross-project dep for IE4;
  H8 **#4923** (flock, invoked transitively by H9).
- **Flip target:** reify `orchestrator.yaml verify_env` (git-tracked; DF orchestrator deep-merges per project), same
  substrate ε1/ε2 use for the numeric knob.

## 4. Cross-PRD / cross-repo seam (G4)

Same seam class as the numeric ζ/ε and cpu-governance α/β/γ↔ζ: **reify ships the primitives, dark-factory wires the
consumer.** Ownership is unambiguous. Cross-project dependency edges (real `add_dependency` qualified refs):

| Edge | Meaning |
|---|---|
| IE1 → `reify:4929` (H9) | the worker's infra runtime entry must be on reify `main` |
| IE2 → `reify:4929`, `reify:4921` (H9, H1) | the lane-live gate runs the real runner over the real classified set |
| IE4 → `reify:4925` (H3) | the flip pulls reify's off-by-default knob to `1` (fires iff H3 on `main` AND IE2 green) |

## 5. Decomposition plan (leaf tasks — each names a user-observable signal, G2)

- **IE1 — worker infra invocation.** Extend `offline_lane.py`'s β2 run so that, at the run-start snapshot head, it
  **also** invokes reify's `run_all --scope host-infra` (reify:4929) — a second sub-run alongside `run-offline-deep.sh`
  — routing any red through β3's confirmation-re-run + failing-test-set dedup + `escalate_info`. Reuse δ's
  `_offline-deep` worktree (IE-D4). *Signal:* on a `main` advance the worker executes **both** sub-runs at the same
  head (visible in the run log); an injected red in a host-exclusive infra test yields a **deduped** `pending` fix
  task (`get_tasks`, failing file-names + suspect range) + an `escalate_info`, with the merge queue **untouched**.
  *Modules:* `orchestrator/src/orchestrator/offline_lane.py` (+ `harness.py` if wiring). *Deps:* **#1953** (β2),
  **#1954** (β3); cross-project **`reify:4929`**.
- **IE2 — infra lane-live integration gate (ζ-analog).** Executable boundary test of the infra sub-run end-to-end:
  triggers from-head; **never a gate** (a normal advance merges unblocked during an in-flight infra run); injected red
  → deduped fix task + `escalate_info` without touching the merge queue; fail-then-pass → "intermittent
  nondeterminism" log, nothing filed. *Signal:* the boundary scenarios pass against a live reify checkout with H1/H9
  landed. *Modules:* integration (no new production module). *Deps:* **IE1**; cross-project **`reify:4929`**,
  **`reify:4921`**.
- **IE3 — infra flip deploy script.** New `scripts/deploy/flip-reify-run-all-exclude-host-infra.sh` (committed
  **executable**, mode 100755): idempotently set `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA:"1"` in reify
  `orchestrator.yaml verify_env`, commit in the reify repo, signal orchestrator reload; `--check` prints the one-line
  diff and exits 0 on an unflipped config. *Signal:* committed + executable; `--check` prints the diff it would apply.
  *Modules:* `scripts/deploy/flip-reify-run-all-exclude-host-infra.sh` (new). *Deps:* **IE2**.
- **IE4 — infra flip deterministic filer (ε2-filer-analog).** After IE3 lands, file the deterministic auto-deploy
  task (`task_kind='deterministic'`, `metadata.before_done={script: IE3, timeout_secs, target_unit:…}`,
  `always_escalates=false`) with deps **IE2** (lane-live) + **`reify:4925`** (H3 knob); include an idempotency guard
  (search before filing). *Signal:* the flip task exists `pending` with those deps; on dispatch reify's `run_all`
  hot-path excludes the host-exclusive set (the `=== Summary ===` count drops by exactly that count); reverting
  restores the full suite. *Modules:* a small committed note + fused-memory MCP orchestration. *Deps:* **IE3**.

**Suggested edges:** IE2→IE1; IE3→IE2; IE4→IE3. **Cross-project (wire at decompose):** IE1→`reify:4929`;
IE2→{`reify:4929`,`reify:4921`}; IE4→`reify:4925`. **DF-internal preconditions:** IE1→{#1953, #1954}.

## 6. Invariants / do-nots

- **Never a gate** (C7, inherited). No infra-extension mechanism blocks/halts/delays the merge queue.
- **Dedup on the failing-test-set signature** (reuse β3; never re-key on `main_sha` — the flood trap).
- **Rely on H9's self-flock** (IE-D2) — do not add a redundant DF-side flock around the infra sub-run.
- **Additive flip.** reify H3 defaults `0`; IE4's flip fires iff **both** IE2 (infra lane live) **and** `reify:4925`
  (H3 on reify `main`) are `done` — no coverage gap, no premature exclusion.
- **The fix goes through the gate** — a normal queued fix task, never the B3 red-main path, never an unattended
  `main` edit.

## 7. Out of scope

- The numeric heavy lane (β1–ε2, DF #1951–1957) — this extension only *reuses* its engine.
- reify's infra primitives themselves (H1–H9) — reify's Part A (#4921–4929).
- Any new single-flight/warm-worktree machinery — reused from Part B.

## 8. Open (tactical) questions

- **IE1 sequencing:** modify β2 (#1953, in-progress) directly vs a follow-up after β2 lands. Suggested: land β2
  first, then IE1 as its dependent (deps already encode this).
- **IE-D4 worktree:** reuse `_offline-deep` (default) vs a lighter cgroup-delegated+reflink context.
- **Combined vs split host lock (IE-D2):** H9 self-flock only (default) vs a coarse lock wrapping numeric+infra.
- **N-advances-red-before-`escalate_blocker`:** reuse β3's `N` or an infra-specific value.
- **IE4 `target_unit` / reload mechanic:** mirror ε2's decision (in-place reload vs detached self-restart).
