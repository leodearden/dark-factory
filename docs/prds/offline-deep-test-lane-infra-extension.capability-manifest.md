# Capability manifest — `offline-deep-test-lane-infra-extension`

Mechanizes G3 + G6 per leaf for `docs/prds/offline-deep-test-lane-infra-extension.md`. Each binding
ties a leaf's asserted capability to **evidence** (grep/command/task-producer). Any **FAIL** binding
blocks queueing until resolved. Verified against dark-factory `main @ 053ef4f447` and reify
`main @ 64b3992b62`, 2026-07-01.

**Domain notes.** This extension is **orchestrator control-flow / config-deploy wiring** that *reuses*
Part B's proven engine (single-flight worker, warm worktree, failing-test-set dedup, never-a-gate) and
adds only one worker invocation, a lane-live gate, and a config flip. It asserts **no new numeric
premise** (PRD §1) — every leaf signal is structural (run records, task-tree/escalation appearance,
`=== Summary ===`-count diff). The reify numeric-floor and result-field-population sentinels are **N/A
by construction**. The live G6 risks are the same three as Part B: **anti-orphan wiring** (each infra
mechanism reaches a consumer on main), the **dedup negative-assertion** (reused wholesale from β3 —
content-agnostic fingerprint), and the **cross-project edges** (the reify H9/H1/H3 consumers must be
real `add_dependency` qualified refs, not prose).

---

## IE1 — Worker infra invocation (β2 also runs reify `run_all --scope host-infra`)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (producer upstream)** | Extends β2's run loop — `producer:task-1953` (β2, `in-progress`, new module `orchestrator/src/orchestrator/offline_lane.py`), wired upstream via `IE1 → 1953`. A second subprocess invocation alongside `run-offline-deep.sh` is symmetric with β2's existing single-flight loop. |
| **Runtime entry (cross-project) exists** | **PASS (producer upstream)** | reify's `run_all --scope host-infra` runner = **H9** `producer:reify:4929` (`pending`), wired upstream via external dep `IE1 → reify:4929` → routes to `metadata.external_deps`, gated at dispatch by `get_external_statuses`. H9 self-acquires the H8 flock (IE-D2), so no separate DF flock invocation — a named upstream producer, not an absent capability. |
| **Dedup + escalate_info reuse** | **PASS (producer upstream)** | Red infra results route through β3's confirmation-re-run + failing-test-set fingerprint dedup + `escalate_info` = `producer:task-1954` (β3, `pending`), wired upstream via `IE1 → 1954`. The fingerprint is **content-agnostic** (infra test-file names fingerprint exactly like numeric test IDs), so no new dedup mechanism — reused wholesale. |
| **Warm worktree (IE-D4)** | **PASS (on main)** | Reuses δ's `_offline-deep` worktree = `producer:task-1952` (δ, **`done`**). Infra `.sh`/synth-workspace-cargo tests never touch `_offline-deep/target/`, so the single-consumer-of-`target/` invariant (C5) holds. |
| **Negative-assertion (never-a-gate)** | **PASS (by IE2/B3-analog)** | "merge queue untouched during an in-flight infra run" binds to the idle-class out-of-band `offline` role inherited from Part A (β1 `#1951` done) + β2's out-of-band loop; asserted executably at IE2. |
| Numeric floor / field-population | **N/A** | Control-flow; asserts no numeric bound, reads no result field (failing-test IDs are opaque strings). |

## IE2 — Infra lane-live integration gate (ζ-analog)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (producer upstream)** | Pure integration of IE1's infra sub-run against a live reify checkout; no new production module. Consumes IE1 = `producer:task-IE1`, wired upstream via `IE2 → IE1`. |
| **Real runner over real classified set** | **PASS (producers upstream)** | Runs the **real** H9 runner (`producer:reify:4929`, `pending`) over the **real** H1-classified host-exclusive set (`producer:reify:4921` = H1 classification manifest, `in-progress`). Both wired upstream via external deps `IE2 → {reify:4929, reify:4921}`. |
| **Executable integration (anti-tabulation, C-as-integration-gate)** | **PASS — this leaf's whole job** | The boundary scenarios (from-head trigger; **never-a-gate**; injected red → deduped fix task + `escalate_info` without touching the merge queue; fail-then-pass → "intermittent nondeterminism" log, nothing filed) are **run**, not tabulated. Blocks the batch if any scenario fails. Binds the load-bearing never-a-gate invariant (C7/§6) executably. |
| Numeric floor / field-population | **N/A** | Integration assertions are structural (run started, queue unblocked, task/escalation present). |

## IE3 — Infra flip deploy script

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (on main)** | Flip target is real: reify `orchestrator.yaml:148 verify_env:` exists (reify-repo-tracked), and the DF orchestrator deep-merges the per-project config (`config.py:1500 verify_env`, `config.py:2177 effective_verify_env`; `verify.py:2141 _resolve_verify_env` reads it). Setting `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA: "1"` there is a config edit — no reify code change (the knob is *read* by H3's `run_all` in reify). Exactly the substrate ε1 (`#1956`) uses for the numeric knob at the same `:148` block. |
| **Anti-orphan / wired** | **PASS (producer downstream is the filer, expected)** | Consumed by IE4 (the filer references this script as the deterministic task's `before_done.script`). The committed-executable-script prerequisite is precisely why IE3 is a separate upstream leaf: `before_done.script` must exist + be `os.X_OK` at `submit_task` time (deterministic guard `tools.py:2520`, `deterministic_task_guard.py:192-203`), so it must land on `main` before IE4 files the deterministic deploy. |
| **Leaf signal is self-produced** | **PASS** | Signal = "committed + executable (mode 100755); `--check` prints the one-line diff it would apply". Both are produced by IE3's own deliverable; no downstream-owned capability. |
| Numeric floor / field-population | **N/A** | Config-edit script; no numeric bound, no result field. |

## IE4 — Infra flip deterministic filer (ε2-filer-analog) — **the batch leaf**

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (on main)** | `task_kind='deterministic'` + `metadata.before_done` + `always_escalates=false` ("auto-deploy" preset) is a first-class `submit_task` capability (CLAUDE.md "Deterministic task kind"; `deterministic_runner.py` present). The filer pattern is proven by ε2-filer `#1957`. |
| **before_done.script exists at file-time** | **PASS (producer upstream — the reason IE4 is a filer, not a deterministic task)** | IE4 files the deterministic deploy **only after IE3 lands** its script on `main` (`producer:task-IE3`, wired upstream via `IE4 → IE3`). Filing it now would be rejected by the deterministic guard (script absent). This is the ε1/ε2 constraint, mirrored exactly. |
| **Flip consumer (the `1` value)** | **PASS (producer upstream)** | The consumer of the `1` value is reify's off-by-default **H3** knob `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` = `producer:reify:4925` (`pending`), wired upstream via external dep `IE4 → reify:4925`. Real `add_dependency` qualified ref, not prose. |
| **DAG-direction (anti-inversion) + additive-flip invariant** | **PASS** | IE4 depends on IE3 (local, upstream) **and** reify:4925 (cross-project, upstream). The deterministic deploy task IE4 *files* carries deps `[IE2 (lane-live), reify:4925 (H3)]`, so the flip fires **iff both** IE2 green **and** H3 on reify `main` (§6 additive-flip invariant) — never a premature exclusion, no coverage gap. |
| **Leaf signal is task-tree observable** | **PASS** | Signal = "the flip task exists `pending` with those deps" — observed via `get_tasks`, exactly as ε2-filer `#1957`'s signal ("ε2 exists as a pending deterministic task…"). The ultimate `=== Summary ===`-count-drop surface belongs to the deploy task IE4 files (a structural identity: excluding the H1-classified set drops the count by exactly that count), not to the filer leaf. |
| Numeric floor / field-population | **N/A** | Config deploy; the count-diff is a structural identity backed by H1's classification (`reify:4921`, upstream via IE2), not a tuned numeric bound. |

---

## Summary

| Leaf | Blocking verdict |
|---|---|
| IE1 worker infra invocation | **PASS** (extends β2 `#1953`; runtime entry = reify H9 `#4929`, dedup = β3 `#1954`, worktree = δ `#1952` done) |
| IE2 infra lane-live gate | **PASS** (executable boundary tests; never-a-gate asserted; real H9/H1 over real classified set) |
| IE3 infra flip deploy script | **PASS** (flip target `reify/orchestrator.yaml:148 verify_env` real; mirrors ε1 `#1956`) |
| IE4 infra flip deterministic filer | **PASS** (filer pattern proven by ε2-filer `#1957`; `→ reify:4925` = real `add_dependency`, upstream) |

**No FAIL bindings.** The batch is clear to queue. All substrate is either on `main` today
(δ `#1952`, `verify_env` deep-merge, deterministic guard) or a named upstream producer in the
transitive dependency closure (β2 `#1953`, β3 `#1954`, reify H9 `#4929`, H1 `#4921`, H3 `#4925`),
wired as real `add_dependency` edges at decompose. The single load-bearing G6 risk (the dedup "no
duplicate" negative-assertion) is **inherited neutralized** from β3 — keyed on the failing-test-set
signature (content-agnostic, so infra failures dedup exactly like numeric ones), never `main_sha`
(the DB3 flood trap). The hard **never-a-gate** invariant (C7/§6) is bound executably to IE2's
boundary scenarios, not left as a tabulated promise. IE4 is a **filer** (not a directly-filed
deterministic task) because the deterministic guard validates `before_done.script` existence at
`submit_task` time and IE3's script does not exist on `main` at decompose — the ε1/ε2 constraint,
mirrored exactly (DF `#1957`).
