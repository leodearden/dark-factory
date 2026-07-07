# PRD: laptop warm verify worktree (flock-guarded) + tie remote-verify lifetime to its dispatch connection

**Status:** active · **Milestone:** reify multi-host verify warmth · **Date:** 2026-07-07
**Approach:** B + H (contract + two-way boundary tests) — high-stakes correctness seam.
**Repo:** `/home/leo/src/dark-factory` · package `orchestrator/src/orchestrator/`.
**fused-memory:** `project_id="dark_factory"`, `project_root="/home/leo/src/dark-factory"`.

---

## 1. Goal

Make the reify **laptop** (`leo-laptop`, a REMOTE post-merge verify runner dispatched over
SSH from the workstation orchestrator `leo-MS-7C35`) a *safe warm* verify host. Two coupled
changes:

- **Change A — flock-guarded persistent warm worktree.** The laptop verify reuses a fixed
  `<worktree_base>/_merge-verify` worktree (retained `target/` across merges) instead of a
  cold ephemeral `_merge-<uuid>`, guarded by a laptop-side `flock` that **waits a bounded
  interval then escalates on contention — it never falls back to a cold worktree**. A second
  verify per host must never exist *silently*.
- **Change B — orphan lifecycle.** The remote verify process's lifetime is **tied to its
  dispatch connection**: when its `VerifyResult` becomes undeliverable (orchestrator killed
  or SSH dropped), it terminates itself **and its whole build subtree** instead of surviving
  as a `setsid` orphan.

They are complementary defense-in-depth: **B prevents orphans; A detects any residual
concurrency.** With B working, A's escalation should essentially never fire — and if it does,
something is genuinely wrong (which is the point).

### User-observable outcome (what a user/operator sees if this lands)

1. Consecutive laptop verifies reuse `_merge-verify` with a **retained, non-empty `target/`**;
   warm build time drops after warm-up (the user's warm-lane benchmarks show ~2× vs
   from-scratch even with a warm sccache — see §3).
2. Two concurrent `verify-merge` on the laptop **serialize** under `.merge_verify.lock`; the
   second, after a bounded wait, files a **born-at-L2 escalation** naming the host + the
   holder/waiter pgids, **blocks the merge**, and does **not** mutate the live build's tree or
   spawn an ephemeral worktree.
3. Killing the orchestrator **or** dropping the SSH connection mid-build leaves **no lingering
   `rustc`/`cargo` subtree** on the laptop within `T` seconds.
4. `cancel-verify --request-id X` still tree-kills the full descendant tree (unchanged
   contract).

---

## 2. Background

Dark-factory runs reify's post-merge verify on two hosts: workstation (orchestrator + local
verify trust anchor) and laptop (REMOTE runner, `verify_runners:` in
`/home/leo/src/reify/orchestrator.yaml`; laptop config
`/home/leo/.config/orchestrator/reify-laptop.yaml`). A `/deb` investigation (memory:
`project_reify_multihost_verify_warmth_2026_07_07`) found the laptop cold-rebuilt reify
(~1–2 h) on every verify. The immediate cause — the laptop sccache cache capped at 20 GB —
**has already been fixed** (bumped to 100 GB) and is **out of scope here**.

This PRD covers the two remaining, coupled changes the user decided on. The earlier
investigation tentatively proposed "flock **with ephemeral fallback**"; the user has since
**overridden that** to *escalate-not-fallback* (see §4, Decision A3, and the rejected
alternative). Where this PRD and that memory disagree, this PRD is authoritative.

---

## 3. Why a warm worktree matters even with a warm sccache (premise)

sccache caches *compilation* but not *linking*. reify links a large number of eval test
binaries against the full dep graph on every merge; that link cost is paid in full on every
cold-`target/` verify. The user's warm-lane benchmarks show a warm-lane build is **~2× faster
than a from-scratch build even with a warm sccache**, because the warm lane retains unchanged
linked artifacts. A persistent warm `target/` is therefore a recurring ~2× win on every
verify. (The merge gate runs `verify.sh --profile both` — debug full-workspace + release
narrow eval set — into one `target/` holding both `debug/` and `release/`, so one persistent
worktree warms both profiles for the serial head.)

---

## 4. Resolved design decisions (do not relitigate)

### Change A — persistent warm worktree + flock, escalate-not-fallback

- **A1 — Enable the persistent worktree (deploy step).** Set `persistent_merge_worktree: true`
  in the laptop config `/home/leo/.config/orchestrator/reify-laptop.yaml`. The code path
  already supports it: with the knob on and the safety valve not due,
  `git_ops.acquire_host_verify_worktree` (git_ops.py **5522–5572**) routes to
  `reset_persistent_merge_worktree` (git_ops.py **5574–5647**: create-once /
  `git reset --hard` + `_clean_lane_retaining_artifacts`, self-heals a stale dir) instead of
  the ephemeral `_create_merge_worktree`.

- **A2 — Laptop-side exclusive `flock` over the persistent-worktree span.** In the laptop CLI
  `verify_merge` (cli.py **315–398**), acquire `fcntl.flock(LOCK_EX)` on a fixed path
  `<worktree_base>/.merge_verify.lock` and hold it for the **entire** persistent-worktree
  verify: from the `git reset --hard` + clean (worktree acquire, git_ops.py 5522+) through the
  whole build to `cleanup_merge_worktree`. The lock wraps the `_run()` span at cli.py
  382–386. **The lock guards the persistent branch only** — an ephemeral/safety-valve run in a
  unique `_merge-<uuid>` needs no lock. The hazard guarded: a *second* `verify-merge`
  resetting/cleaning the shared `_merge-verify` tree out from under a live build → a
  spuriously passed/failed verify (a **correctness** hazard, not just a crash). This flock is
  what supplies the per-host serial invariant that
  `_bump_host_verify_attempt_count`'s docstring (git_ops.py 5501–5505) admits it relies on but
  that `enforce_persistent_worktree_serial_lane` (merge_liveness.py **657–727**, workstation
  startup only via harness.py:6146) does **not** provide on the laptop.

- **A3 — On contention: bounded wait, then escalate + fail; never fall back to ephemeral.**
  Attempt the lock with a **bounded wait of ~10 s** (see §Open-questions for tunability;
  justification below). If still not acquired: do **not** touch the tree, do **not** create an
  ephemeral worktree — emit a **distinguished flock-contention outcome** (the contract in §8,
  carrying host + holder pgid read from the holder's pgid file + this waiter's pgid) and fail
  the verify.
  - **Wait rationale:** the orchestrator dispatches at most one verify per host
    (`HostAllocator` one BUSY slot per host, verify_runner.py 2044–2050; `_MERGE_AHEAD_BOUND
    = 1` at merge_queue.py:259; free-host-gated dispatch at merge_queue.py **9442–9444**), and
    the host lease is released only *after* `run_merge_verify` returns. So in normal operation
    there is **zero legitimate overlap** — any contention is anomalous. The ~10 s wait exists
    only to absorb OS-level lock-release lag at process exit and a brief handoff race; it is
    deliberately short so a genuine second verify surfaces promptly rather than wasting a
    ~1–2 h cold build.
  - **Escalation (Decision Q1 — workstation files it):** the escalation MCP server binds
    `127.0.0.1:8100` (config.py **484–489**) and the laptop CLI holds **no** escalation
    client — it cannot POST cross-host. Therefore the laptop loser only *emits* the
    distinguished contention outcome over its live stdout/`VerifyResult` channel; the
    **workstation dispatcher** recognizes it and files the born-at-L2 via the in-process
    `EscalationQueue.submit(Escalation(level=2, agent_role='orchestrator-verify-host-monitor',
    …))` path — the same local-disk-queue path `merge_liveness.py` **444–462** already uses for
    per-verify-host alarms. The born-at-L2 names the host and (holder, waiter) pgids and
    **blocks the merge** (loud, no silent degradation — aligns with the user's standing
    "prefer loud escalation over silent degradation" directive).
    - *Delivery is sound under K=1:* the flock loser is always the **later** arrival; a new
      dispatch always comes from `RemoteRunner` over a **live** SSH channel, so the loser can
      always report back. The only party that could be an orphan (dead channel) is the
      **holder** — the thing being detected — never the loser.
    - **Rejected alternative (fall-back-to-ephemeral on contention):** silently degrades and
      masks the "impossible" second verify; also wastes a ~1–2 h cold build. Rejected.
    - **Rejected alternative (laptop files directly over HTTP):** would require rebinding the
      escalation MCP server off `127.0.0.1` to the LAN (a RED-tier restart-only config change)
      and exposing an unauthenticated MCP server to the network. Rejected.

- **A4 — Safety valve.** Keep `persistent_merge_worktree_safety_valve_every_n` (config.py:959,
  default 0 = disabled) available; its counter is only correct under the serial invariant the
  flock now provides. Default disabled is accepted; the user has not asked to enable it.

### Change B — tie remote verify lifetime to its dispatch connection

- **B1 — Death trigger: stdin heartbeat-watchdog (chosen mechanism).** The dispatcher
  `RemoteRunner.run_merge_verify` (verify_runner.py **701–853**) currently invokes
  `ssh -o BatchMode=yes -o ConnectTimeout=10 <host> <argv>` (809–812) with **no PTY, no
  `ServerAliveInterval`**, and the remote `verify-merge` `setsid`s itself when `--request-id`
  is set (cli.py 375–378 → `verify_cancel.start_own_process_group`, **os.setsid**). That is
  exactly why an abandoned verify orphans: nothing reaches the detached session.
  - **Mechanism:** change the dispatcher to open the ssh child with `stdin=PIPE` and write a
    periodic **heartbeat** byte/line every `H` seconds down the ssh channel; the remote
    `verify-merge` spawns a **watchdog** (daemon thread / `select` loop on fd 0) that fires on
    **either** stdin **EOF** (clean channel close from orchestrator-kill or SSH-drop) **or**
    **no heartbeat for ~2H** (a hard network partition with no clean close). On fire it
    `killpg`s its own process group (SIGTERM → SIGKILL grace) and exits non-zero.
  - **Why this over the candidates:** (a) it fires on *all three* failure modes (kill, clean
    drop, silent partition) and is **self-contained** — it does not depend on the laptop
    `sshd`'s `ClientAliveInterval`; (b) it **keeps `setsid` + the pgid file unchanged**, so
    `cancel-verify` (cli.py **409–453** → `verify_cancel.cancel_request`) still tree-kills by
    pgid with **zero contract churn** — `setsid` does not close fd 0, so the watchdog reads
    stdin regardless of session. The brief's tentative "switch `setsid`→`setpgid`" is
    **unnecessary**: the watchdog, not session semantics, provides death.
  - **Rejected — `ssh -tt` PTY + SIGHUP handler:** `-tt` merges stderr into stdout and adds CR
    translation, which would **corrupt the stdout the `VerifyResult` JSON is parsed from**
    (verify_runner.py 824–829). Rejected.
  - **Rejected — `systemd-run --scope` tied to the SSH session:** the remote is invoked
    non-interactively; no PAM/login session is tied to the ssh channel by default. Fragile,
    host-config-dependent. Rejected as primary.

- **B2 — Preserve `cancel-verify`.** The graceful-abort path must still kill the full
  descendant tree. Preserved by construction (B1 leaves `setsid`+pgid intact); **verified as a
  required boundary test** in task H, not a standalone mechanism.

- **B3 — No cross-restart reattach (established fact).** There is no reattach of an in-flight
  remote verify across an orchestrator restart: a fresh orchestrator builds a fresh
  `HostAllocator` with no memory of the orphan (`RemoteRunner` is constructed per call; harvest
  is in-process only). A survived verify is therefore pure wasted work **and** the source of
  the orphan Change A's flock guards against — which is why B (die-on-abandonment) is the
  primary fix, not reap-on-dispatch.
  - **Rejected — reap-orphans-on-dispatch alone:** leaves a window (orphan runs between
    abandonment and the next dispatch, especially across a hard restart). It MAY be added later
    as belt-and-suspenders but does not replace B. (Out of scope for this PRD.)

---

## 5. Pre-conditions for activating

All substrate confirmed present on `main` (survey 2026-07-07); no external prerequisite tasks.
The only new code is: the flock wrap + contention outcome (A2/A3, laptop), the workstation
contention consumer + born-at-L2 filing (A3, workstation), the stdin heartbeat/watchdog
(B1, both sides), and the deploy script (A1). Two small internal extensions are specified in
the §8 contract: a **contention discriminant on `VerifyResult`** and a **`stdin=PIPE` +
heartbeat** on the dispatcher ssh child.

---

## 6. Cross-PRD relationship

**No cross-PRD seams.** This is a single-PRD, in-repo change. The one load-bearing seam is
**internal**: the orchestrator-dispatch (`RemoteRunner`) ↔ laptop `verify-merge` CLI boundary,
across which both Change B's connection-death contract and Change A's contention→escalation
outcome live. **This PRD owns that seam** (contract in §8, two-way tests in §9 / task H).

---

## 7. Decomposition plan

Labels are placeholders; task IDs assigned at decompose. The DAG is ordered to serialize edits
to the two shared files (`cli.py`, `verify_runner.py`) and to gate the deploy behind the full
guard+lifecycle.

| Label | Title | Modules | Depends | Kind |
|---|---|---|---|---|
| **α** | Flock-guard laptop persistent-worktree verify span + contention outcome (no ephemeral fallback) | `cli.py`, `verify_cancel.py`, `verify_runner.py` (VerifyResult shape) | — | normal |
| **β** | Workstation consumes flock-contention outcome → born-at-L2 + block merge | `verify_runner.py` / `merge_queue.py` | α | normal |
| **γ** | Connection-death stdin heartbeat-watchdog (die when dispatch connection gone) | `cli.py` (remote watchdog), `verify_runner.py` (dispatcher heartbeat) | α, β | normal |
| **H** | Two-way boundary tests across the dispatch↔CLI seam (integration gate) | tests + both sides | α, β, γ | normal |
| **δ** | Scripted deterministic deploy: enable `persistent_merge_worktree` on the laptop | deploy script + `reify-laptop.yaml` (laptop) | α, β, γ, H | deterministic |

**Per-task observable signals (G2):**

- **α (intermediate — consumed by β, γ, H).** Two concurrent `verify-merge` on the laptop with
  the knob on **serialize** under `.merge_verify.lock`; the second, after the bounded wait,
  **mutates no tree, spawns no `_merge-<uuid>`**, and emits the **distinguished
  flock-contention `VerifyResult`** carrying host + holder pgid (read from the holder's pgid
  file) + waiter pgid. *Unlocks:* β (consumer of the outcome), the correctness invariant H
  asserts.
- **β (intermediate — consumed by H).** Given α's distinguished contention `VerifyResult`, a
  **born-at-L2 escalation appears in the workstation escalation queue** naming the host and the
  (holder, waiter) pgids, and the **merge is blocked** (no ephemeral, no silent pass).
  *Unlocks:* H's contention scenario.
- **γ (intermediate — consumed by H).** Killing the orchestrator **and** dropping the SSH
  connection mid-build **each** terminate the whole laptop build subtree (no lingering
  `rustc`/`cargo`) within `T` s; a heartbeat-starved (simulated hard partition) build likewise
  dies within ~2H; the live path is unaffected (heartbeat flows, build completes normally).
  *Unlocks:* H's lifecycle scenarios.
- **H (LEAF — integration gate).** The §9 boundary-test suite's observable postconditions all
  hold end-to-end against the real dispatch↔CLI seam (not synthetic-input unit tests): the
  three connection-death modes leave no subtree within `T`; `cancel-verify --request-id` still
  tree-kills under the watchdog; flock contention yields a workstation born-at-L2 + blocked
  merge + no ephemeral + no tree mutation; a normal single verify reuses `_merge-verify` with a
  retained `target/`.
- **δ (LEAF — deterministic deploy).** After the guard+lifecycle land, the **next laptop
  verify is observed using `_merge-verify`** with a **retained, non-empty `target/`** across
  consecutive merges (and warm build time drops after warm-up). The deploy's `before_done`
  script (workstation-side) ssh's to the laptop, idempotently sets the flag, and validates the
  YAML parses.

---

## 8. Contract (B + H) — the dispatch↔CLI seam

Two directions cross this seam. Specify both so an architect can implement either side without
further discussion.

### 8.1 Connection-death (Change B)

**Dispatcher (`RemoteRunner.run_merge_verify`, workstation):**
- Opens the ssh child with `stdin=PIPE` (today stdin is unset/inherited).
- Writes a heartbeat token (e.g. a newline) to the ssh child's stdin every `H` seconds for the
  full duration of the verify. Heartbeat write failure (EPIPE) is benign — the child is already
  gone; log and proceed to normal transport-failure handling.
- **Invariant:** the existing best-effort main push, merge-sha push, stdout→`VerifyResult`
  parse (824–829), and ref cleanup (843–849) are unchanged. Adding stdin=PIPE + a heartbeat
  writer must not alter the returned `VerifyResult` on the happy path.

**Remote (`verify-merge` CLI, laptop):**
- When `--request-id` is set (i.e. dispatched, `setsid`'d, pgid-file written), spawn a
  **watchdog** before the build begins. The watchdog owns fd 0.
- Watchdog fires on **EOF on fd 0** (channel closed) **OR** **no heartbeat within `2H`**.
- On fire: `killpg(pgid, SIGTERM)`, brief grace, `killpg(pgid, SIGKILL)`, then `os._exit`
  non-zero. `pgid` is the same value written to the pgid file (`os.getpgrp()` after `setsid`).
- **Invariant (B2):** `setsid` and the pgid file are **unchanged**, so
  `cancel-verify --request-id` (`verify_cancel.cancel_request`, /proc-tree SIGKILL + `killpg`
  backstop) continues to tree-kill. The watchdog and `cancel-verify` may both fire; killing an
  already-dead group is idempotent.
- **Timing:** `T ≈ 2H + kill-grace`. With `H = 5 s` and a ~5 s SIGTERM→SIGKILL grace,
  `T ≈ 15 s`. `T` is **derived from the mechanism**, not a guessed constant; `H` is tunable
  (§Open questions).

### 8.2 Flock-contention outcome (Change A)

**Producer (laptop CLI, on bounded-wait timeout):** returns a `VerifyResult` bearing a
**distinguished contention discriminant** — a machine-readable marker (e.g. a
`contention` reason/category field plus a structured payload `{host, holder_pgid,
waiter_pgid}`) — with `passed=False`. It performs **no** `git reset`/clean/build and creates
**no** ephemeral worktree. (If `VerifyResult` has no free field, α adds one; consumer is β —
G1 satisfied.)

**Consumer (workstation dispatcher / merge-queue):** on recognizing the contention
discriminant, files a born-at-L2 via `EscalationQueue.submit(Escalation(level=2,
agent_role='orchestrator-verify-host-monitor', category='verify_worktree_contention',
summary=…, detail=… naming host + holder/waiter pgids))` (exemplar: merge_liveness.py 444–462)
**and** blocks the merge (a `passed=False` result already blocks — Invariant 5 — so the added
behavior is the escalation filing + a specific, non-degrading disposition).

- **Invariant:** a *non-contention* `passed=False` result is handled exactly as today (no new
  escalation). Only the distinguished discriminant triggers the born-at-L2.

---

## 9. Boundary-test sketch (task H's observable signal)

Each row drives the **real** seam (spawn a real `verify-merge`, act, observe process/queue/tree
state) — user-observable postconditions, not synthetic-input asserts.

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| 1 | Orchestrator killed mid-build | knob on; dispatched verify building; heartbeat flowing | Within `T` s: no `rustc`/`cargo` under the laptop build subtree; pgid group gone |
| 2 | SSH connection dropped mid-build | as #1 | Within `T` s: subtree gone (EOF-on-stdin path) |
| 3 | Heartbeat starved (simulated hard partition) | as #1; dispatcher stops heartbeat but channel not cleanly closed | Within ~2H: subtree gone (heartbeat-timeout path) |
| 4 | `cancel-verify` under the watchdog | dispatched verify building | `cancel-verify --request-id X` tree-kills the full descendant tree; no orphan remains |
| 5 | Flock contention | knob on; one verify holds `.merge_verify.lock` mid-build; a second `verify-merge` launched on the same host | Second waits ≤ bounded wait, then: **no** tree mutation of #1's `_merge-verify`, **no** `_merge-<uuid>` created, emits the contention discriminant; workstation files a born-at-L2 naming host + (holder, waiter) pgids; merge blocked |
| 6 | Normal warm path (no contention) | knob on; single dispatched verify | Reuses `_merge-verify` with a retained non-empty `target/`; **no** escalation; **no** watchdog fire; `VerifyResult` returned unchanged |

---

## 10. Out of scope (explicit)

- **Shared sccache backend (redis / object store).** Redis is infeasible (in-RAM; ~150 GB won't
  fit in 125 GiB workstation RAM); an object store (MinIO/S3) is the correct backend if ever
  pursued, but it is lower priority and separate.
- **The laptop sccache 100 GB cache bump** — already done.
- **Verdict-parity gap:** the laptop `verify_env` omits `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` /
  `REIFY_GATE_EXCLUDE_HEAVY` that the workstation sets — a separate correctness investigation.
- **Reap-orphans-on-dispatch** as a standalone mechanism (see B3 rejection); may be added later
  as belt-and-suspenders.
- **Laptop stale-orphan `_merge-*` worktree disk hygiene** (~29–34 dirs, ~22 GB) — routine
  cleanup, separate.

---

## 11. Open questions (tactical — deferred, not design-blocking)

1. **Bounded-wait value + knob-ness.** Default ~10 s (justified in A3). Decide at impl whether
   to hard-code a constant or add a (reload-tunable) config leaf. **Suggested:** small constant
   first; promote to config only if it ever needs field-tuning.
2. **Heartbeat interval `H`.** Default `H = 5 s` → `T ≈ 15 s`. Decide at impl (task γ) whether
   `H` is a constant or a config leaf. Balance liveness-detection latency vs channel chatter.
3. **`VerifyResult` discriminant shape.** Reuse an existing failure `reason`/`category` field
   vs add a dedicated `contention` payload. Decide at impl (task α); must be losslessly
   parseable by β (§8.2).
4. **Deploy `before_done` script committing.** The deterministic runner requires
   `before_done.script` to be a **committed, executable repo path at submit time**. Decide at
   decompose whether δ's script (`scripts/deploy/enable_laptop_persistent_worktree.sh` or
   similar) is committed by a small prerequisite (landing before δ is filed) or alongside a
   prior task — so δ's `submit_task` validation passes. **Suggested:** land the script via α's
   or a dedicated prereq's merge, then file δ referencing it.
5. **Killpg vs `/proc`-tree kill in the watchdog.** B1 specifies `killpg(pgid)`; genuine
   session-escapes (a child that itself `setsid`s) would evade it. reify's `cargo`/`rustc` do
   not; the sccache server is a pre-existing separate daemon (not our child). Decide at impl
   whether to also mirror `cancel_request`'s `/proc`-descendant sweep for extra robustness.
```
