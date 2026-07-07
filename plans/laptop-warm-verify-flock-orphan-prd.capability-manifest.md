# Capability manifest — laptop warm verify worktree (flock) + orphan lifecycle

Beside PRD `plans/laptop-warm-verify-flock-orphan-prd.md`. Mechanizes G3 (assumed-substrate
verified / wired-not-declared) + G6 (premise validity) per leaf. Every capability a task's
user-observable signal asserts is bound to evidence; a binding resolving to a FAIL value
(`declared-only | test-only | producer-absent | producer-downstream | producer-extent-short |
fixture-ERROR | bound≤floor | rejection-absent`) blocks the batch. **No FAIL bindings — batch
cleared to queue.** Substrate confirmed by the 2026-07-07 anchor survey (line anchors below are
the corrected ones).

**Label → task-id:** α=2306 · β=2307 · γ=2308 · H=2309 · δ1=2310 · (δ2 = deferred follow-up,
not yet filed).

**DAG:** β→α · γ→α,β · H→α,β,γ · δ1→H.

---

## α (2306) — Flock-guard + contention outcome (laptop `cli.py`)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `fcntl.flock(LOCK_EX)` on a fixed lock path | Python stdlib `fcntl` | PASS (library fn) |
| Persistent-worktree branch reachable + identifiable | `grep:git_ops.py:5553` — `if not self.config.persistent_merge_worktree` routes on/off; on→`reset_persistent_merge_worktree` (5574) on the production `acquire_host_verify_worktree` path (5522–5572) | PASS (wired on production path) |
| Holder pgid readable from the pgid file | `grep:cli.py:375-378` — dispatched verify (`--request-id`) writes the pgid file via `verify_cancel.write_pgid_file`; `verify_cancel.pgid_file` resolves the path | PASS (producer writes it: the holder is a `--request-id` verify) |
| Fixed `_merge-verify` path exists | `grep:git_ops.py:5574` — `reset_persistent_merge_worktree` returns `self.persistent_merge_worktree_path` | PASS |
| Distinguished contention discriminant on `VerifyResult` (NEW) | `producer:task-2306` (self) — α's own deliverable on the production stdout path; downstream consumer is β (2307), correctly ordered (β depends on α) | PASS (self-produced; anti-orphan: named consumer β upstream-of-none-but-downstream-of-α) |

*G6:* no numeric/exactness/rejection premise in α's signal (it asserts serialization + "no tree
mutation, no ephemeral, emits discriminant" — all end-to-end capabilities traced to α's own
deliverable + existing substrate).

## β (2307) — Workstation consumes contention → born-at-L2 + block

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| Contention discriminant produced upstream | `producer:task-2306` **upstream** (β depends on α) | PASS (DAG-direction correct) |
| `EscalationQueue.submit(Escalation(level=2, …))` reachable workstation-side | `grep:merge_liveness.py:444-462` — the verify-host-unreachable alarm builds `Escalation(agent_role='orchestrator-verify-host-monitor', …)` and calls `escalation_queue.submit(esc)` on the workstation orchestrator path | PASS (wired production exemplar) |
| `level=2` retained (born-at-L2, no downgrade) | `grep:deterministic_runner.py:100,177` — `orchestrator-*` role prefix keeps `level=2` past the server downgrade gate; β uses `orchestrator-verify-host-monitor` | PASS |
| Merge blocked on `passed=False` | `grep:verify_runner.py:824-841` — parseable `VerifyResult` returned unchanged (Invariant 5) → a `passed=False` result blocks the merge | PASS |

*G6 (branch 3, end-to-end):* the "file born-at-L2 from workstation" capability is delivered by
β itself using the existing in-process `escalation_queue`; not owned by any downstream task. PASS.

## γ (2308) — Connection-death stdin heartbeat-watchdog (both sides)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| Remote reads stdin under `setsid` (EOF trigger) | `grep:verify_cancel.py:131-153` — `start_own_process_group` = `os.setsid()`; `os.setsid` does not close fd 0, so fd 0 still points at the ssh channel → readable | PASS (reasoned from substrate) |
| ssh forwards stdin to the remote | `grep:verify_runner.py:809-812` — `ssh` invoked with no `-n`; stdin forwarded. Dispatcher `stdin=PIPE` + heartbeat is γ's own deliverable | PASS (self-delivered dispatcher side) |
| Dispatcher can open ssh child with `stdin=PIPE` | `grep:verify_runner.py:614` — `asyncio.create_subprocess_exec(...)` supports `stdin=PIPE` | PASS (library capability) |
| `killpg(pgid)` kills the whole build subtree | `grep:cli.py:375-378` (pgid = `os.getpgrp()` after setsid, written to pgid file) + `verify_cancel.cancel_request` `killpg` backstop | PASS (same group cancel-verify already tree-kills) |
| Numeric bound `T ≈ 2H + kill-grace` (G6 branch 1) | Derived from the mechanism (heartbeat interval H + SIGTERM→SIGKILL grace), not a guessed constant. Floor: a timeout has no analytical accuracy floor; `T > 0` for any `H > 0`; H tunable (PRD §11 Q2). H=5s → T≈15s | PASS (`bound` derived, `bound > floor`) |

*B2 (cancel-verify coexistence):* γ keeps `setsid`+pgid file unchanged → `cancel-verify` preserved
by construction; the regression is asserted as a boundary test in H, not re-proven here.

## H (2309) — Two-way boundary tests (integration gate) — **the leaf**

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| Connection-death (3 modes) kills subtree within T | `producer:task-2308` **upstream** (H depends on γ) | PASS (DAG-direction) |
| Flock contention → distinguished result, no tree mutation, no ephemeral | `producer:task-2306` **upstream** | PASS |
| Contention → workstation born-at-L2 + block | `producer:task-2307` **upstream** | PASS |
| `cancel-verify --request-id` still tree-kills under the watchdog | `grep:cli.py:409-453` (`cancel_verify` → `verify_cancel.cancel_request`) + `producer:task-2308` preserves it | PASS |
| Normal warm path reuses `_merge-verify` with retained `target/` | `grep:git_ops.py:5574-5647` (`reset_persistent_merge_worktree`, **existing** substrate from task 1692/1699, `done`). **Row #6 drives the code path with the knob on in the test harness — it does NOT require the production laptop flip (δ2), so no downstream inversion** | PASS (existing substrate, not δ2) |

*G6 (branch 3):* every boundary-test capability traces to α/β/γ (all upstream deps) or to
existing on-main substrate — none to a task that depends on H. No inversion.

## δ1 (2310) — Author + commit the deploy script

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| `persistent_merge_worktree` config field exists | `grep:config.py:868` — `persistent_merge_worktree: bool = Field(default=False, …)` | PASS |
| Laptop config file present (operational precondition) | `/home/leo/.config/orchestrator/reify-laptop.yaml` (laptop-local; referenced by the remote dispatch `--config` and the seed investigation memory) — δ1 validates it via ssh at dry-run | PASS (existing operational file) |
| ssh to the laptop works | `grep:verify_runner.py:809` — RemoteRunner already ssh's to the remote host | PASS |

*Scope note:* δ1 authors+commits the script only; it does **not** flip the flag. The production
flip is δ2 (deferred deterministic deploy, `before_done` = δ1's committed script), which cannot
be filed until δ1's script is on main (deterministic `before_done.script` is validated
exists-and-executable at submit time). This split is PRD §11 Q4's sanctioned resolution.

---

**Result:** 0 FAIL bindings. All capabilities are either existing on-main substrate (survey-
confirmed), self-produced on the production path with a named downstream consumer, or delivered
by an upstream dependency. Batch cleared to queue.
