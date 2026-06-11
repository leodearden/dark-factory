# Capability manifest — concurrent-merge-verify-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified on `main`
2c7bfe1286, 2026-06-11. Line refs drift; symbols are canonical.

## α — Remote cancellation contract

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `verify-merge` CLI exists to extend | grep:`cli.py:279` `@main.command('verify-merge')` | PASS wired |
| No existing lock/pidfile to conflict with | grep over cli.py for flock/pidfile → zero hits | PASS (clean field) |
| RemoteRunner already generates a per-request id | grep:`verify_runner.py:629` `_id_factory`/`uuid4` + ref `refs/merge-verify/<request_id>` (:678) | PASS wired |
| Process-group kill primitive | `os.setsid`/`os.killpg` (stdlib) | PASS |

## β — Host allocator

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Pool + quarantine + remote-eligibility substrate | grep:`verify_runner.py:790-868` `VerifyRunnerPool`, `quarantine`, `eligible_remote`, `_select_runner` | PASS wired |
| Worker-level quarantine set threaded into pool construction | grep:`merge_queue.py:6751` `quarantine=self._runner_quarantine` | PASS wired |
| Runner caching hook reserved | grep:`verify_runner.py:645` `_last_pushed_main_sha` ("Filed as a follow-up: cache runners") | PASS wired |
| Cancel-confirm slot release | producer:task-α (upstream of β) | PASS producer upstream |
| Drift check constructs a both-host pool today (must route via allocator) | grep:`merge_queue.py:7681` region pool with `_build_remote_runners` | PASS wired |
| "Queue never stalls on dead remote" is producible | existing `RunnerUnavailable` → local fallback contract (verify_runner.py:883-913) retained | PASS wired |

## γ — Verifier split (dispatch + ordered finalize)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Single verifier loop + inline verify to restructure | grep:`merge_queue.py:6147` `_verifier_loop`; `:6344` `await self._verify_and_advance(item)` | PASS wired |
| Abort-poll wrapper to reuse per in-flight verify | grep:`merge_queue.py:6742-6787` `ensure_future(_run_post_merge_verify(...))` + `VERIFY_ABANDON_POLL_SECS` (:4533) | PASS wired |
| Ordered CAS substrate | grep:`git_ops.py:2199` `advance_main(expected_main=...)`; `merge_queue.py:6856-6863` CAS loop | PASS wired |
| Chain-invalidation / re-merge machinery | grep:`merge_queue.py:6427` `_remerge`; remerge_reason block `:6246-6272` | PASS wired |
| Speculation depth K + ledger registration on handoff | grep:`merge_queue.py:5980` `_register_owned_merge_worktree(merge_result.merge_worktree)`; semaphores `:4559/:4565` region | PASS wired |
| Trains bypass verify (no-host pass-through is consistent) | grep:`merge_queue.py:6324-6339` `immediate_outcome` branch | PASS wired |
| Warm-swap stays per-host-single | grep:`merge_queue.py:6684-6712` `_acquire_warm_verify_worktree` inside the (local) verify path; allocator slot ≤1/host | PASS wired |
| "Two overlapping spans" producible with fake runners | merge_verify events carry runner + duration (verify_runner.py:916-927); fake slow runners overlap deterministically | PASS |
| Downstream abort effective on remote | producer:task-α via task-β (both upstream) | PASS producer upstream |
| Host slots | producer:task-β (upstream) | PASS producer upstream |

## δ — Ledger-aware ENOSPC prune

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Prune function + both call sites | grep:`git_ops.py:2079` `prune_stale_merge_worktrees(keep=...)`; call sites `merge_queue.py:545,:744` | PASS wired |
| The keep-set source (live owned worktrees) | grep:`merge_queue.py:4749` `_owned_merge_worktrees: set[Path]` (1728, landed) | PASS wired |
| Persistent worktree already prune-exempt | grep:`git_ops.py:2099-2102` exemption via `_iter_merge_worktrees` | PASS wired |

## ε — Multi-verify observability

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Snapshot + singular verify state to generalize | grep:`merge_queue.py:4912` `snapshot()`; `:4651` `_verify_item`/`_verify_phase` | PASS wired |
| Heartbeat line to extend | grep:`merge_queue.py:5104` region `_maybe_log_queue_heartbeat` + touch loop | PASS wired |
| In-flight collection | producer:task-γ (upstream of ε) | PASS producer upstream |

## ζ — Overlap boundary gate

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| All structural capabilities | producers: α, β, γ, δ, ε — all upstream of ζ | PASS producer upstream |
| B7 single-host byte-identical regression producible | no-runner config → one host slot → degenerate serial path; existing suite (test_merge_queue*.py) is the oracle | PASS wired |
| B8 heartbeat-under-overlap regression | 1728 ledger + touch loop landed (`:4787-4806`, `:5104`); γ-1730 boundary tests exist as precedent | PASS wired |
| No numeric throughput floor asserted in any leaf | PRD §5 decision 8 — live measurement deferred to ops checklist | PASS (floor branch n/a) |

No FAIL bindings. Batch clear to queue.
