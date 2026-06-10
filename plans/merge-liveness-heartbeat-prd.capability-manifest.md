# Capability manifest — merge-liveness-heartbeat-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified on `main`
8b703a550a, 2026-06-10. Line refs drift; symbols are canonical.

## α — Owned-worktree liveness heartbeat

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Independent heartbeat coroutine that runs while merger/verifier block | grep:`orchestrator/src/orchestrator/merge_queue.py:4914` `_heartbeat_loop` ("Runs independently of the merger and verifier loops"); spawned at :4937 | PASS wired |
| Per-tick cadence constant | grep:`merge_queue.py:178` `_HEARTBEAT_POLL_S: float = 30.0` | PASS wired |
| Worktree path carried per queued item (ledger registration anchor) | grep:`merge_queue.py:3223` `SpeculativeItem.merge_wt: Path \| None` | PASS wired |
| Cleanup sites to deregister at | grep:`merge_queue.py:5675,:5698` (verifier abandon/halt), `:760` fail-path in `_run_post_merge_verify`, `_remerge` discard path | PASS wired |
| mtime touch primitive | `os.utime` (stdlib); reaper reads `wt.stat().st_mtime` at `merge_queue.py:3009` — same inode (worktree root dir) | PASS |
| Owned-worktree ledger | NEW — produced by α itself; consumers β (formula premise), γ (boundary tests) are downstream | PASS producer:task-α upstream |

## β — Guard re-derivation (heartbeat model)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `check_merge_liveness_margin` / `enforce_merge_liveness_margin` / `MergeLivenessConfigError` | grep:`merge_queue.py:6622,:6750,:6740` | PASS wired |
| `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS = 10800` | grep:`merge_queue.py:2184` | PASS wired |
| Fail-closed call site to rewire (drop raw `_k`) | grep:`harness.py:3355-3360` (`except MergeLivenessConfigError: raise`) | PASS wired |
| Heartbeat floor (the new formula's input) | producer:task-α (upstream of β) | PASS producer upstream |
| "K=2 crash params pass" is producible | formula no longer contains K → 7200s timeout irrelevant to verdict; floor (≈600s) < threshold (8100s) | PASS floor:600 < 8100 |
| "Over-budget still refused" is producible | guard retains injectable `liveness_secs`; inject ≤ floor/safety_factor → raises | PASS |
| K=1 regression | shipped defaults already pass today's guard; new model is config-independent for the same constants | PASS |

## γ — K=2 startup + reaper-protection boundary gate

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `verify_runners` config key live (K=2 constructible in a test config) | grep:`config.py:1328` `verify_runners: list[VerifyRunnerConfig]`; `:1411` `enabled_verify_runners`; harness consumes at `harness.py:3347` (df 1716, merged 26894cceec) | PASS wired |
| `_start_merge_worker` startable under test | grep:`harness.py:3319`; precedent `orchestrator/tests/test_harness_k_from_config.py` | PASS wired |
| Startup-repro test RED on pre-batch main | reproduced live 2026-06-10 (`MergeLivenessConfigError`, reify incident revert) — premise true by observation | PASS |
| Reaper coalesce/reap branches testable | grep:`merge_queue.py:2947` `coalesce_or_enqueue_merge_request`, alive branch :3012, reap branch :3021-3040; `liveness_secs` injectable (`:2954`) | PASS wired |
| Guard passes K=2 / refuses over-budget | producer:task-β (upstream of γ) | PASS producer upstream |
| Live-owner mtime freshness | producer:task-α (upstream of γ) | PASS producer upstream |

No `declared-only`, `test-only`, `producer-absent`, `producer-downstream`,
`fixture-ERROR`, or `bound≤floor` bindings. Batch clear to queue.
