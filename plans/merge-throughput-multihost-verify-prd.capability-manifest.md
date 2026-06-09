# Capability manifest — Multi-host merge-verify (Lever C)

Mechanizes G3 + G6 per task: every capability a task's signal asserts, bound to evidence.
Evidence verified 2026-06-09 against `main` (cite-by-symbol — `main` moves; re-locate at impl time).
**No binding resolves to a FAIL value → batch is not blocked.**

Legend: `wired` = referenced on the production path on main · `producer:task-X upstream` = delivered by an upstream task in the dependency closure · `net-new` = built by this PRD (producer named) · `host` = host/OS capability proven by ε.

---

## α — MergeVerifySpec + VerifyResult serialization *(intermediate)*
| Capability | Binding | Verdict |
|---|---|---|
| `MergeVerifySpec` frozen dataclass | net-new · producer:task-α | PASS |
| `VerifyResult` JSON codec (byte-identical round-trip) | net-new · producer:task-α | PASS |

## β — LocalRunner + VerifyRunnerPool; route `_run_post_merge_verify` *(intermediate; slice)*
| Capability | Binding | Verdict |
|---|---|---|
| `_run_post_merge_verify` integration point | grep:merge_queue.py:~364 wired (called by the merge worker) | PASS |
| `run_scoped_verification` / `_run_unscoped_typechecks` bundle | grep:merge_queue.py:~473/~1555 + verify.py wired | PASS |
| `MergeVerifySpec` / `VerifyResult` | producer:task-α upstream | PASS |
| `VerifyRunner` protocol / `LocalRunner` / `VerifyRunnerPool` | net-new · producer:task-β | PASS |
| verify event `runner=` provenance field | net-new · producer:task-β (event emit) | PASS |

## γ — `orchestrator verify-merge` host subcommand *(intermediate)*
| Capability | Binding | Verdict |
|---|---|---|
| `orchestrator` CLI entry point (host for subcommand) | grep:orchestrator/pyproject.toml `[project.scripts] orchestrator = "orchestrator.cli:main"` wired | PASS |
| verify bundle reused host-side (same code → byte-identical) | producer:task-α + grep verify fns wired | PASS |
| `verify-merge` subcommand | net-new · producer:task-γ | PASS |

## δ — RemoteRunner + pool fail-safe fallback *(intermediate)*
| Capability | Binding | Verdict |
|---|---|---|
| `RemoteRunner` (git push sha → ssh invoke → parse) | net-new · producer:task-δ | PASS |
| `orchestrator verify-merge` on host | producer:task-γ upstream | PASS |
| `git push` / `ssh` to host | host (LAN/SSH/git remote) — proven by ε | PASS |
| `RunnerUnavailable` → fallback to LocalRunner | net-new · producer:task-δ | PASS |

## ε — Laptop verify-env provisioning + parity *(the D6 G3 task)*
| Capability | Binding | Verdict |
|---|---|---|
| faithful laptop verify env (toolchain pin, `verify_env`, OS deps, sccache reach, SSH/git) | **net-new · producer:task-ε** (provisioned + *proven*, not assumed — the design §6.3 load-bearing item) | PASS |
| verdict parity over known-pass/known-fail SHA corpus | producer:task-ε (the standing fidelity proof) | PASS |

## ζ — Raise speculation depth to K + liveness recompute *(intermediate)*
| Capability | Binding | Verdict |
|---|---|---|
| `_speculation_slot` (Event) + `_merge_ahead_cap = Semaphore(_MERGE_AHEAD_BOUND)` | grep:merge_queue.py:~3788/~3795/~103 wired | PASS |
| `check_merge_liveness_margin` + `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS=10800` | grep:merge_queue.py:~5477/~1847 wired | PASS |
| `advance_main` `expected_main` CAS (serial/ordered) | grep:merge_queue.py wired | PASS |
| K-permit generalization + startup margin guard | net-new · producer:task-ζ | PASS |

## η — Per-host serial guard + laptop warm worktree *(leaf-ish; depends_on dark_factory:1692)*
| Capability | Binding | Verdict |
|---|---|---|
| κ's global `_MERGE_AHEAD_BOUND==1` startup guard (to reframe) | **producer:dark_factory:1692 UPSTREAM** (external dep wired; anti-inversion: 1692 upstream of η) | PASS |
| `git.persistent_merge_worktree` knob | producer:dark_factory:1692 upstream | PASS |
| `prune_stale_merge_worktrees` (exemption point) | grep:git_ops.py wired | PASS |
| per-host guard reframe + laptop fixed-path warm worktree | net-new · producer:task-η | PASS |

## ι — Drift detector *(depends_on δ, ε)*
| Capability | Binding | Verdict |
|---|---|---|
| escalation emit (dedup'd) | grep escalation MCP `escalate_*` wired | PASS |
| dual-host same-sha verify | producer:task-δ (RemoteRunner) + task-ε (parity) upstream | PASS |
| drift detector + `verdict_parity_ok` event | net-new · producer:task-ι | PASS |

## κ — Shared sccache backend *(depends_on ε)*
| Capability | Binding | Verdict |
|---|---|---|
| sccache active (`RUSTC_WRAPPER=sccache`) | design §6 evidence (reify env) | PASS |
| shared backend (redis/memcached/s3/gcs) + both hosts pointed at it | net-new · producer:task-κ | PASS |
| `sccache --show-stats` remote-hit observability | host tool present | PASS |

## λ — End-to-end throughput integration gate *(LEAF; depends_on η, ι, κ)*
| Capability (G6 end-to-end — each must come from λ's dependency closure, never downstream) | Binding | Verdict |
|---|---|---|
| `runner=local\|laptop` provenance on verify events | producer:task-β/δ upstream | PASS |
| drift parity holds over the window | producer:task-ι upstream | PASS |
| both hosts warm (multiplier realized) | producer:task-η/κ upstream | PASS |
| merge-queue oldest-age / depth heartbeat events | grep heartbeat events wired on main | PASS |
| completion-rate / throughput delta vs single-host baseline | measured at this leaf; all inputs upstream — **no producer-downstream** | PASS |
