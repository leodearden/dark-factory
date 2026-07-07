# PRD: Dispatcher-authoritative verdict env + cross-host env-drift alarm

**Status:** draft (authored 2026-07-07). Not yet decomposed/queued.
**Author context:** follows the reify multi-host verify investigation (memory `project_reify_multihost_verify_warmth_2026_07_07`). A tactical hand-patch has already restored parity (see "Prior art / already done"); this PRD makes the fix structural so it cannot recur.

## Problem / consumer + user-observable surface

Dark-factory runs a project's post-merge verify on multiple hosts (reify: workstation `LocalRunner` trust-anchor + laptop remote runner). **Verify env that determines the verdict** — which tests the gate runs — currently lives in each host's own config (`orchestrator.yaml` on the workstation, `reify-laptop.yaml` on the laptop), which are hand-maintained and drift apart. When they diverge, hosts run **different effective test scopes**, so a merge's pass/fail depends on *which host verified it*.

Concretely, this already happened and went undetected: the workstation set `REIFY_GATE_EXCLUDE_HEAVY=1` and `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA=1` (excluding 6 heavy binaries + 4 host-exclusive infra tests from the merge gate, per design), while the laptop set neither — so the laptop ran a strict superset at the gate, including host-exclusive tests documented to "give wrong answers or false-REDs under concurrent host load" (`reify/tests/infra/run-all-classification.manifest`). Result: non-deterministic verdicts by host and a live false-block risk.

**Consumer (G1):** the remote verify execution path (`run_merge_verify_on_worktree` → `LocalRunner`) consumes dispatcher-authoritative verdict env; the L2 escalation-watcher/human consumes the drift alarm. **User-observable surface:** a merge is held to the *same* gate scope regardless of which host verifies it, and a verdict-env divergence between hosts raises a visible escalation instead of silently changing outcomes.

## Why the current design leaves the gap (root, not symptom)

`build_merge_verify_spec` already ships the workstation's `effective_verify_env` in the `MergeVerifySpec`, but the remote **discards it**: for a single-root-config project (reify has no per-module `orchestrator.yaml`) `module_configs == []`, and `LocalRunner` "drives execution from its injected callables + live config, not from the spec (by design)" (`verify_runner.py:~530`). So the remote cargo runs with the *remote host's own* `config.verify_env` (`verify.py` `_resolve_verify_env`, ~2378-2396). That is deliberate — because `verify_env` legitimately contains **host-specific** keys that must NOT be shipped from the dispatcher (jobserver FIFO path, RAM-tuned test concurrency). The gap is that **verdict-affecting env and host-specific env are conflated in one bag**, so keeping the host-specific keys local also strands the verdict-affecting keys as per-host, drift-prone copies.

## Approach (sketch)

Split verify env into two classes and route each correctly:

1. **Verdict-authoritative env** — declared *once* on the orchestrator, shipped in the spec, and applied on every verify host **over** the host's local config (dispatcher wins). Cannot drift per host.
2. **Host-specific env** — stays in each host's local config, never overridden by the dispatcher (jobserver FIFO, `REIFY_TEST_SEMAPHORE_*`, `REIFY_PSI_GATE_MAX_WAIT`, sccache backend). Unchanged behaviour.

Plus a **drift alarm** backstop: wire the already-implemented-but-unused `compare_env_fingerprints` / `EnvFingerprint` (`verify_runner.py:~1232-1323`) into the live drift path so a divergence in the verdict-env subset raises an escalation even if the propagation is somehow bypassed.

## Pre-conditions / substrate verified (G3)

- `MergeVerifySpec` already carries `verify_env` and round-trips via `spec_to_json`/`spec_from_json` (`verify_runner.py:~196-241`); adding a sibling field is a schema extension, not new infra.
- `run_merge_verify_on_worktree` (`verify_runner.py:~362-417`) is the single remote seam where an overlay can be applied before `LocalRunner` is constructed.
- `EnvFingerprint` + `compare_env_fingerprints` exist and are exported but **not wired** into `_run_drift_check` (the live drift path compares *verdicts*, every `verify_drift_check_every_n_lands`=20 lands, not env) — confirmed by the config-propagation investigation.
- `effective_verify_env` fold at `load_config` (`config.py:~2571-2576`) and `_resolve_verify_env` (`verify.py:~2378-2396`) are the env-resolution points on the remote.
- Escalation API for filing a born-at-L2 / L1 drift escalation exists (escalation server).

## Resolved / to-resolve design decisions

**D1 — How verdict env is declared (core decision, resolve in author review):**
- (a) *Dedicated config field* `verdict_env: {K: V}` on OrchestratorConfig, shipped in the spec, applied over remote local `verify_env`. Explicit, no magic, no allowlist to maintain. **Recommended.**
- (b) *Name allowlist* — extract a subset of `verify_env` by key pattern/list and ship that. Reuses existing config but risks an operator placing a host-specific key under a matching name and having it clobbered.
- (c) *Drift alarm only, no propagation* — cheapest; detects but does not prevent drift. Rejected as the primary mechanism (leaves the hand-copy burden and the false-block window open), but its alarm is retained as the backstop under (a).

**D2 — Overlay precedence:** dispatcher `verdict_env` wins over remote local `verify_env` for its keys; all other (host-specific) keys in the remote local config are preserved untouched. Keys must be disjoint by construction (verdict keys move OUT of per-host `verify_env` into `verdict_env` in the migration step).

**D3 — Drift alarm cadence/severity:** compare the **verdict-env subset only** (never the full env — host-specific keys legitimately differ and would false-alarm) on the existing `_run_drift_check` cadence; file at L1 (auto-watcher) escalating to L2 if unresolved, naming the divergent keys and the two hosts.

## Out of scope

- The **tactical** laptop parity fix (adding the two keys to `reify-laptop.yaml`) — **already applied** 2026-07-07; this PRD supersedes it by moving the keys into `verdict_env` (migration step removes the per-host copies).
- The **flock + orphan-lifecycle** work (persistent warm worktree safety) — separate PRD already in flight; disjoint (worktree/lifecycle vs env). No shared files beyond the dispatch↔CLI seam.
- **Cache-backend** propagation (sccache redis/object-store) — separate, deferred; `RUSTC_WRAPPER`/`CARGO_INCREMENTAL` are portable-but-not-verdict and may later ride the same `verdict_env`/`cache_env` seam, but are not in scope here.
- Propagation of **host-specific** env — explicitly NOT done; those stay local.

## Cross-PRD relationship + seam ownership (G4)

| Seam | Owner |
|---|---|
| Orchestrator-dispatch ↔ remote verify-merge CLI (env resolution) | **this PRD** |
| Same seam (worktree lifecycle: flock/orphan) | flock/orphan PRD (disjoint aspect) |
| verdict-vs-host-specific key classification | **this PRD** (D1) |

## Premise validation (G6)

- *"A clean partition of verify env into verdict-affecting vs host-specific exists."* TRUE and enumerated: verdict-affecting = `REIFY_GATE_EXCLUDE_HEAVY`, `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` (+ any future gate-scope var); host-specific = `CARGO_MAKEFLAGS` (jobserver FIFO), `REIFY_TEST_SEMAPHORE_*`, `REIFY_PSI_GATE_MAX_WAIT`, sccache backend. The partition is real but must be **operator-declared** (D1), not inferred.
- *"The drift is real and currently unguarded."* TRUE — the divergence existed undetected until this investigation; `compare_env_fingerprints` is implemented but unwired. The alarm is the active detection mechanism the premise requires.

## Stakes / approach (G5)

HIGH — this is the merge-verdict path; a mis-applied overlay could silently change what the gate tests. Use contracts + **two-way boundary tests** across the dispatch↔CLI seam: assert that (1) a remote run's *resolved* cargo env equals `dispatcher verdict_env` merged over `host local verify_env` with verdict_env winning; (2) host-specific keys survive unchanged; (3) a local (workstation) run and a remote (laptop) run for the same spec resolve to the *same verdict-env subset*.

## Decomposition plan (one leaf per bullet; observable signal named)

1. **Config:** add the `verdict_env` mechanism (per D1) to `OrchestratorConfig`; existing configs unaffected. *Signal: a config with `verdict_env` loads; a config without it behaves byte-identically.*
2. **Spec:** add `verdict_env` to `MergeVerifySpec` + `spec_to_json`/`spec_from_json`. *Signal: round-trips through serialize→deserialize preserving the dict.*
3. **Dispatch:** `build_merge_verify_spec` populates `spec.verdict_env` from the orchestrator config. *Signal: the shipped spec carries the verdict keys.*
4. **Remote apply:** `run_merge_verify_on_worktree` overlays `spec.verdict_env` over the remote's local `verify_env` (verdict wins) before building `LocalRunner`. *Signal: the remote cargo env contains the dispatcher's verdict keys AND the host's own jobserver FIFO.*
5. **Migration:** move `REIFY_GATE_EXCLUDE_HEAVY` / `REIFY_RUN_ALL_EXCLUDE_HOST_INFRA` from both hosts' `verify_env` into the orchestrator `verdict_env`; delete the per-host copies (incl. the 2026-07-07 laptop hand-patch). *Signal: both hosts resolve identical gate scope with the keys declared exactly once.*
6. **Drift alarm:** wire `compare_env_fingerprints` into `_run_drift_check`, comparing the verdict-env subset; file an escalation on divergence. *Signal: an injected verdict-env divergence between hosts fires an escalation naming the keys + hosts; identical env fires nothing.*
7. **Boundary tests (G5):** two-way dispatch↔CLI env-resolution tests per the stakes section. *Signal: tests assert overlay precedence and cross-host verdict-env identity.*

## Open (tactical) questions

- Whether `verdict_env` should be a flat dict or namespaced (e.g. allow per-project defaults). Start flat.
- Whether to fail-closed (refuse dispatch) vs alarm-only when a remote host is reachable but its local config still carries a legacy verdict key post-migration. Lean alarm-only first, fail-closed as a follow-up once migration is proven.
