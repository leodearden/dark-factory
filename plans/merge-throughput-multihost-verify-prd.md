# PRD — Multi-host merge-verify (Lever C)

**Status:** active · generic orchestrator capability, first consumer = reify · authored 2026-06-09.
**Source design:** `plans/merge-throughput-disjoint-former-design.md` (measurement-backed, reify `runs.db` 67d/97 runs + `main` first-parent 300 merges, 2026-06-09). Lever **C** is the primary recommendation (~85% P(improves throughput) if built); **A′** is the sibling `plans/merge-throughput-coupling-tolerant-train-former-prd.md`; **B** / disjoint-only batching are dead (design §9 — not relitigated here).
**Approach:** **B + H** (contract + two-way boundary tests). This PRD sits on the load-bearing path to `main`; a remote PASS that would be a local FAIL = unverified code on `main` (constraint 3 violation). The fidelity invariant earns the full contract (§A) + boundary-test sketch (§B).
**Scope guard (load-bearing):** every verify dispatched to any runner remains a **full, independent merge gate** (same scoped `run_scoped_verification` + the fail-closed `_run_unscoped_typechecks`). C buys throughput from **non-contending CPU capacity** (a second host), never from narrowing or skipping a gate.

---

## 1. Goal — raise merge-queue throughput by adding non-contending verify capacity

reify's completion rate is bounded by **merge-queue throughput**, not task execution: the queue advances `main` through a **single serial verifier** and the post-merge verify is the long pole (design §1; live run mean queue depth 3.2, 52% of dequeues backlogged — M1). The three hard constraints (design §2) freeze `verify_seconds` and forbid CPU-contending concurrent verifies. C moves the one remaining orthogonal lever: **add verify capacity on a separate CPU** (a second host), so two verifies run concurrently on non-contending cores.

**User-observable end state (consumer = the orchestrator `SpeculativeMergeWorker` + ops/dashboard + every land-waiter):**

| | Today (single host) | After C (2 runners) |
|---|---:|---:|
| Concurrent post-merge verifies | 1 | 2 (1 local + 1 laptop), non-contending |
| Throughput multiplier (backlogged regime) | 1× | ~1.5–1.8× (r≈0.5–0.7; design §6) |
| Verify event provenance | none | `runner=local\|laptop` on every verify event |
| Gate scope / correctness | full | **full (unchanged)** — every runner runs the same full gate |
| Code on `main` | verified | **verified (unchanged)** — remote-vs-local verdict parity enforced |

Multiplier figures are **expectations, not gated thresholds** (G6): no task freezes a guessed minute/× into a RED test. Each task asserts a *measured improvement direction + a recorded delta vs the single-host baseline*, with these numbers as the expectation.

## 2. Background — why a second host is the only non-contending lever

Throughput identity (design §3): `tasks/hour ≈ tasks_landed_per_verify / verify_seconds`. Constraints 1–2 freeze `verify_seconds` and forbid same-CPU concurrency. The only moves are amortize (trains — A′), elide (disjoint-skip — dead, design §9), or **add capacity on non-contending CPU**. C is the last: a second host literally raises the verify-seconds budget per wall-second without violating constraint 2's CPU-saturation rationale, and is **coupling-agnostic** — reify's tight crate coupling (M3) does not touch it.

**The second host (confirmed available):** a laptop, 16 hardware threads, 64 GB RAM, spare NVMe. The real multiplier is set by *warmth and thermal headroom*, not raw hardware (design §6): sustained all-core verifies throttle a laptop (budget ~60–80% desktop-equivalent), and **sccache is the multiplier knob** — a *shared* sccache backend lets the laptop skip cold dependency compiles (~1× warm) instead of paying them (~0.5× cold).

**The pipeline is already build-ahead-shaped.** The Merger merges N+1 against N's commit; the Verifier re-merges N+1 if N fails (chain-invalidation, `_verifier_loop` / `_remerge`, merge_queue.py ≈:4613/:4855). Depth is capped at 1 today by `_speculation_slot` **only because there is one CPU** (merge_queue.py ≈:3788, `_merge_ahead_cap = Semaphore(_MERGE_AHEAD_BOUND)` ≈:3795) — not for correctness. C raises that cap to K = number of runners; CAS-advance of `main` stays strictly serialized and ordered (existing `advance_main` `expected_main` CAS).

## 3. Sketch of approach — a verify-runner pool behind `_run_post_merge_verify`

The unit dispatched is the **pre-advance verify bundle**: scoped `run_scoped_verification` + the fail-closed `_run_unscoped_typechecks` gate — both need the materialized merge tree, so they run together on one runner and return one combined `VerifyResult`. The post-advance unscoped pyright (`_check_post_merge_pyright`, fails-open, runs after advance) is **not** on the serialized critical path and stays local.

**Integration point:** `_run_post_merge_verify` (merge_queue.py ≈:364) replaces its direct `run_scoped_verification(merge_wt, …)` + `_run_unscoped_typechecks(merge_wt, …)` calls with `pool.dispatch(merge_sha, spec)`. Surrounding logic (disk guard, ENOSPC prune-retry, timeout loop-breaker) is unchanged. Also covers `_reverify_rebased_tree` and `_do_train_merge`'s verify (so A′ trains compose for free). `LocalRunner` reuses `merge_wt`; `RemoteRunner` ships `merge_sha`.

Mechanisms (each has a named consumer — G1):

| # | Mechanism | Consumer |
|---|---|---|
| 1 | `MergeVerifySpec` (frozen dataclass) + `VerifyResult` JSON round-trip | the runners (host-independent gate reproduction) |
| 2 | `VerifyRunner` protocol (`health`, `run_merge_verify`) | `VerifyRunnerPool` |
| 3 | `LocalRunner` (wraps the current path, no re-checkout) | `VerifyRunnerPool` |
| 4 | `RemoteRunner` (git push sha → ssh invoke → JSON; `RunnerUnavailable` on transport error) | `VerifyRunnerPool` |
| 5 | `VerifyRunnerPool.dispatch()` (pick-free, prefer-remote-when-local-busy, fail-safe fallback to local) | `_run_post_merge_verify` |
| 6 | `orchestrator verify-merge --sha --spec` host CLI subcommand (runs the **same** verify code) | `RemoteRunner` |
| 7 | Speculation-depth generalization (`_speculation_slot` Event → K-permit Semaphore; raise `_merge_ahead_cap`) | the Merger/Verifier pipeline |
| 8 | Per-host serial guard (reframe κ's global `_MERGE_AHEAD_BOUND==1` assertion to "≤1 in-flight verify per host") + laptop fixed-path warm worktree | startup guard + RemoteRunner host |
| 9 | Liveness-margin recompute for K>1 (`check_merge_liveness_margin`) | startup guard |
| 10 | Drift detector (periodic same-sha dual-host verdict parity + divergence escalation) | ops / fidelity guarantee |
| 11 | Shared sccache backend (both hosts) + config | both runners (the multiplier) |
| 12 | Config: `merge.verify_runners`, per-runner host/ssh/git-remote, drift cadence, sccache backend pointer | operator |

## 4. Resolved design decisions

- **D1 — dark-factory PRD; all tasks `dark_factory:`.** C is a *generic orchestrator capability* (any verify-bound project benefits), implemented entirely in DF orchestrator code; reify opts in via a config knob (`merge.verify_runners`). This inverts warm-builds' D1 (a reify PRD with one DF task) because C is ~all DF code with a thin reify/host-config sliver. *(Leo, 2026-06-09.)*
- **D2 — Transport = SSH + a new `orchestrator verify-merge` subcommand, not a daemon.** `RemoteRunner` does `git push <host> <merge_sha>:refs/merge-verify/<request_id>` then `ssh <host> orchestrator verify-merge --sha <sha> --spec <json>`, which runs the **same orchestrator verify code** → byte-identical `VerifyResult` parse (fidelity by construction). Warmth lives in the **filesystem** (a fixed-path warm worktree + retained `target/` on the laptop) and a standalone sccache server — both process-independent — so SSH+script is *warm too*; a long-running daemon would add only marginal warmth (skips per-invoke process startup) at the cost of daemon lifecycle/health/crash-recovery/partition handling. Rejected the §10 framing that "warm reuse needs a daemon." *(Leo, 2026-06-09.)*
- **D3 — Depend on warm-builds κ (`dark_factory:1692`); reframe its serial guard to per-host.** C raises in-flight verify count to K=2, but the 2nd verify runs on the **laptop**, so each *host* still runs ≤1 verify → the per-host serial invariant that κ's single warm worktree rests on **holds**. C `depends_on` 1692 and (a) reframes κ's global `_MERGE_AHEAD_BOUND==1` startup guard to **"≤1 in-flight verify per host,"** and (b) provisions the laptop's own fixed-path warm worktree mirroring κ's invariants 1–6. Both PRDs stay warm; clean composition. *(Leo, 2026-06-09.)*
- **D4 — K starts at 2 (one local + one laptop); K is config-driven = number of healthy runners.** Generalize `_speculation_slot` to a K-permit semaphore; CAS-advance stays strictly serialized and in order. Failure semantics unchanged (chain-invalidation re-verifies N+1 on N-fail). *(Leo, 2026-06-09.)*
- **D5 — Fail-safe is absolute: a dead/slow/partitioned/closed laptop must NEVER stall the queue.** `RunnerUnavailable` (transport failure) → fall back to `LocalRunner` for that request. Distinguish *transport* failure (retry local) from *verify timeout* (existing timeout handling — the verify ran, it just didn't finish). *(Leo, 2026-06-09.)*
- **D6 — Laptop env fidelity is verified, never assumed.** The one load-bearing G3 risk (design §6.3) is a faithful verify env on the laptop. A dedicated provisioning+parity task stands it up and proves verdict parity on a corpus of known-pass/known-fail SHAs **before** the laptop is trusted in the live pool; the drift detector is the standing guarantee thereafter. *(Leo, 2026-06-09.)*

## 5. Pre-conditions for activating

- **`dark_factory:1692` (warm-builds κ — persistent warm merge worktree)** — hard cross-task prereq for the per-host worktree work (C-η). C reframes the guard κ installs; landing C-η before κ would assert against a guard that doesn't exist yet.
- **Laptop reachable over LAN/SSH; git push to the host; faithful verify env** — provisioned + parity-proven by C-ε (D6), not assumed.
- **Shared sccache backend** stood up (C-κ) for the warm multiplier; C functions without it (laptop at ~0.5×) but the multiplier is depressed.

## 6. Substrate verification (G3) — integration points exist; the novel substrate is in-scope to build

Verified at authoring (2026-06-09, `main` HEAD; cite-by-symbol — re-locate at impl time):

| Capability | Evidence (verified) |
|---|---|
| `_run_post_merge_verify` integration point | present, merge_queue.py ≈:364; calls `_run_unscoped_typechecks` ≈:473 (def ≈:1555) |
| `run_scoped_verification` / `_run_unscoped_typechecks` gate fns | present (verify.py / merge_queue.py) — the bundle C dispatches |
| Speculative pipeline (`_verifier_loop`, `_remerge`, chain-invalidation) | present, merge_queue.py ≈:4613 / ≈:4855 |
| `_speculation_slot` (Event, depth-1) + `_merge_ahead_cap = Semaphore(_MERGE_AHEAD_BOUND)` | present, merge_queue.py ≈:3788 / ≈:3795; `_MERGE_AHEAD_BOUND=1` ≈:103 |
| `check_merge_liveness_margin` + `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS=10800` | present, merge_queue.py ≈:5477 / ≈:1847 |
| `orchestrator` CLI entry point (host for the new subcommand) | present, `[project.scripts] orchestrator = "orchestrator.cli:main"` |
| `_reverify_rebased_tree` / `_do_train_merge` (also routed through the pool) | present, merge_queue.py |
| **`VerifyRunner` / `RemoteRunner` / `orchestrator verify-merge` / `MergeVerifySpec`** | **absent today — net-new, built by this PRD** (C-α…C-δ); not assumed substrate |

The **load-bearing assumed substrate** is the laptop's verify-env fidelity (toolchain pin via `rust-toolchain.toml`, replicated `verify_env`, OS deps, sccache reachability, SSH/LAN). It is **not** verified at authoring — it is *built and proven* by C-ε and *monitored* by C-ι. No `.ri` grammar surface; no DB schema. G3 otherwise a no-op.

## 7. Cross-PRD / cross-repo relationship (G4)

| Other | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| warm-builds κ = `dark_factory:1692` | C **consumes** the local warm worktree + **reframes** its serial guard | persistent fixed-path worktree lifecycle + the `_MERGE_AHEAD_BOUND==1` startup guard → "≤1 verify per host" | **C** owns the reframe (C-η); κ owns the local worktree | κ pending; C-η `depends_on` it |
| A′ = `plans/merge-throughput-coupling-tolerant-train-former-prd.md` | **compose** (a train's single union verify dispatches through the same pool) | `pool.dispatch(merge_sha, spec)` is verify-mechanism-agnostic | each owns its side; no integration task | independent — neither blocks the other |
| warm-builds (reify PRD, linker/debuginfo/OCCT phases) | sibling, same throughput goal, different lever (verify *cost* vs verify *capacity*) | none (disjoint code) | — | independent |

No reciprocal "the other owns it." The one genuine seam (κ's guard) is resolved: **C owns the per-host reframe**.

## 8. Decomposition plan — task DAG with observable signals (G2)

Greek labels; task IDs assigned at decompose. Leaves name a user-/operator-observable signal; intermediates name the downstream they unlock. All `×` numbers are expectations, never frozen thresholds (G6).

**Phase 1 — contract + local-pool vertical slice**
- **α — `MergeVerifySpec` + `VerifyResult` JSON serialization.** *(intermediate → unlocks β, γ, δ.)* A frozen `MergeVerifySpec` (verify_commands, unscoped_typecheck spec, task_files, verify_env, cold_timeout, is_merge_verify) and a `VerifyResult` JSON codec. **Signal:** a golden round-trip test serializes a spec+result and parses it back byte-identically; unlocks the runners. *Modules:* new `orchestrator/src/orchestrator/verify_runner.py`. *Leaf?* no — intermediate.
- **β — `VerifyRunner` protocol + `LocalRunner` + `VerifyRunnerPool`; route `_run_post_merge_verify` through `pool.dispatch`.** *(vertical slice; intermediate → unlocks ζ.)* One local runner only; behaviour byte-identical to today. **Signal:** the merge-gate verify runs through `pool.dispatch()` with a single `LocalRunner`, the existing single-host suite stays green (regression), and every verify event now carries `runner=local`. *Modules:* `verify_runner.py`, `merge_queue.py`.

**Phase 2 — the remote runner (adds the laptop)**
- **γ — `orchestrator verify-merge` host CLI subcommand.** *(intermediate → unlocks δ.)* New subcommand on the existing CLI that materializes a worktree at `--sha`, runs the **same** verify bundle from `--spec`, emits a JSON `VerifyResult`. **Signal:** `orchestrator verify-merge --sha <x> --spec <json>` on a host emits a `VerifyResult` JSON that parses identically to a local run of the same SHA (CLI integration test). *Modules:* `cli.py`, `verify_runner.py`.
- **δ — `RemoteRunner` + pool fail-safe fallback.** *(leaf.)* `git push <host> <sha>:refs/merge-verify/<id>` → ssh-invoke γ → parse; transport error → `RunnerUnavailable` → pool falls back to `LocalRunner`. **Signal:** with the laptop reachable, a dispatched merge verify returns a `VerifyResult` and the verify event carries `runner=laptop`; with the laptop down/closed (fault-injected), the **same** dispatch falls back to local and the queue does **not** stall. *Modules:* `verify_runner.py`.
- **ε — Laptop verify-env provisioning + parity verification (the D6 G3 task).** *(leaf.)* Pin toolchain, replicate `verify_env`, match OS deps, confirm sccache reachability + SSH/git push. **Signal:** running the **same** `merge_sha` on both hosts yields identical verdicts over a corpus of N known-pass + N known-fail SHAs (committed parity report); this is the standing fidelity guarantee that lets the laptop join the live pool. *Modules:* host config, `verify_runner.py` health.

**Phase 3 — concurrency (the throughput slice)**
- **ζ — Raise speculation depth to K + liveness-margin recompute + startup guard.** *(intermediate → unlocks η, λ.)* Generalize `_speculation_slot` Event → K-permit semaphore; raise the effective in-flight bound to K; **CAS-advance stays strictly serialized and ordered** (unchanged). Recompute `check_merge_liveness_margin` for K>1 (worst-case queued-worktree age < 0.75 × `INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`, per host) and reject an over-budget config at startup. **Signal:** with K=2 and 2 runners, two speculatively-stacked merges verify **concurrently** (event log shows 2 overlapping verify spans) while `main` advances **one merge at a time in order**; chain-invalidation still re-verifies N+1 on N-fail; a deliberately over-budget bound×timeout config is rejected at startup. *Modules:* `merge_queue.py`.
- **η — Per-host serial guard reframe + laptop fixed-path warm worktree.** *(leaf; `depends_on` `dark_factory:1692` + δ + ζ.)* Reframe κ's global `_MERGE_AHEAD_BOUND==1` startup assertion to **"≤1 in-flight verify per host"**; provision the laptop's own fixed-path warm worktree mirroring κ invariants 1–6 (reset-in-place, prune-exempt, periodic from-scratch safety valve). **Signal:** with `git.persistent_merge_worktree` on **and** K=2, startup does **not** trip the old global guard; each host keeps exactly one warm worktree across attempts (not pruned); the from-scratch safety-valve verify still passes. *Modules:* `merge_queue.py`, `git_ops.py`, RemoteRunner host.

**Phase 4 — fidelity monitoring + the multiplier**
- **ι — Drift detector.** *(leaf; `depends_on` δ, ε.)* Periodically (or on a sample) run the same `merge_sha` on both hosts and assert identical verdicts. **Signal:** a forced divergence (deliberately-broken laptop env) raises a **dedup'd** drift escalation; matching verdicts emit a `verdict_parity_ok` event. *Modules:* `verify_runner.py`, escalation emit.
- **κ — Shared sccache backend (the multiplier knob).** *(leaf; `depends_on` ε.)* Point both hosts at a shared sccache backend (redis/memcached/s3/gcs — choice tactical, §Open). **Signal:** a laptop verify shows sccache **remote-hit rate > 0** against the shared backend (`sccache --show-stats` after a warm run) and a recorded cold-vs-warm laptop-verify wall-time delta. *Modules:* host config, `verify_env`.

**Phase 5 — integration gate (the B+H leaf, §B boundary tests)**
- **λ — End-to-end throughput integration gate.** *(leaf; `depends_on` η, ι, κ.)* With both runners live + warm + drift detector on, run a backlogged window and record reify completion-rate and merge-queue oldest-age + depth (heartbeat events) deltas vs the single-host baseline, with verify events carrying `runner=local|laptop` and drift parity holding throughout. **Signal:** the §B boundary-test sketch passes end-to-end: a measured throughput improvement direction + recorded delta, provenance tags present, zero drift-divergence escalations over the window. *Modules:* integration test harness, dashboard read.

**DAG:** α → {β, γ, δ}; γ → δ; β → ζ; ζ → {η, λ}; δ → {η, ι, κ}; ε → {η, ι, κ}; `dark_factory:1692` → η; {η, ι, κ} → λ.

## 9. Out of scope

- **Narrowing or skipping any merge verify** — FORBIDDEN (scope guard / design §2 constraint 3). Disjoint verify-skip (B) is dead (design §9).
- **Train formation / amortization** — that is A′ (`plans/merge-throughput-coupling-tolerant-train-former-prd.md`); C only adds capacity. They compose via the pool but A′ is independent.
- **Shortening the verify itself** — separate work (warm-builds reify PRD `docs/prds/warmer-builds-merge-verify.md` + constraint 1). C is the *capacity* lever; warm-builds is the *cost* lever.
- **A daemon transport** — rejected (D2); kept as a future optimization if per-invoke startup ever dominates.
- **>2 runners / a third host** — K is config-driven so it generalizes, but only the 2-runner (local+laptop) case is built/validated here.
- **Raising single-host `_MERGE_AHEAD_BOUND` for latency** — dead (design §9; breaches the liveness margin on one host).

## 10. Open questions (tactical — surfaced, not blocking)

1. **Shared sccache backend choice** (redis / memcached / s3 / gcs). **Suggested:** redis on the orchestrator host (LAN-local, low-latency, both hosts reach it). Decide during κ.
2. **Drift-detector cadence** (every Nth land vs sampled vs nightly). **Suggested:** every 20th land + nightly, tightened on any observed divergence (mirrors warm-builds invariant-6 cadence). Decide during ι.
3. **`pick_free` policy under contention** (strict prefer-remote-when-local-busy vs latency-aware). **Suggested:** start with prefer-remote-when-local-busy (design §6.1); revisit if the laptop's thermal throttle makes it the slow path. Decide during β/δ.
4. **`refs/merge-verify/<id>` cleanup cadence on the host.** **Suggested:** prune on `VerifyResult` return + a periodic sweep. Decide during δ.

---

## §A — Contract (B+H)

The seam is `VerifyRunner`. An architect implementing either side works to these signatures + invariants without further discussion.

```python
@dataclass(frozen=True)
class MergeVerifySpec:
    verify_commands: tuple[VerifyCommand, ...]   # from config + module_configs, scoped by task_files
    unscoped_typecheck: UnscopedTypecheckSpec     # the _run_unscoped_typechecks gate spec
    task_files: tuple[str, ...] | None
    verify_env: Mapping[str, str]                 # RUSTC_WRAPPER, CARGO_INCREMENTAL, sccache pointer, …
    cold_timeout_secs: float                      # merge_verify_cold cascade
    is_merge_verify: bool = True

class VerifyRunner(Protocol):
    name: str                                     # 'local' | 'laptop'
    async def health(self) -> bool: ...
    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult: ...

class VerifyRunnerPool:
    async def dispatch(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult: ...
```

**Invariants (load-bearing):**
1. **Verdict-equivalence.** For any `merge_sha`, `RemoteRunner.run_merge_verify` MUST return the verdict `LocalRunner` would. The host runs the *same* orchestrator verify code (D2) → parse is byte-identical by construction; the residual risk is environment, owned by C-ε + monitored by C-ι. A remote PASS that would be a local FAIL is the one defect class that puts unverified code on `main`.
2. **Fail-safe liveness.** `dispatch` NEVER raises `RunnerUnavailable` to its caller — a transport failure is caught and retried on `LocalRunner`. A closed/offline laptop degrades throughput to 1×, never to a stall (D5).
3. **Advance ordering is independent of verify concurrency.** Verifies run on ≥1 runner concurrently; `main` advances one merge at a time, in order, via the existing CAS. Raising K touches `_speculation_slot`/`_merge_ahead_cap` only — never the advance path.
4. **Per-host serial.** At K=2 each host runs ≤1 verify, so each host keeps exactly one warm worktree (the per-host reframe of κ's guard). Raising effective concurrency *per host* above 1 requires a per-host worktree pool (not in scope).
5. **Transport ≠ timeout.** A `RunnerUnavailable` (host down / ssh fail / push fail) retries local; a *verify timeout* (the verify ran and exceeded `cold_timeout_secs`) flows through the existing timeout handling unchanged.

## §B — Boundary-test sketch (B+H) — faces both the dispatch side and the runner side

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| B1 | Local-only parity (slice) | pool has 1 `LocalRunner`; `_run_post_merge_verify` routed through it | verdict identical to the pre-C direct-call path; verify event carries `runner=local`; full suite green |
| B2 | Remote happy path | laptop up, env parity proven; pool has local+laptop | a merge verify dispatched to the laptop returns a `VerifyResult`; event carries `runner=laptop` |
| B3 | Remote fail-safe | laptop down/closed mid-dispatch | dispatch falls back to `LocalRunner`; verdict produced; queue does **not** stall; one `runner_unavailable` log, no escalation |
| B4 | Verdict parity (fidelity) | same `merge_sha` over N known-pass + N known-fail SHAs | local and laptop verdicts identical on every SHA; any divergence is a hard alarm |
| B5 | Drift divergence alarm | laptop env deliberately broken (toolchain/dep mismatch) | drift detector raises one dedup'd escalation; the live pool drops the laptop until re-proven |
| B6 | K=2 concurrency + ordered advance | K=2, queue depth ≥3, 2 runners | 2 verify spans overlap in the event log; `main` advances one merge at a time, in order; N-fail re-verifies N+1 via chain-invalidation |
| B7 | Per-host warmth under K=2 | `persistent_merge_worktree` on, K=2 | startup guard does not trip; each host retains its single warm worktree across attempts; safety-valve from-scratch verify still passes |
| B8 | Liveness margin at K>1 | over-budget bound×timeout config | startup rejects the config (margin guard); an in-budget config starts cleanly |
