# Reify merge-queue throughput — design input for /prd

**Status:** design-complete, measurement-backed. Ready to decompose via `/prd`.
**Recommended PRD scope:** lever **C (multi-host verify)** first (primary); **A′
(coupling-tolerant train former)** as a separate, lower-priority follow-up; **B** and
disjoint-only batching are **dead ends** (recorded below so they aren't relitigated).

This doc is the design artifact; it intentionally does NOT pre-decompose into tasks — that
is `/prd`'s job. It hands `/prd` the premise, the evidence, the chosen designs, the
interfaces, and the open decisions, mapped to the G1–G6 gates where useful.

---

## 1. Problem & premise (G6)

reify's task-completion rate is limited by **merge-queue throughput**, not by task
execution. The merge queue advances `main` through a **single verifier**, and the
post-merge verify is the long pole. Premise substantiated by measurement: the current
live run is heavily backlogged (mean queue depth 3.2, 52% of dequeues have a waiting
backlog — §4 M1). So work is arriving faster than the single serial verifier can clear it.

## 2. Hard constraints (from the requester; non-negotiable)

1. **Don't shorten the verify** (worked separately).
2. **No concurrent verifies that contend for the same CPU** (verify is ~CPU-saturating).
3. **Never put unverified code on `main`.**

## 3. Throughput identity (why the lever space is narrow)

```
tasks/hour ≈ tasks_landed_per_verify / verify_seconds
```

The constraints freeze `verify_seconds`. So throughput moves only by raising
`tasks_landed_per_verify` — **amortize** (one verify lands N tasks → trains) or **elide**
(skip a verify a task provably doesn't need) — OR by the orthogonal move of adding verify
**capacity on non-contending CPU** (a second host: literally raises the `verify_seconds`
budget per wall-second without violating constraint 2's contention rationale).

## 4. Measurements (reify `runs.db`, 67 days/97 runs; `main` first-parent history, 300 merges)

- **[M1] Backlog: YES, growing.** All-history thin (depth≥1 = 11.5%); **current live run:
  52% of dequeues backlogged, mean depth 3.2, 36% at depth ≥4, max 15.** (No `queue_depth`
  field in events; reconstructed from queued/dequeued timeline.)
- **[M2] Verify cost: ~170 s median, p90 ~400–480 s, tail ~600–1200 s, ~$0.90 each.
  WORKSPACE/union verify cost UNMEASURED** — reify runs `merge_verify_workspace=false`, so
  it has never run one. This number gates A′; it does not gate C.
- **[M3] Disjointness: NEAR-ABSENT.** 32 member crates under `crates/`; per-merge touched
  crates median 1 / mean 1.69, but **rdep closure median ~22 (dev-deps) / 6.5 (no
  dev-deps)**. Adjacent-disjoint merge pairs **0–17%**; mean largest-disjoint subset **~1.0
  (K=3) to ~1.5 (K=5)**. Hubs: `reify-compiler` & `reify-eval` each in 33.7% of merges;
  `reify-core` is a dep of 23/32. **The workspace is a tightly-coupled core.**
- **[M4] Failure mix:** landed 56% (done 51% + already_merged 5%); **cas_retry 31%** (cheap
  contention churn — a symptom of a hot, deep single-task main); conflict 8% (current run
  12%); equivalence/timeout/unknown ~9% costly-no-land.
- **[M5] Coupling hubs** (`reify-compiler`/`reify-eval`/`reify-stdlib`/`reify-ir`/
  `reify-core`) ⇒ no useful `always_verify` allowlist; the core *is* the hub.

## 5. Decision & rationale

The seductive levers (disjoint verify-skip; disjoint-only batching) are **dead for reify**
because [M3] shows it is never crate-disjoint. The SAME coupling, however, *inverts* the
amortization economics in trains' favour — see A′. Ranking:

| Lever | Verdict | P(improves reify throughput) | Gated on |
|---|---|---|---|
| **C — multi-host verify** | **Primary** | **~85% if built** | hardware (have it) + dispatch eng + env fidelity — NOT reify's structure |
| **A′ — coupling-tolerant former** | **Secondary** | **~45%** | [M2] workspace-verify cost + thrash rate `s(N)` |
| B — disjoint verify-skip | Dead | <10% | never disjoint [M3]; also slim branch verify (§9) |
| disjoint-only batching | Dead | <15% | ~1.0–1.5 tasks/verify [M3] |
| speculation/merge-ahead bump (1 host) | No | ~5% | latency not throughput; unsafe vs liveness margin |

---

## 6. PRIMARY DESIGN — Lever C: multi-host verify

**Concept.** Keep the merge (commit creation + CAS-advance of `main`) serialized on the
orchestrator. Dispatch only the expensive **post-merge verify** to a *pool of verify
runners* (the orchestrator host + a second host). Two verifies run concurrently on
**separate CPU** — non-contending, so constraint 2's rationale (CPU saturation) does not
apply. Each verify remains a full, independent gate, so constraint 3 holds.

**The second host (confirmed available):** a laptop, 16 hardware threads, 64 GB RAM, spare
NVMe. Adequate for cargo (NVMe matters for `target/` I/O; 64 GB is ample). Caveats that set
the real multiplier, not the hardware:
- **Thermal throttling:** a merge verify is sustained all-core; back-to-back verifies will
  throttle a laptop. Budget ~60–80% of desktop-equivalent sustained.
- **Expected multiplier:** in the backlogged regime, throughput ≈ `(1 + r)/median_verify`
  where `r` = laptop verify rate ÷ main-host rate. At `r ≈ 0.5–0.7` ⇒ **+50–70% completion
  rate (~1.5–1.8×)**. Coupling-agnostic — the [M3] result does not touch it.
- **sccache is the multiplier knob.** reify already sets `RUSTC_WRAPPER=sccache`. If it is a
  *local-disk* cache, the laptop pays cold compiles (the 0.5× end). Point both hosts at a
  **shared sccache backend** (redis/memcached/s3/gcs) and the laptop skips cold compilation,
  approaching warm main-host speed (~1×). Highest-leverage prep step; also helps the main
  host's cold merge-worktree verifies (adjacent to the separate verify-shortening work).

### 6.1 Interface — verify-runner pool

The unit dispatched is the **pre-advance verify bundle**: scoped `run_scoped_verification`
+ the fail-closed unscoped type-check gate (`_run_unscoped_typechecks`) — both need the
materialized merge tree, so they run together on one runner and return one combined verdict.
The post-advance unscoped pyright (`_check_post_merge_pyright`, fails-open, runs after
advance) may stay local or dispatch later; it is not on the critical serialized path.

```python
@dataclass(frozen=True)
class MergeVerifySpec:
    # everything a runner needs to reproduce the gate, host-independently
    verify_commands: ...        # from config + module_configs (scoped by task_files)
    unscoped_typecheck: ...     # the _run_unscoped_typechecks gate spec
    task_files: list[str] | None
    env: dict[str, str]         # verify_env (RUSTC_WRAPPER, CARGO_INCREMENTAL, ...)
    cold_timeout_secs: float    # merge_verify_cold cascade
    is_merge_verify: bool = True

class VerifyRunner(Protocol):
    name: str                                    # 'local' | 'laptop'
    async def health(self) -> bool: ...
    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        # materialize a worktree at merge_sha, run the verify bundle, return VerifyResult
        ...

class LocalRunner:   # wraps the CURRENT path: run on the existing _merge-* worktree, no re-checkout
class RemoteRunner:  # 1. git push <host> <merge_sha>:refs/merge-verify/<request_id>
                     # 2. ssh <host> orchestrator-verify --sha <sha> --spec <json>  → JSON VerifyResult
                     # 3. transport error / host down → raise RunnerUnavailable
                     #    (NB: run the SAME orchestrator verify code on the host so the
                     #     VerifyResult parse is byte-identical — fidelity by construction)

class VerifyRunnerPool:
    async def dispatch(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        runner = await self._pick_free(prefer_remote_when_local_busy=True)
        try:
            return await runner.run_merge_verify(merge_sha, spec)
        except RunnerUnavailable:
            return await self._local.run_merge_verify(merge_sha, spec)   # FAIL-SAFE: never stall the queue
```

**Integration point:** `_run_post_merge_verify` (`merge_queue.py:404`) replaces its direct
`run_scoped_verification(merge_wt, …)` + `_run_unscoped_typechecks(merge_wt, …)` calls with
`pool.dispatch(merge_sha, spec)`. The surrounding logic (disk guard, ENOSPC prune-retry,
timeout loop-breaker bookkeeping) is unchanged. LocalRunner reuses `merge_wt`; RemoteRunner
ships `merge_sha`. Also covers `_reverify_rebased_tree` and `_do_train_merge`'s verify.

### 6.2 Pipeline change — against the existing speculative pipeline

The pipeline is already built for build-ahead with failure handling: the Merger merges N+1
against N's commit and the Verifier re-merges N+1 if N fails (chain-invalidation,
`_verifier_loop:4587`; `_remerge:4813`). Today the depth is capped at 1 by `_speculation_slot`
**because there is one CPU** — not for correctness.

- Raise speculation depth to **K = number of runners** (start 2). With K runners, up to K
  speculatively-stacked merges are verified concurrently, each on a distinct runner.
- **CAS-advance stays strictly serialized and in order** (the existing `advance_main`
  `expected_main` CAS + chain-invalidation already enforce this). Concurrency is in the
  *verify* step only; `main` still moves one merge at a time.
- Failure semantics are unchanged: if N fails verify, N+1 (built on N) re-merges+re-verifies
  via existing chain-invalidation. For reify (coupled, N+1 stacked on N) this is the right
  model — verifying N+1 *against main-without-N* would just trigger the rebase-reverify gate
  on advance anyway (coupling ⇒ overlap ⇒ re-verify). So speculative-stack + multi-host is
  strictly better than independent-against-main for a coupled workspace.

### 6.3 Substrate invariant (G3) — the one thing that can violate constraint 3

A remote **PASS that would be a local FAIL = unverified code on main.** The remote verify
must be behaviourally identical:
- Pin the toolchain (`rust-toolchain.toml` — Rust enforces this), replicate `verify_env`,
  match OS-level deps.
- **Drift detector:** periodically (or on a sample) run the *same* `merge_sha` on both hosts
  and assert identical verdicts; alert on divergence. This is the standing guarantee that
  the substrate stays faithful — and is itself a clean G2 observable.

### 6.4 Fail-safe & liveness (must-haves the PRD specifies)

- **Dead/slow/partitioned host ⇒ fall back to local** (the `RunnerUnavailable` path). A laptop
  that is closed/offline must NEVER stall reify's queue. Distinguish *transport* failure
  (retry local) from *verify* timeout (existing timeout handling).
- **Recompute `check_merge_liveness_margin`** (`merge_queue.py:5420`) for K>1: the local
  reaper governs only *local* `_merge-*` worktrees; the remote host holds its own. Confirm
  worst-case queued-worktree age stays under `0.75 × INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS`
  with the new in-flight count, or raise the liveness window.

### 6.5 PRD-scoping notes for C

- **G1 consumer:** the orchestrator merge worker (`SpeculativeMergeWorker`) consumes the
  runner pool; ops/dashboard consume the throughput gain.
- **G2 user-observable leaf signal:** reify completion rate rises; merge-queue oldest-age /
  depth (heartbeat events) fall; verify events carry `runner=local|laptop`; the drift
  detector reports verdict-parity. Any of these is a concrete, observable pass signal.
- **G3 assumed substrate:** LAN/SSH reachability; git SHA shipping; **faithful verify env on
  the laptop** (the load-bearing assumption — verify it explicitly, don't assume it).
- **G4 seam:** the `VerifyRunner` protocol is the seam between `merge_queue` (owns dispatch &
  advance ordering) and the runner host (owns execution). The shared contract is
  env/toolchain fidelity.

---

## 7. SECONDARY DESIGN — Lever A′: coupling-tolerant train former

### 7.1 Why not "only disjoint" (the reframing)

My first sketch batched only crate-disjoint tasks. That was an error: I conflated
"won't break each other" with "crate-disjoint."
- **Correctness never needs disjointness** — a train runs ONE full (union-scoped) verify on
  the merged tip before advance; whatever lands is verified together. Disjointness was only a
  *selection heuristic* for (a) conflict-free stacking and (b) low thrash.
- Neither needs crate-disjointness: clean stacking is a **line-level** property (same crate,
  different lines = fine); low thrash is about *semantic* interaction, which the combined
  verify *catches* anyway.
- **The inversion:** for a coupled workspace, batching coupled tasks saves **more**. Their
  rdep closures overlap, so the **union closure ≈ a single task's closure** — one union
  verify costs ≈ one single verify but lands N tasks (the shared core is built/tested once,
  not N times). Disjoint batching only amortizes fixed per-worktree overhead. So reify's
  coupling — which kills disjoint-skip — makes amortization *attractive*, if you select by
  stackability and scope to the union closure.
- **The price** of coupled batching: failure attribution. A disjoint train that fails verify
  is a real per-member bug; a coupled train that fails *might* be an interaction (neither
  member individually broken), so on failure you re-verify members separately (or bisect) —
  ~1 extra verify. Keep N small (2–3) to bound it.

### 7.2 The former + the correctness fix

A scheduler-side former (sibling of δ₂ `_maybe_enqueue_group_merge`, `workflow.py:547`) that,
when ≥2 tasks are merge-ready, selects a small (N≤3) **line-level-stackable** subset, stacks
their branches (rebase b2→b1→main; reuse `rebase_onto_main` + sibling-predecessor branching
`git_ops.py:572`), assigns synthetic `metadata.train.{id,order,members}`, and lets the tip
enqueue a `GroupMergeRequest`.

**Mandatory correctness fix (G3):** the existing train path **under-verifies** when
`merge_verify_workspace=false`. `GroupMergeRequest` is built with the **tip's** `task_files`/
`module_configs` only (`workflow.py:653-656`), so scoped verify would cover only the tip's
crates and leave lower members' crates unverified. reify never hit this (zero trains), but
A′ MUST set the train's `task_files`/`module_configs` to the **UNION over all members** (or
force workspace). The union of closures is the sound minimum and, given overlap, ≈ the tip
alone.

### 7.3 Economics, risk, and the two de-risking experiments

- +EV in verify-time when union ≈ single (overlapping coupled members) iff combined-verify
  success `s(N) > 1/N`. Upside ≈ (N−1) near-full verifies per successful train. Secondary
  win: collapses the 31% CAS-retry churn [M4] and the rebase-reverify amplification (coupling
  ⇒ ~always-overlapping ⇒ a deep single-task main forces re-verifies ~83–100% of the time the
  rebase gate fires; fewer separate advances relieves this).
- Risks (all economic, none touch correctness): **thrash** (`s(N)` unmeasured),
  **stacking conflicts** (line-level; floor = the 8–12% single-merge conflict rate [M4]),
  and **workspace-verify cost [M2]** for the force-workspace variant.
- **Two experiments before building A′:**
  1. **Union-verify wall-time [M2]:** time a cold cargo verify scoped to a 2–3-task union
     closure vs the ~170 s single. union≈single ⇒ upside real; union≈N×single ⇒ break-even.
  2. **`s(N)` proxy:** from history, sample sets of ~3 contemporaneous landed merges and
     check whether their combined tree would have passed (or proxy: how often two merges
     landing close together produced a follow-up fix-forward). Estimates the thrash rate.

---

## 8. Independence & sequencing

C and A′ are **independent** features touching different layers (C: verify execution/transport
+ pipeline depth; A′: scheduler batch formation + train scoping). They compose (a train's
single union verify can itself be dispatched to a runner) but neither blocks the other. Build
**C first** (higher confidence, coupling-agnostic, hardware in hand). Consider A′ only after
its two experiments clear.

## 9. Dead ends (recorded so /prd does not relitigate)

- **B (disjoint verify-skip):** dead for two independent reasons. (1) reify is never
  crate-disjoint [M3]. (2) The branch (task-phase) verify is deliberately **slimmer** than the
  merge verify — it omits the unscoped whole-package type-check, which runs **only** at merge
  phase (`_run_unscoped_typechecks`, `merge_queue.py:473`,`1687`; task-phase verify is plain
  scoped, `workflow.py:1305`,`3472`). So skipping the merge verify and trusting the branch
  verify would land code that only cleared the lower bar — even with zero main churn. To make
  it safe you must re-expand the branch verify to merge stringency, which either moves the cost
  without removing it, or — if run concurrently across tasks — creates concurrent whole-package
  verifies = the CPU saturation constraint 2 forbids.
- **Disjoint-only batching:** dead — would average ~1.0–1.5 tasks/verify [M3], i.e. rarely
  forms a batch.
- **Raising speculation depth / `_MERGE_AHEAD_BOUND` on one host:** latency not throughput,
  and `bound=2 × 7200 s cold = 14400 s > 0.75 × 10800 s = 8100 s` breaches the liveness margin
  (`merge_queue.py:103`,`5420`).

## 10. Open decisions each PRD must still resolve

- **C:** runner transport (SSH+script vs a long-running verify daemon on the laptop — daemon
  enables warm worktree/cache reuse); shared-sccache backend choice; K (start 2); the
  drift-detector cadence; liveness-margin recompute outcome.
- **A′:** N cap; line-level stackability test + conflict fallback (drop member vs shrink batch);
  union-scope vs force-workspace; failure attribution (re-verify singles vs bisect).
