# PRD: True concurrent merge verifies — overlapping verify spans across hosts

**Date:** 2026-06-11 · **Status:** approved for decomposition · **Predecessors:**
`plans/merge-throughput-multihost-verify-prd.md` (Lever C — supersedes its deferred
"K-permit free/busy refinement"), `plans/merge-liveness-heartbeat-prd.md` (1728–1730,
landed — the liveness substrate this PRD builds on).

Cite by symbol; line refs are as-of `main` 2c7bfe1286 and drift.

## 1. Consumer + user-observable surface

- **Consumer:** the reify merge-throughput goal (the merge queue is the bound). Lever C
  as shipped is serial-offload: `run()` spawns exactly one `_verifier_loop`
  (merge_queue.py:6147) and `VerifyRunnerPool._select_runner` is prefer-remote — flipping
  C today routes ALL verifies to the laptop serially, gaining ~nothing (measured warm:
  workstation ≈11 min, laptop ≈12 min). This PRD makes K hosts verify **concurrently**.
- **User-observable surface:** with one enabled `verify_runners` entry, the event log
  shows **two overlapping merge-verify spans** (one `runner=local`, one `runner=laptop`)
  while `main` advances strictly one merge at a time in submission order; sustained
  merge throughput approaches the sum of host rates (direction + recorded delta — no
  frozen numeric threshold, G6); a dead/closed laptop degrades to today's serial-local
  behaviour without a stall.

## 2. Premise validation (G6 — resolved by code reading, 2026-06-10/11 sessions)

1. **Verifies are serial today by structure, not by accident:** one verifier coroutine;
   `_verify_and_advance` (merge_queue.py:6666) runs verify → CAS inline per item.
2. **Speculation substrate already exists:** the merger builds ahead K items
   (`_speculation_slot`/`_merge_ahead_cap`, K permits each), each with its own local
   worktree registered in the 1728 liveness ledger
   (`_register_owned_merge_worktree`, :5980). Overlap does not need new merge-ahead
   machinery — it needs the verifier to *drain* concurrently.
3. **Liveness is already overlap-proof:** the heartbeat (1728) touches every owned
   worktree every 30s regardless of schedule shape; the guard (1729) is
   topology-independent. No liveness work in this PRD.
4. **Abort machinery half-exists:** `_verify_and_advance` already wraps the verify in an
   abort-poll (`VERIFY_ABANDON_POLL_SECS`, triggers: sole-waiter abandon, operator halt
   — task 1681). What's missing is making an abort *effective on a remote host*:
   killing the local ssh does not reliably kill the remote command (no pty), and the
   laptop's η fixed-path warm worktree means a zombie `verify-merge` run plus a fresh
   dispatch would share one path — **corruption, not just waste**. The remote CLI has no
   lock/pidfile today (`cli.py:279`).
5. **Timing premise:** warm verify ≈11 min (workstation, idle-host benchmark) vs
   ≈12 min (laptop, parity-report round 2) → overlap yield ≈1.9×. Favourable.
6. **ENOSPC prune is overlap-hostile today:** `prune_stale_merge_worktrees(keep=<one>)`
   (git_ops.py:2079; call sites merge_queue.py:545,:744) removes every other `_merge-*`
   worktree — under overlap that destroys the *other* in-flight verify's worktree.
   The 1728 ledger is the exact keep-set needed.

## 3. Approach

Split the verifier into **concurrent dispatch + strictly-ordered serial finalize**,
backed by a **host allocator** and a **remote cancellation contract**.

- **Dispatch stage:** pull items off `_verifier_queue`, run the existing
  pickup logic that does not depend on the predecessor's verdict (Mechanism 1 cap
  release on-drain; Mechanism 2 `main_advanced` staleness re-merge for real items;
  abandoned/halt drains), acquire a host from the allocator, and launch the
  abort-poll-wrapped verify as a task. Up to K verifies in flight, ≤1 per host.
  Items with no verify to run (`immediate_outcome` trains, `skip_verify`) enter the
  in-flight deque as no-host pass-throughs.
- **Finalize stage (strictly serial, in submission order):** await the **head** of the
  in-flight deque; run CAS `advance_main(expected_main=item.base_sha)` +
  `_finalize_advanced_merge` exactly as today. On head failure/abort: abort every
  downstream in-flight verify (local cancel + remote `cancel-verify`), discard
  worktrees, `_remerge` onto actual main, re-dispatch — chain-invalidation semantics
  preserved, now paid as aborted speculative verifies instead of never-started ones.
  Out-of-order completions wait their turn (head-of-line latency is accepted; with
  ~equal host speeds the cost is small).
- **Host allocator:** worker-lifetime object owning one slot per host (local + each
  enabled runner). Policy: **prefer-local-when-free** (trust anchor, marginally faster
  — deliberately inverts the shipped prefer-remote, whose purpose was offload).
  `RunnerUnavailable` → quarantine host (existing `_runner_quarantine` set) +
  re-dispatch on a free host; a dead laptop degrades to serial-local, never a stall.
  Remote slot release: on clean completion, or **after `cancel-verify` confirms**;
  cancel failure → host quarantined until a probe shows no `verify-merge` running.
  Caches `RemoteRunner` instances (activating the reserved `_last_pushed_main_sha`
  main-push dedup).
- **Remote cancellation contract (γ-CLI extension, user-selected design):**
  `verify-merge` gains `--request-id`, runs its work in its own process group and
  records the pgid keyed by request_id; new `cancel-verify --request-id` kills the
  group (idempotent: success when nothing is running). RemoteRunner passes the
  request_id it already generates for the push ref.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Kill ssh + probe-before-reuse (no remote CLI) | Zombie burns laptop CPU ≤12 min and the slot is held on a probe loop; saves little vs the small cancel CLI. User selected the cancel endpoint. |
| Never abort remote (hold slot to completion) | Each N-fail/abandon wastes a full laptop verify of overlap capacity; contradicts 1681's existing abort-the-wasted-compute semantics. |
| Concurrent CAS / out-of-order landing | Violates the ordered-advance invariant everything downstream assumes (equivalence gates, trains, recover-main); not on the table. |
| Keep prefer-remote selection | With overlap, first-free + prefer-local engages the trust anchor for single-item windows and overflow goes remote; prefer-remote was an offload-era policy. |

## 4. Pre-conditions (G3 — verified on main this session)

`_verifier_loop` (merge_queue.py:6147) + `_verify_and_advance` (:6666) with abort-poll
(:6742-6787, `VERIFY_ABANDON_POLL_SECS` :4533) and warm-swap (:6684-6712);
`_merger_loop` handoff + ledger registration (:5980); 1728 ledger API
(`_owned_merge_worktrees` :4749, register/deregister/cleanup/touch :4757-4806, touched
from heartbeat :5104); `_remerge` (:6427); `_finalize_advanced_merge` (:898);
`advance_main` (git_ops.py:2199); `VerifyRunnerPool` + quarantine + `eligible_remote`
(verify_runner.py:790-868); `RemoteRunner` + per-request id + ref push
(verify_runner.py:592-762, reserved `_last_pushed_main_sha` :645); `verify-merge` CLI
(cli.py:279 — **no lock/pidfile today**; the cancellation contract is NEW surface,
queued as task α); `prune_stale_merge_worktrees` (git_ops.py:2079; call sites
merge_queue.py:545,:744); drift check pool construction (:7681 region);
`snapshot()` (:4912) + `_verify_item`/`_verify_phase` singular state (:4651);
`MERGE_LANES` priority lanes (:241 — orthogonal, untouched); trains/`immediate_outcome`
path (:6324-6339); `enabled_verify_runners` (config.py:1411). New mechanisms: the
**host allocator** (produced by β) and the **cancel contract** (produced by α).

## 5. Resolved design decisions

1. **Remote abort = explicit cancel endpoint** (user decision 2026-06-11): pgid file
   keyed by request_id; `cancel-verify` kills the group; slot freed only on confirmed
   cancel; cancel failure quarantines the host. `verify-merge` must `setsid` so the
   kill never takes sshd along.
2. **Ordered finalize is non-negotiable** — CAS-advance stays strictly serial in
   submission order; out-of-order completions wait.
3. **Abort downstream on head-failure** (consistent with 1681): frees host capacity
   ~one verify earlier; the cancel contract makes it safe.
4. **Prefer-local-when-free allocation**; remote takes overflow. Per-host ≤1 in-flight
   (η's serial-lane invariant) is enforced by the allocator's slots — the persistent
   warm worktree swap (one per host) is safe by construction.
5. **Who goes through the allocator:** the speculative worker's verifies and the drift
   check (it needs both hosts by definition). Legacy/recovery callers of
   `_run_post_merge_verify` (MergeWorker, rebase-reverify, main-health probe) get a
   transient **local-only** pool — recovery paths stay on the trust anchor and out of
   the slot accounting.
6. **Single-host config (no `verify_runners`) must remain behaviour-identical** — one
   host slot degenerates to today's serial loop; the regression gate (ζ) holds this.
7. **Speculation/merge-ahead depth stays K** — the dispatch bubble (merge time while a
   host idles) is seconds against ~11-min verifies; deepening the buffer is a measured
   follow-up, not this PRD (Open questions).
8. **No numeric throughput floor in any leaf signal** (G6): the live two-host
   measurement is the ops enable checklist's job; leaves assert structure (overlap,
   order, abort) on fake runners.

## 6. Out of scope

- ~~Flipping reify's `verify_runners` on (operator action)~~ **Superseded 2026-06-11:**
  activation is automated as task **η** (user directive; orchestrator restarts
  explicitly authorized for this case). What remains out of scope: live
  fault-injection (powering the laptop off — ζ's B3 covers the logic with fakes) and
  the multi-hour λ throughput-delta measurement (observational follow-up once a real
  backlog window has run; η reports the first overlap signals).
- Dashboard rendering of multi-verify state (dashboard package; ε ships the snapshot
  shape; a dashboard task can consume it later — split-multi-package rule).
- Merge-ahead buffer deepening (K+1) and any host-weighted scheduling.
- A third verify host (allocator generalizes; nothing here assumes K=2).

## 7. Cross-PRD seams (G4)

| Seam | Owner | Status |
|---|---|---|
| Lever C PRD "K-permit refinement deferred to ζ" | superseded — this PRD is that slice | this PRD |
| Liveness heartbeat (1728–1730) | consumed unchanged; ledger becomes the prune keep-set (δ) and covers all overlap-queued worktrees | landed; regression-held by ζ |
| Warm-builds per-host worktree (1692/η) | allocator slots enforce ≤1/host; `enforce_persistent_worktree_serial_lane` call unchanged | this PRD enforces, 1692 invariants hold |
| Trains γ2 (1717-1722) | `immediate_outcome` items bypass hosts, flow through ordered finalize | this PRD (no-host pass-through) |
| Drift detector ι | acquires host slots through the allocator | this PRD (β) |
| γ CLI contract (`verify-merge`) | extended with `--request-id`/`cancel-verify`; both hosts must update together | this PRD — η syncs the laptop df checkout before the flip |
| Lever C enable checklist (`plans/lever-c-enable-path-gap-2026-06-10.md` steps 2-3) | executed by η (automated); η also updates the checklist + deletes the obsolete per-host comment in reify yaml | this PRD (η) |

## 8. Decomposition (G5: B+H — contract = decisions 1-6; boundary tests = ζ)

- **α — Remote cancellation contract** (`cli.py` + small helper module + tests).
  `verify-merge --request-id` runs work in its own process group (`setsid`), writes a
  pgid file keyed by request_id; `cancel-verify --request-id` kills the group and
  removes the file — idempotent success when nothing matches; non-zero only when a
  live group could not be killed. **Signal:** on one host, a long `verify-merge` is
  killed (whole build process tree) within seconds by `cancel-verify` from another
  shell, rc=0; cancelling an unknown/finished id exits 0; the pgid file lifecycle is
  observable in the run dir. **Consumer:** β (slot release), γ (downstream abort).
- **β — Host allocator** (`verify_runner.py` + the pool-construction seam in
  `merge_queue.py` + tests). Worker-lifetime allocator: one slot per host;
  prefer-local-when-free; cached RemoteRunners (+ main-push dedup via
  `_last_pushed_main_sha`); `RunnerUnavailable` → quarantine + redispatch (never a
  stall); remote slot freed on clean completion or confirmed `cancel-verify`, else
  host quarantined until a `pgrep`-probe is clean; drift check acquires through the
  allocator; non-worker callers get transient local-only pools. **Signal:** unit
  tests — two fake hosts: ≤1 in-flight per host under load; both-free → local
  selected; RunnerUnavailable quarantines + falls back with the queue still draining;
  cancel-confirm frees the slot, cancel-fail quarantines. **Consumer:** γ.
- **γ — Verifier split: concurrent dispatch + ordered serial finalize**
  (`merge_queue.py` + tests). The structural change: in-flight deque, dispatch stage
  (Mechanism 1 release, Mechanism 2 staleness, abandoned/halt drains, host acquire,
  abort-poll verify task), finalize stage (in-order CAS + `_finalize_advanced_merge`;
  head-failure → abort all downstream via allocator + `_remerge` + re-dispatch;
  operator halt → abort ALL in-flight + requeue; trains/skip_verify as no-host
  pass-throughs; warm-swap unchanged inside the local-host verify; finalize-time
  rebase-reverify re-acquires a host — no deadlock: in-flight verifies complete
  independently). **Signal:** with two fake slow runners, two merge-verify spans
  overlap in the event log while main advances in submission order; N-fail
  mid-overlap aborts N+1's in-flight verify, re-merges, re-verifies, and both
  outcomes resolve correctly; single-host config runs the suite byte-identically.
  **Consumer:** ε, ζ, the throughput goal.
- **δ — Ledger-aware ENOSPC prune** (`git_ops.py` + `merge_queue.py` call sites +
  tests). `prune_stale_merge_worktrees(keep=...)` takes a keep-SET; the worker passes
  its 1728 ledger snapshot at both call sites (:545, :744); orphans (not in ledger,
  not persistent) still pruned. **Signal:** with two live in-flight worktrees +
  queued ledger worktrees + one orphan, a disk-pressure prune removes ONLY the
  orphan. **Consumer:** γ (overlap safety), the ENOSPC recovery path.
- **ε — Multi-verify observability** (`merge_queue.py` + tests).
  `_verify_item`/`_verify_phase` → ordered in-flight collection in `snapshot()`
  (back-compatible shape for existing consumers — reify 3112 tool); queue heartbeat
  line gains in-flight count + per-host occupancy. **Signal:** snapshot lists N
  in-flight items with per-item phase + host; heartbeat line shows occupancy;
  existing snapshot consumers' fields still present. **Consumer:** operators, ζ.
- **ζ — Overlap boundary gate (B+H integration tests)** (tests). B1 overlap + ordered
  advance; B2 chain-invalidation under overlap (head-fail → downstream abort →
  re-merge → re-verify → correct outcomes); B3 host-down mid-overlap (quarantine +
  local continuation, zero stall); B4 cancel-confirm slot release / cancel-fail
  quarantine; B5 operator halt aborts all + requeues; B6 ENOSPC prune protects live
  set (δ); B7 single-host byte-identical regression; B8 heartbeat covers all
  in-flight+queued worktrees under overlap (1728 regression). **Signal:** all eight
  green; B7 demonstrably exercises the no-runner config through the new code path.
  **Consumer:** the operator enable checklist — the go-signal that flipping C now
  buys real overlap.

- **η — Automated Lever C activation** (ops; reify repo config + service restart —
  **user-authorized 2026-06-11, restarts explicitly permitted**). Steps: (0) preflight
  `ssh leo-laptop true`; unreachable → abort + escalate (don't flip toward a dead
  remote). (1) Sync the laptop df checkout (path via the `/usr/local/bin/orchestrator`
  wrapper; `git pull --ff-only` — ships the α cancel contract; venv is editable).
  (2) Edit reify `orchestrator.yaml`: enable the staged `verify_runners` block
  (values at :119-133: laptop / leo-laptop / leo-laptop /
  `~/.config/orchestrator/reify-laptop.yaml`), `verify_drift_check_every_n_lands: 20`,
  the staged `sccache:` block; **delete the obsolete incident-revert paragraph** (the
  disproven per-host hypothesis). Update the gap-report enable checklist (steps 2-3
  done). (3) Dirty-tree safety: if reify has OTHER tracked modifications, abort +
  escalate (never commit over live WIP); else commit. (4) Quiet-window restart: poll
  the reify merge state (escalation port 8100 — python-urllib JSON-RPC shim, curl is
  broken on this host) for no in-flight merge verify, bounded ≤90 min, then
  `systemctl --user restart orchestrator-reify.service` (timeout → restart anyway,
  authorized). (5) Hard validation: journal shows "Speculative merge worker started",
  no `MergeLivenessConfigError`, NRestarts stable ≥10 min, and a df-venv assertion
  `len(load_config(<reify yaml>).enabled_verify_runners) == 1` (K=2). (6) sccache
  best-effort: `sccache --stop-server` on each host only when no build is running
  (next compile auto-starts the server with `SCCACHE_REDIS` from the verify env);
  busy → skip with a loud note (local-cache degradation is safe; drift detector is
  the standing guarantee). (7) Bounded observation ≤60 min: first `runner=laptop`
  merge-verify event in the journal — **reported, not gating** (depends on natural
  traffic; G6). **Signal (hard):** flipped config committed in reify; reify
  orchestrator restarted and stable on K=2; obsolete comment gone; checklist updated.
  **Consumer:** the merge-throughput goal — completes Lever C end-to-end.

**DAG:** α → β; δ → β (linearizes `merge_queue.py` edits); β → γ; γ → ε; ε → ζ; ζ → η.
(α ∥ δ run first in parallel — disjoint files.)

## 9. Open questions (tactical)

- pgid-file directory on each host (somewhere under the host's run/cache dir; α picks
  and documents).
- Exact snapshot back-compat strategy in ε (keep legacy singular fields mirroring the
  deque head vs version the shape).
- Merge-ahead buffer K+1 (measure the dispatch bubble first; follow-up if >2% of
  wall-clock).
- Whether the allocator probe interval / quarantine-clear cadence reuses the drift
  detector's cadence config or a new constant.
