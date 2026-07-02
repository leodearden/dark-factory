# PRD — Interactive warm worktree (claim a warm build environment from an interactive session)

**Status:** deferred · authored 2026-07-02 · greenfield feature

> **⚠ Fork status (author was AFK at authoring time):**
> 1. **Mechanism = "fresh CoW-seeded worktree, no pool"** — **DEFAULTED, still overridable** (rejected: reserved-band pooled lane, shared-budget pooled lane). See *Resolved design decisions* §1.
> 2. **Consumer = escalation verb + wire `/do` + a dedicated explicit-only `/warm` skill** — **RESOLVED by user 2026-07-02** (verb-only rejected; `/do` wiring and the `/warm` skill both adopted). See §2.

---

## Goal

An interactive session (a human or a Claude interactive session — e.g. working on **reify**) can obtain a git worktree whose **build cache is already warm**, instead of paying a full cold build. It calls one MCP verb, gets back a ready-to-use worktree path on a `task/<slug>` branch with `target/` CoW-reflinked from the orchestrator's rolling warm base, works in it, merges via `/merge-queue`, and the worktree is cleaned up on release / merge / crash.

The user-observable win: the **first** `cargo build`/test in the returned worktree does near-zero recompilation (it inherits the warm base's artifacts), where a cold `EnterWorktree` worktree would rebuild from scratch.

## Background

This PRD answers the question "*is there a way for an interactive session to claim a warm build lane?*" — investigated 2026-07-02. Findings that shape the design:

- The orchestrator's warm-lane **pool** (`WarmLanePool`, `orchestrator/src/orchestrator/warm_lane_pool.py`) is an in-memory FREE/ASSIGNED state machine whose only claim entry point is `GitOps.acquire_warm_lane` (`git_ops.py:1752`), reached **only** from workflow dispatch (`workflow.py:1497`). Every allocation is keyed to a `task/<id>` branch. There is no external claim surface.
- Crucially, **dispatch capacity and lane accounting are decoupled**: dispatch is gated by `Semaphore(max_concurrent_tasks)` + `scheduler._dispatched`, *not* by the pool's FREE count (`harness.py:1114,1139,1251`). A non-task holder of a pool lane is therefore invisible to dispatch → beyond the `spare_warm_lanes` buffer the scheduler over-dispatches and bounces tasks with `WarmLanePoolExhausted`, and under exhaustion `reclaim_victim` (`warm_lane_pool.py:148`) can **force-steal** a lane out from under a live non-dispatched holder (its "non-dispatched ⇒ stale" invariant, `harness.py:2056`). A pooled interactive claim also **leaks** if it has no backing DB task (filtered out of the reclaim candidate set), surviving only until an orchestrator restart.
- The "warm" payload is the **build cache**: `_seed_warm_lane(dir, mode)` (`git_ops.py:1227`) runs `<dir>/scripts/seed-warm-lane.sh <base_target> <dir> <mode>` to **CoW-reflink `target/`** from `self.warm_lane_base_target_path` — already resolving reify's symlinked gen-dir and holding a `flock -s` reader-refcount lock against the concurrent `refresh-warm-base.sh` GC. A CoW reflink is near-instant, and `git worktree add` is cheap.

The last two facts are the whole design: **a fresh worktree that CoW-seeds from the warm base gets essentially the same warmth as an existing pool lane, at a tiny extra cost, without joining the pool** — sidestepping every capacity/reclaim/restart hazard above. So we do **not** claim a pool *lane*; we mint an isolated warm *worktree*.

The control-plane transport already exists: the escalation MCP server is constructed **in-process** with the orchestrator and closes over a live `harness` handle (`harness.py:5330`), served over HTTP on the project's escalation port. Its `@mcp.tool()` closures already reach `harness.git_ops` (exactly how `merge_request`/`get_merge_queue`/`halt_scheduler` work). A new claim/release verb attaches here with **no new transport**.

## Sketch of approach

A new **`_iact-*` worktree band**, strictly disjoint from the pool's `_lane-*` and `_spec-*` bands, minted and reaped by `GitOps`/`Harness`, exposed through two escalation MCP verbs, consumed by `/do`.

- **`GitOps.create_interactive_worktree(slug, *, start_ref=None)`** — `git worktree add -b <branch_prefix><slug> <worktree_base>/_iact-<slug> <start_ref|current main>`, then `await self._seed_warm_lane(path, '--fresh-checkout')` (reused verbatim — it reads the seed script from the worktree's own checked-out tree and inherits the gen-dir/flock safety). Writes an `.task/interactive.json` stamp (session owner, created-at) for the reaper. Enforces a `max_interactive_worktrees` cap. **Never** calls into `WarmLanePool`. Fail-soft: seed absent/failed → `warm=False`, worktree still usable (cold).
- **Escalation verbs `claim_warm_worktree` / `release_warm_worktree`** — thin `@mcp.tool()` closures over `harness` that call the primitive / `git worktree remove --force` (+ prune branch if already merged).
- **Reaper** — a periodic + startup sweep in `Harness` (folded into the existing warm-lane GC cadence) that removes `_iact-*` worktrees whose branch landed on main, or whose stamp exceeds `interactive_worktree_ttl` with no commit since, or under disk pressure (reusing the warm-lane disk guard). This is the crash-safety net the pool's `_recover_crashed_tasks` does **not** provide for non-lane worktrees.
- **`/do` consumer** — prefer the warm claim, cold-fall-back to `EnterWorktree` when the escalation MCP is unreachable.

Per-project by construction: each project's orchestrator serves the verb on its own escalation port and seeds from its own `warm_lane_base_target_path`; a project with no `seed-warm-lane.sh` degrades to a plain (cold) worktree. reify is the motivating consumer (cargo `target/`).

## Resolved design decisions

1. **Isolated warm worktree, NOT a pool lane** *(defaulted — fork 1)*. The pooled options (reserved band / shared budget) buy only the saved `git worktree add` + one reflink, while dragging in dispatch-capacity coupling, force-steal exposure, and restart bookkeeping. Because CoW-seed + `worktree add` are both cheap, the isolated path delivers the same warmth at far lower blast radius. The `_iact-*` band is **invariantly disjoint** from `_lane-*`/`_spec-*` and never affects dispatch capacity.
2. **Escalation verb + wire `/do` + a dedicated explicit-only `/warm` skill** *(fork 2 — resolved by user 2026-07-02)*. Two consumer surfaces over the same verbs: **(a)** `/do` **prefers** the warm claim and **cold-falls-back** to `EnterWorktree` when no orchestrator/base is reachable, so the autonomous hand-off path is warm without losing robustness; **(b)** a new **`/warm` skill** — **explicit-only** (triggers solely on a typed `/warm`, never auto-invoked, mirroring `/do`'s trigger discipline) — gives an interactive session a first-class "put me in a warm worktree now" command for ad-hoc work that does not go through `/do`. Both name `/merge-queue`-compatible `task/<slug>` branches. Trade-off accepted: a claimed `_iact-*` worktree is a raw worktree the Claude Code harness does not track (no `ExitWorktree`); cleanup is owned by the release verb + reaper, and `/merge-queue` is branch-based so it is unaffected.
3. **Branch identity `task/<slug>`** (via `config.branch_prefix`), so `/merge-queue` lands it unchanged; `.task/interactive.json` stamp gives the reaper an owner + age without needing a DB task.
4. **Reuse `_seed_warm_lane` verbatim** — do not fork the seed logic; the interactive worktree gets reify's symlink-gen-dir resolution and `flock` reader-refcount GC safety for free.
5. **Fail-soft warmth** — absent/failed seed never blocks the claim; the verb returns `warm=False` and a usable cold worktree.

## Pre-conditions for activating

- No upstream code prerequisite — every substrate capability the design leans on exists on main (`_seed_warm_lane`, `warm_lane_base_target_path`, in-process escalation `harness` handle, `git worktree`). G3 verified below.
- **Deploy is a deferred deterministic follow-up.** The live consumer for reify is a running orchestrator that serves the new verbs; that requires restarting the reify (and df) orchestrators. File a `task_kind='deterministic'` deploy capstone depending on ζ **only after** ζ lands and a restart script exists (`scripts/restart-all-orchestrators.sh`, incoming from the merge-queue-refactor batch) — deterministic `before_done.script` existence is validated at `submit_task` time, so it cannot be filed earlier. See *Open questions*.

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/warm-lane-branch-lifecycle-decouple-prd.md` | consumes | `GitOps._seed_warm_lane` + `warm_lane_base_target_path` (read-only reuse) | that-PRD owns the primitive; this-PRD is a read-only consumer | no contested ownership — this PRD never mutates pool state |

No new cross-PRD seam is introduced. The isolation invariant (§Contract) is precisely what keeps this PRD from touching the decouple PRD's `_lane-*` territory.

## Contract (the one load-bearing invariant — light H)

**Isolation invariant (I1):** an interactive `_iact-*` worktree MUST NOT
- occupy or appear in any `WarmLanePool` (`_lane-*`) or spec-pool (`_spec-*`) slot,
- be counted against `Semaphore(max_concurrent_tasks)` or `scheduler._dispatched`,
- be reachable by `reclaim_victim` / `_reconcile_terminal_lanes` / `_recover_crashed_tasks`.

Corollary: claiming or releasing an interactive worktree leaves the pool's FREE lane count and the scheduler's dispatch capacity **unchanged**. This is the boundary assertion ζ must prove.

**Reaper contract (I2):** every `_iact-*` worktree is eventually removed by exactly one of {explicit `release_warm_worktree`, branch-landed-on-main sweep, TTL-with-no-commit sweep, disk-pressure eviction} — no path leaks a worktree past `interactive_worktree_ttl` + one sweep interval.

## Boundary-test sketch (ζ's signal)

| Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|
| Warm claim populates cache | orchestrator up, warm base seeded | returned worktree's `target/` is CoW-populated; a build in it recompiles near-zero vs a cold-control worktree |
| Isolation (I1) | pool has K FREE lanes | after claim+release, pool FREE count == K; `scheduler._dispatched` unchanged; `_iact-*` never in pool maps |
| Cold fallback | escalation MCP unreachable | `/do` path falls back to `EnterWorktree`, session still gets a (cold) worktree, no error |
| Explicit release | a claimed `_iact-*` worktree | `release_warm_worktree` removes it; absent from `git worktree list` |
| Crash-leak reap (I2) | an `_iact-*` worktree whose stamp is older than TTL, no recent commit, session gone | next reaper sweep removes it; a within-TTL live one is preserved |

## Decomposition plan

Labels are placeholders; task IDs assigned at decompose. DAG: **α → {β, δ}; β → {ε, η, ζ}; δ → ζ.** Leaves: **ε, η, ζ**. Intermediates: **α, β, δ**.

- **α — `GitOps.create_interactive_worktree` primitive + config knobs** *(intermediate; unlocks β, δ)*
  - Modules: `orchestrator/src/orchestrator/git_ops.py`, `config.py`/`defaults.yaml`.
  - New: `_iact-<slug>` worktree band; `git worktree add` from start-ref; reuse `_seed_warm_lane(path,'--fresh-checkout')`; `.task/interactive.json` stamp; `max_interactive_worktrees` cap; knobs `max_interactive_worktrees`, `interactive_worktree_ttl`, `iact_prefix='_iact-'`.
  - Signal (unlocks β/δ): integration test — returned worktree has `target/` CoW-seeded AND `warm_lane_pool` FREE count is unchanged by creation (**I1**).
  - Capability bindings: `_seed_warm_lane` → `grep:git_ops.py:1227` (wired, production seed path) · `warm_lane_base_target_path` → `grep:git_ops.py:1263` · `git worktree add` → existing.

- **β — escalation verbs `claim_warm_worktree` / `release_warm_worktree`** *(intermediate; unlocks ε, ζ)*
  - Modules: `escalation/src/escalation/server.py`.
  - New: two `@mcp.tool()` closures over `harness` → `harness.git_ops.create_interactive_worktree(...)` / `git worktree remove --force` + prune-if-merged. Return `{path, branch, warm, base_ref}`.
  - Signal: an MCP call against a running orchestrator returns a warm worktree dict; release removes it (gone from `git worktree list`).
  - Capability bindings: in-process `harness` handle → `grep:harness.py:5330` (server built with live `harness=self`) · α's primitive → `producer:task-α` (upstream).

- **δ — interactive-worktree reaper + disk-guard integration** *(intermediate; unlocks ζ; depends α)*
  - Modules: `orchestrator/src/orchestrator/harness.py` (+ reuse `git_ops`/disk-guard).
  - New: periodic + startup sweep removing `_iact-*` worktrees by {branch-landed, TTL-no-commit, disk-pressure}; wired into the existing warm-lane GC cadence (`harness.py:~3532`).
  - Signal: a simulated stale/merged `_iact-*` worktree is removed on the next sweep (log line + absent from `git worktree list`); a within-TTL live one is preserved (**I2**).
  - Capability bindings: harness periodic loop → existing · warm-lane disk guard (`warm-lane-disk-guard.sh`) → verify present at decompose · α's stamp → `producer:task-α`.

- **ε — wire `/do` to prefer warm claim (cold fallback)** *(LEAF — G1 consumer; depends β)*
  - Modules: `skills/do/SKILL.md` (step 1, line 40).
  - Change: call `claim_warm_worktree` on the project's escalation MCP; on success `cd` into the returned path on the `task/<slug>` branch; add a release-on-session-end step; on unreachable/failure fall back to `EnterWorktree`.
  - Signal (user-observable): running `/do` with a live orchestrator + warm base lands the session in a warm `_iact-*` worktree whose first build recompiles near-zero; cleanly cold-falls-back to `EnterWorktree` when the orchestrator is down.
  - Capability bindings: β's verbs → `producer:task-β` (upstream) · `EnterWorktree` fallback → existing harness builtin.

- **η — explicit-only `/warm` skill** *(LEAF — G1 consumer; depends β)*
  - Modules: `skills/warm/SKILL.md` (new).
  - New skill whose description is **explicit-only** — triggers ONLY on a typed `/warm`, never auto-invoked (mirror `/do`'s "ONLY runs when the user explicitly types /do — never auto-invoke it" language). On invocation: call `claim_warm_worktree` on the project's escalation MCP, `cd` into the returned `task/<slug>` worktree, surface `warm`/`base_ref` to the user, register release-on-session-end; on unreachable/failure print a clear cold-fallback note (never silent).
  - Signal (user-observable): typing `/warm` lands the session in a warm `_iact-*` worktree whose first build recompiles near-zero; with the orchestrator down it reports the fallback rather than erroring.
  - Capability bindings: β's verbs → `producer:task-β` (upstream) · skill-trigger discipline → mirror `skills/do/SKILL.md`.

- **ζ — integration-gate boundary test** *(LEAF — the H integration gate; depends β, δ)*
  - Modules: `orchestrator/tests/` (new end-to-end test/example).
  - Proves the Boundary-test sketch rows end-to-end: claim → warm target/ → near-zero-recompile vs cold control → **I1** (pool FREE + dispatch capacity unchanged) → release removes → reaper cleans a simulated crash leak (**I2**).
  - Signal: the boundary test passes in CI, exercising claim/release/reap and the isolation invariant. This is the C-as-integration-gate leaf roping α/β/δ into a user-observable pass.

## Out of scope for this PRD

- Reusing an **actual** pool `_lane-*` lane from an interactive session (reserved-band / shared-budget pooled claim) — rejected in §1; would need dispatch-capacity coupling.
- Warming non-`target/` artifacts (e.g. a Python `.venv` for a non-Rust project) — the feature degrades to cold there; per-project seed scripts own what "warm" means.
- The deterministic **deploy** capstone that restarts orchestrators to make the verbs live — deferred follow-up (see Pre-conditions + Open questions).
- Any change to `WarmLanePool`, dispatch, or the pool's reclaim/reconcile machinery.

## Open questions (tactical — surfaced, not decided)

1. **`max_interactive_worktrees` default.** Suggested `2`. Decide during α; tune per-project via config.
2. **`interactive_worktree_ttl` default.** Suggested `24h` with "no commit since" as the idle discriminant. Decide during δ.
3. **Deploy capstone.** File a `task_kind='deterministic'` deploy (restart reify+df orchestrators via `scripts/restart-all-orchestrators.sh`) depending on ζ — **only after** ζ lands and the restart script exists (submit-time `before_done.script` existence validation). Decide when ζ is done.
4. **Release-on-session-end mechanism for `/do`.** Whether `/do` calls `release_warm_worktree` explicitly at `/reflect`, or leans entirely on the reaper. Suggested: explicit release when clean, reaper as the safety net. Decide during ε.
