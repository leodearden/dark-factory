# Capability manifest — interactive-warm-worktree PRD

Mechanizes G3+G6 per **leaf** task. One block per leaf: `capability → evidence`.
PASS evidence: `grep:<file>:<line>` (wired on main) or `producer:task-<label>` (upstream in the dependency closure). Any FAIL value (`declared-only|test-only|producer-downstream|producer-absent|producer-extent-short|fixture-ERROR|bound≤floor|rejection-absent`) blocks the batch.

Leaves: **ε** (`/do` wiring), **η** (`/warm` skill), **ζ** (integration gate). Intermediates α/β/δ carry no leaf signal (they name downstream consumers) — not manifested.

---

## ε — wire `/do` to prefer warm claim (cold fallback)

Signal: running `/do` lands the session in a warm `_iact-*` worktree (first build near-zero recompile on a project with a warm base), cold-falls-back to `EnterWorktree` when the orchestrator is unreachable.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `claim_warm_worktree` escalation verb exists & returns `{path,branch,warm,base_ref}` | `producer:task-β` (upstream; direct dep) | PASS |
| `create_interactive_worktree` primitive + CoW-seeded `_iact-*` worktree | `producer:task-α` (transitive upstream via β) | PASS |
| `EnterWorktree` cold-fallback path | `grep:skills/do/SKILL.md:40` (current worktree step, harness builtin) | PASS |
| DAG-direction (producers upstream of ε) | β, α both upstream of ε | PASS |

## η — explicit-only `/warm` skill

Signal: typing `/warm` lands the session in a warm `_iact-*` worktree (first build near-zero recompile where a warm base exists); with the orchestrator down it reports the fallback rather than erroring.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `claim_warm_worktree` escalation verb | `producer:task-β` (upstream; direct dep) | PASS |
| CoW-seeded `_iact-*` worktree | `producer:task-α` (transitive upstream via β) | PASS |
| Explicit-only skill-trigger discipline (never auto-invoke) | `grep:skills/do/SKILL.md:1` — `/do` is the existing explicit-only exemplar ("ONLY runs when the user explicitly types /do — never auto-invoke it") to mirror | PASS |
| DAG-direction | β, α upstream of η | PASS |

## ζ — integration-gate boundary test (isolation I1 + reap I2)

**G6 scoping (resolution — signal moved to where it is producible).** The end-to-end warmth observable ("warm `target/` → near-zero recompile") is a **reify-side** truth (reify's cargo `target/`); dark-factory's own Python CI has **no** cargo build cache, so a dark-factory integration test **cannot** produce that assertion. Per G6 resolution (a), ζ's **CI** signal is scoped to the invariants that ARE producible in dark-factory's suite; the warmth observable is verified out-of-band on **reify** at the deferred deploy capstone (PRD §Open questions #3). This avoids a `producer-extent-short` FAIL from asserting reify-only behaviour in a dark-factory test.

ζ CI signal (producible): boundary test asserts I1 (pool FREE + `scheduler._dispatched` unchanged by claim/release), I2 (stale `_iact-*` reaped, within-TTL live preserved), claim/release roundtrip, and `_seed_warm_lane` invoked with `--fresh-checkout` (fail-soft when the seed script is absent).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `claim_warm_worktree` / `release_warm_worktree` verbs | `producer:task-β` (upstream; direct dep) | PASS |
| `create_interactive_worktree` + `_iact-*` band + I1 isolation | `producer:task-α` (transitive upstream via β) | PASS |
| interactive-worktree reaper (I2) | `producer:task-δ` (upstream; direct dep) | PASS |
| `_seed_warm_lane('--fresh-checkout')` CoW-seed is invoked | `grep:orchestrator/src/orchestrator/git_ops.py:1227` (wired production seed method; `--fresh-checkout` mode at `git_ops.py:1763`) | PASS |
| Observe WarmLanePool FREE-lane count (to assert I1) | `grep:orchestrator/src/orchestrator/warm_lane_pool.py` (`_lanes: dict[Path,LaneState]` + `assignments_snapshot()` — FREE count = lanes not ASSIGNED, test-accessible) | PASS |
| End-to-end "warm target/ → near-zero recompile" | **NOT bound in dark-factory CI** — deferred to reify post-deploy verification (PRD §Open questions #3); out of ζ CI scope by design | N/A (reify-side) |
| DAG-direction | β, δ, α all upstream of ζ | PASS |

---

**Result: no FAIL bindings.** The one extent nuance (reify-only warmth observable) is resolved by scoping ζ's CI signal to the producible invariants and deferring the warmth check to the reify deploy. Batch may queue.
