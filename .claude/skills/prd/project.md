# Dark-factory PRD overlay

Project specialization for the generic `/prd` skill (source `skills/prd/` in this repo,
exposed via `~/.claude/skills/prd`). The generic skill reads this file at Step 0 and applies
it as authoritative extensions/overrides to its gates. This is dark-factory's FIRST overlay
(created 2026-08-28) — before it, this repo ran the generic skill with uncalibrated defaults.

## Identity & paths

- **project_id:** `dark_factory`
- **project_root:** `/home/leo/src/dark-factory`
- **PRD path:** `plans/<slug>-prd.md` (design docs and PRDs both live in `plans/`)

## Task sizing bands (PROVISIONAL 2026-08-28 — recalibration owned by task 4827)

Decompose-mode overrides, applied when producing the leaf batch:

- **Target leaf: ~300–1,500 changed LOC of expected scope, ≤10–12 declared files.** Prefer
  fewer, larger leaves within this band over many small ones.
- **Floor: do not author leaves under ~100 LOC expected scope** unless the work is genuinely
  `complexity='simple'` or `task_kind='deterministic'`. Measured basis (2026-08-27
  investigation, n=1,307 landed df tasks, replicated on reify n=1,179): sub-100-LOC tasks
  spend ~80% of budget on plan+review scaffolding and ~17% executing; per-task fixed overhead
  ≈ $7.66 against $1.82 marginal per kLOC; cost per delivered 100 LOC falls ~20× monotonically
  from the smallest to the largest size bucket.
- **Review trigger, not a cap: >15 declared files.** In df history that bucket degraded hard
  (60% blocking-escalation rate, n=25); in reify the same threshold is soft/continuous. Treat
  crossing it as a prompt to re-examine the leaf's coherence, not as a mechanical split rule.
- **Coalesce sibling froth:** a chain of tiny same-module leaves should be one leaf.
  Same-file serialization is unchanged and load-bearing — two leaves editing the same file
  must be serialized by real `add_dependency` edges (worktree rebase collisions).
- **Keep integration-gate leaves** where the decomposition carries one; sizing bands never
  delete the gate.

Provenance: memory `project_task_sizing_investigation_2026_08_27` (overhead economics,
sub-linear risk per delivered scope, the E4 re-test). These bands are provisional — the
observational basis cannot cleanly separate size effects from model-era/ops confounds.
**Task 4827** (parked `deferred`) re-derives them once prospective evidence exists (the E3
granularity A/B in reify, the 2026-08-28 backlog triage-and-coalesce sweep, and ≥30 days of
operation in the coarser regime); it updates this section and dates the new calibration.

## Cross-package scope (rule amended 2026-08-28)

A single task MAY span `orchestrator/` + `fused-memory/` (+ `dashboard/`, `shared/`). The
2026-05-13 hard pre-split rule is RETIRED: its founding failure — task 1229's $12.07
architect cap-death with a package half missing from the plan — does not reproduce under the
current architect (E4 re-test 2026-08-27: 4/4 combined cross-package specimens planned both
halves completely; df's architect `max_turns` was raised 120→180 the same week). Split only
at a genuine contract seam with multiple consumers (ordinary G5/B+H logic). When a split IS
chosen, the owning package's task carries the canonical contract and the consumer sibling
depends on it, duplicating shared schema via idempotent `CREATE TABLE IF NOT EXISTS`.
Tripwire: a plan that saturates even the 180-turn architect cap means the leaf is over-scoped
— decompose, don't retry (the task-2169 lesson).

## Memory namespace

`project_id="dark_factory"`. Load-bearing records for this overlay:
- `split-multi-package-tasks` (file memory, amended 2026-08-28) — the cross-package rule.
- `project_task_sizing_investigation_2026_08_27` — the evidence base for the sizing bands.
