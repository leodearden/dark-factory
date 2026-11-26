# Dark-factory PRD overlay

Project specialization for the generic `/prd` skill (source `skills/prd/` in this repo,
exposed via `~/.claude/skills/prd`). The generic skill reads this file at Step 0 and applies
it as authoritative extensions/overrides to its gates. This is dark-factory's FIRST overlay
(created 2026-08-28) — before it, this repo ran the generic skill with uncalibrated defaults.

## Identity & paths

- **project_id:** `dark_factory`
- **project_root:** `/home/leo/src/dark-factory`
- **PRD path:** `plans/<slug>-prd.md` (design docs and PRDs both live in `plans/`)

## Task sizing bands (calibrated 2026-11-19, task 4827)

Decompose-mode overrides, applied when producing the leaf batch:

- **Target leaf: ~500–3,000 changed LOC of expected scope.** Prefer fewer, larger leaves
  within this band over many small ones. The upper figure is a target, not a cap.
- **Floor: do not author leaves under ~100 LOC expected scope** unless the work is genuinely
  `complexity='simple'` or `task_kind='deterministic'`.
- **Many declared files: expect a slow landing.** A leaf declaring more than ~12 files holds
  more locks for longer and contends with more of the queue, so it tends to land late. Treat
  >15 declared files as a prompt to re-examine the leaf's coherence and lock footprint, not
  as a mechanical split rule.
- **Coalesce sibling froth:** a chain of tiny same-module leaves should be one leaf.
  Same-file serialization is unchanged and load-bearing — two leaves editing the same file
  must be serialized by real `add_dependency` edges (worktree rebase collisions).
- **Keep integration-gate leaves** where the decomposition carries one; sizing bands never
  delete the gate.

Measured basis (dark-factory, 1,120 tasks implemented by claude-opus-5-5 and landed
2026-10-07..11-19; cost is summed `invocations.cost_usd`; the same shapes hold in reify):

| Changed LOC | n | Cost per kLOC | Blocking escalations per kLOC | Share of tasks with ≥1 |
|---|---|---|---|---|
| ≤100 | 192 | $118 | 4.1 | 12% |
| 101–300 | 246 | $45 | 1.1 | 15% |
| 301–500 | 152 | $32 | 0.9 | 25% |
| 501–1,500 | 287 | $22 | 0.6 | 32% |
| 1,501–3,000 | 140 | $15 | 0.4 | 44% |
| >3,000 | 101 | $7 | 0.2 (0.4 with each task's LOC capped at 3,000) | 49% |

- Judge a split by outcomes per delivered line, not per task: a larger leaf escalates more
  often, but the same scope cut into smaller leaves escalates more in total. Escalations per
  line fall steeply up to ~500 LOC and are flat from there past 3,000.
- Cost per line keeps falling through the whole range, also when the spend on dispatched but
  unlanded tasks is counted.
- Declared-file count is a gradient (≥1 blocking escalation: 24% / 29% / 41% / 52% at ≤5 /
  6–12 / 13–15 / >15 files), mostly explained by size. There is no cliff at 15.
- The architect's 180-turn cap did not bind (p99 122, max 157 of 3,847 runs); the tripwire in
  the next section stands.

Not measured: defects that escape after landing, by leaf size. A 2026-11-19 attempt found no
sign of a rise above 1,500 LOC, but the factory records too little to tell (no fix task names
the task that caused its defect). Line count is also a weak measure of scope above ~1,500 LOC
(moves, data files). Levels shift with the model: re-measure after a model change before
quoting these figures.

Provenance: this calibration replaces the provisional 2026-08-28 bands (300–1,500 LOC,
≤10–12 files). Their cost line ($7.66/task + $1.82/kLOC) was read from `task_results`, which
records about 70% of invoiced cost, and their ">15 files: 60% escalation" figure did not
replicate. The reify E3 granularity experiment that was to inform this was stood down on
2026-11-19 without a result
(`/home/leo/src/reify/docs/prds/v0_6/e3-decomposition-granularity-ab.md`, closing addendum).

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

## Leaf review vocabulary

Every leaf is authored and reviewed against `docs/code-quality.md` — quality is the cost
and risk of the next agent change; fourteen named heuristics with agreed readings. Cite the
heuristics by name in a leaf's brief where they bind, especially *files make internal sense
in isolation* and *no file too large — no cheating by stitching files together with imports*
for any leaf that splits or extracts a module. G7 still gates against
`docs/legibility/design-invariants.md`, the checkable subset; the quality document is the
definition those invariants serve, and it is the single normative copy — point at it, never
restate it in a PRD.

A PRD that remediates findings from a quality instrument (`/review`, `/hotspot-survey`,
`/census`, `/review-all`) carries each finding's key, and every leaf filed from it carries
`metadata.x_finding_key` (list) and `metadata.x_finding_run` per
`docs/quality-findings-contract.md` §8, so the next run of that instrument finds the work
instead of re-filing it. Decompose adds the program's trigger-chain gate as a dependent of
the batch's `high` and `critical` leaves (`skills/_shared/filing-the-trigger-chain.md` §4).

## Memory namespace

`project_id="dark_factory"`. Load-bearing records for this overlay:
- `split-multi-package-tasks` (file memory, amended 2026-08-28) — the cross-package rule.
- fused-memory search "task 4827 sizing bands recalibration" — the evidence base for the sizing bands
  (supersedes `project_task_sizing_investigation_2026_08_27`).
