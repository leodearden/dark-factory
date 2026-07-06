# PRD: worktree-lane-lifecycle — authoritative durable lane state + `.task/` out of the git tree

**Status:** active — 2026-07-06. Stream **W11** (wave 2) of the bug-hotspot
remediation program (`plans/bug-hotspot-remediation-program-2026-07-06.md`).
**Approach B+H** (high-stakes: task-state integrity across crashes). Builds on
stream **M1** (`plans/gitops-chokepoints-prd.md`).

## Goal

Give the warm-lane pool **one authoritative durable state record per lane** written
through **one writer**, and move the `.task/` execution sidecar **out of the git
worktree** so contamination of `main` is *structurally impossible* rather than
defended by a stack of redundant guards. Concretely, after this PRD an operator
observes:

1. **Contamination is structural.** A hostile `git add -A && git commit` (or
   `git add -A` by an agent) inside a warm lane **cannot stage task metadata** —
   the `.task/`-class artifacts are not in the git tree. No scrub, no pathspec
   exclusion, no `.gitignore` is load-bearing for this anymore.
2. **Lane identity survives cleaning.** `git clean -xfd` / `git checkout -f` inside
   a lane leaves both the lane's durable state record and its task metadata intact
   (they live under `<worktree_base>/.lane-state/` and `<worktree_base>/.task-meta/`,
   siblings of the lane dir, not inside it).
3. **Recovery reads truth, then quarantines on divergence.** On restart, startup
   recovery reads each lane's durable record, verifies it against git reality, and
   **quarantines** any lane whose record and git state disagree — it **never
   silently re-pins** a stale lane (killing the 2097/2098 re-poisoning class) and
   **never silently heals** an illegal state.
4. **Every transition goes through one writer.** A lane moves
   `seed → registered → assigned → in-use → released` through a single
   `LaneLifecycle` writer; any illegal transition (e.g. `released → in-use`)
   **asserts and escalates born-at-L2**, it does not self-repair.

## Background / evidence (git-worktrees hotspot, survey 2026-07-06)

The survey's **git-worktrees hotspot** (`git_ops.py` + `warm_lane_pool.py` +
`worktree_identity.py`) names four structural findings this PRD closes:

- *"Warm-lane lifecycle has no authoritative durable state — five derivable sources
  reconciled ad hoc at every decision point."* Lane state today is in-memory only
  (`WarmLanePool._lanes` / `_assignments`, `warm_lane_pool.py`), reconstructed on
  startup from git registration + `.task/plan.json` + the `.pool-root` sentinel +
  branch reality + pool config. Each of the historical incidents (2097 orphaned
  registration with a surviving assignment; 2098 stale `plan.json` re-pinning;
  detached-HEAD stale-branch collision, task 2062) is **one unhandled cell of that
  cross-product** in `_recover_crashed_tasks` (`harness.py:1948-2287`).
- *".task/ sidecar lives inside the git worktree — one leak class defended by a stack
  of redundant guards instead of removed."* The `git_ops.py` module docstring
  (`:3-24`) enumerates the defense layer (see §Guard inventory below). Contamination
  incidents: de7398eb91, 0157796b74, 13e8eca1e5, reify esc-4920-163.
- *"Pool-storage presence is a scattered sentinel protocol with one hidden writer and
  duplicated bootstrap escapes."* The `.pool-root` sentinel
  (`POOL_ROOT_SENTINEL`, `git_ops.py:175`) is read at 6+ sites, has exactly one
  writer (`_seed_warm_lane` → `mark_pool_storage_present`, `git_ops.py:955-975`),
  and two bootstrap escapes (`_pool_storage_bootstrap_ok`, `git_ops.py:994-1046`).
- *"acquire_warm_lane is a 7-route god-function whose sibling routes silently
  diverge."* `acquire_warm_lane` (`git_ops.py:2502-3065`) has 7 substantive routes
  with divergent teardown idioms; M1's `_abort_lane_acquisition` unifies only the
  **fault** exits, deliberately leaving the route classifier to W11.

### Substrate reality (G3, code-verified 2026-07-06)

| Capability | Location | Status |
|---|---|---|
| `quarantine_base` property | `git_ops.py:6687` (`worktree_base.parent/<name>-orphaned`) | EXISTS (sibling, off-mount) |
| `quarantine_worktree()` | `git_ops.py:6733` | EXISTS |
| `.pool-root` sentinel + `pool_storage_present/mark/_note/_bootstrap_ok` | `git_ops.py:175, 930-1046` | EXISTS |
| `worktree_base` | `git_ops.py:850` (`project_root/config.worktree_dir`, default `.worktrees`) | EXISTS |
| `TaskArtifacts` (root `= worktree/'.task'`) | `artifacts.py:160-775` | EXISTS (primary, **not sole**, `.task/` builder) |
| `_recover_crashed_tasks` heuristic tree | `harness.py:1948-2287` | EXISTS |
| `restore_assignment` / `note_assignment` / `drop_assignment` | `warm_lane_pool.py:226-266` | EXIST |
| dashboard `read_task_artifacts(worktree_path)` | `dashboard/src/dashboard/data/orchestrator.py:170-236` | EXISTS (reads `.task/`) |
| Guard layer (`scrub_task_dir_from_tree`, `_assert_no_task_dir`, `commit()` `:!.task` + unstage net, merge_gates `:!.task/` ×4, gitignore writers) | see §Guard inventory | EXISTS |
| M1 `_prune_registrations` chokepoint | task **2185** (in-progress) | **prerequisite** |
| M1 `_abort_lane_acquisition` | — | **ABSENT** (M1 δ not filed) — prerequisite |
| M1 `PROTECTED_PREFIXES` registry | — | **ABSENT** (M1 ε not filed) — prerequisite |

### Guard inventory (what M2 makes dead code)

Verified sites the `.task/`-relocation renders unnecessary (`_assert_no_task_dir` is
retained as a cheap migration-window tripwire, then dropped in the final leaf):

1. `scrub_task_dir_from_tree` (`git_ops.py:244-325`) — call sites `1625`
   (create_worktree), `3044` (acquire tail), `4782` (post-merge), `5956` (merge).
2. `_assert_no_task_dir` (`git_ops.py:344-364`) — call site `5861` (pre-advance-main).
3. `commit()` `:!.task` staging exclusion (`git_ops.py:3860`) + post-staging unstage
   net (`3866-3878`).
4. `merge_gates.py` four `:!.task/` pathspec exclusions (`1087, 1117, 1261, 1282`).
5. `.gitignore` writers: `_ensure_task_gitignore` (`git_ops.py:328-341`, called at
   `1515, 1615, 1824, 3043, 3120`) and `TaskArtifacts.init()` nested `.gitignore`
   (`artifacts.py:194-200`).
6. Assorted `:!.task` (no trailing slash) pathspecs in `git_ops.py`
   (`4501, 4597, 4663, 6054, 6059, 6477, 6481`) and agent-prompt admonitions
   (`agents/roles.py:277, 288, 342, 574, 843, 1134`).

The pre-commit hook (`hooks/pre-merge-commit`, task 7) stays — it is a repo-wide
merge guard, not a `.task/` guard.

## Consumers (G1)

Every mechanism has a named consumer:

- **`LaneLifecycle` writer + `.lane-state/<lane>.json` record** — consumed by
  `acquire_warm_lane`/`release_warm_lane` (GitOps writer side) and by
  `_recover_crashed_tasks` + `restore_assignment`/`note_assignment` (Harness
  reader/recovery side). It is the single source of truth `WarmLanePool`'s in-memory
  map becomes a cache of.
- **Legal-transition table** — consumed by every mutator (the assert+escalate fires
  on any caller attempting an illegal edge); user-observable surface is the born-at-L2
  escalation record on an illegal transition.
- **`TaskArtifacts` single path-derivation owner over `.task-meta/`** — consumed by
  `workflow.py` (execution), `harness.py` (recovery reads), `git_ops.py`
  (disk-backstop read + interactive stamp), `dashboard/data/orchestrator.py`
  (`read_task_artifacts`), and `agents`/`mcp_lifecycle` (the path handed to task
  agents via config). M3's `dashboard-alignment` stream **documents** this coupling
  (its task ζ format-coupling doc block) but explicitly **cedes derivation ownership
  to W11** (M3 PRD §G4).
- **Fold of the `.pool-root` sentinel protocol into `LaneLifecycle`** — consumed by
  the same acquire/recovery paths; removes the "scattered sentinel + hidden writer"
  finding.
- **7-route unification over `LaneLifecycle` transitions** — consumer is the acquire
  / requeue path: no divergent per-route teardown; every route is a named transition,
  and fault exits delegate to M1's `_abort_lane_acquisition`.

User-observable surfaces: the contamination test (a hostile `git add -A` cannot
stage task metadata), the crash-recovery quarantine log line, and the born-at-L2
illegal-transition escalation.

## Sketch of approach

### Mechanism 1 — `LaneLifecycle` single-writer + durable per-lane record

New module `orchestrator/src/orchestrator/lane_lifecycle.py`:

- **`LaneState`** enum: `SEED, REGISTERED, ASSIGNED, IN_USE, RELEASED, QUARANTINED`.
- **`LEGAL_TRANSITIONS`**: an explicit table (see §Contract). Any `(from, to)` not in
  the table → `IllegalLaneTransition` raised + a born-at-L2 escalation via the
  existing escalation client (sentinel `agent_role='harness-lane-lifecycle'`); the
  record is **not** mutated (never silent-heal).
- **Durable record** `<worktree_base>/.lane-state/<lane>.json`:
  `{state, task_id, title, branch, seeded_from_sha, updated_at}`. Written atomically
  (tmp-write + `os.replace`). Lives **on the pool mount** (a `worktree_base` child,
  sibling of the `_lane-*` dirs) so it shares the lanes' lifetime and vanishes with
  the mount — coherent with the `.pool-root` sentinel.
- **`transition(lane, to, **fields)`** — the ONE mutator: load record, validate the
  edge, write atomically, return the new record. Both GitOps (acquire/release) and
  Harness (recovery restore/note) call it.
- **Fold the `.pool-root` sentinel**: pool-storage presence becomes a `LaneLifecycle`
  concern (the sentinel is written/read only via the lifecycle module), collapsing
  the scattered protocol + hidden writer + duplicated bootstrap escapes into one
  place.
- **Startup recovery** (replacing the `_recover_crashed_tasks` cross-product): for
  each lane, read its record → verify git reality (registration present? branch/HEAD
  match?) → **adopt** on match, **quarantine** (via `quarantine_worktree`) on
  divergence. Each historical bug is one divergence cell now handled uniformly.

`WarmLanePool`'s in-memory `_lanes`/`_assignments` become a cache rebuilt from the
durable records on startup; `restore_assignment`/`note_assignment` route through
`transition(...)` so the durable record and the cache never drift.

**Scope:** the durable record covers the warm-lane pool (`_lane-*`). Spec-pool
(`_spec-*`), persistent `_merge-verify`/`_offline-deep`, and merge/solo bands are
**out of scope** for the record (the transition table is extensible to them in a
later PRD).

### Mechanism 2 — `.task/` out of the git tree

- **New base**: task artifacts move from `<worktree>/.task/` to
  `<worktree_base>/.task-meta/<worktree-name>/`. This is a sibling directory of the
  worktree (not inside it) → `git add -A`, `git clean -xfd`, `git checkout -f` in the
  worktree cannot reach it. It is on the pool mount, sharing the worktree's lifetime.
- **`TaskArtifacts` is the single path-derivation owner**: its constructor takes the
  derived `meta_root` (computed once from `worktree_base` + worktree name, supplied
  via config). Every one of the ~10 hand-built `<wt>/.task/...` sites
  (`git_ops.py:1831, 2952, 2730, 3030`; `harness.py:2004, 2030, 2241, 2252, 2277`;
  `workflow.py:3170`; `dashboard/.../orchestrator.py:180`) routes through
  `TaskArtifacts` (or a shared derivation fn it exposes) — no direct `.task` joins.
- **Compat window** (migration caution): all readers check **new-path-then-old-path**
  during the window; a final leaf, gated on a full green cycle, drops old-path support
  (new-path-only) and deletes the `.gitignore` writers.
- **Guard-layer deletion** (gated on the contamination integration gate proving
  staging is structurally impossible): remove the §Guard-inventory items 1, 3, 4, 5,
  6; keep `_assert_no_task_dir` (item 2) as a cheap tripwire through the migration,
  then drop it in the final leaf.
- **Interactive lanes**: `_iact-*` worktrees carry only `.task/interactive.json`
  (`git_ops.py:1831`), never `plan.json`; the relocation moves that stamp to
  `.task-meta/<iact-name>/interactive.json` and the interactive reaper
  (`reap_interactive_worktrees`, `git_ops.py:5431+`) reads the new path.

### Mechanism 3 — unify `acquire_warm_lane`'s 7 routes over `LaneLifecycle`

Rewrite the 7-route classifier so each route is expressed as a `LaneLifecycle`
transition, and every **fault** exit delegates to M1's `_abort_lane_acquisition`
(building on M1 δ, which subsumes the divergent teardown idioms). The route table
becomes: reuse / reuse-repair (2097) / create-once fresh / create-once reattach /
registered-disk-backstop reuse / reset-in-place reattach / recycle — each a named
`from→to` edge, no ad-hoc per-route teardown. **Hard prerequisite: M1 δ
(`_abort_lane_acquisition`) on main** (see §Pre-conditions).

## Resolved design decisions

1. **Two separate durable stores, both `worktree_base` children.**
   `.lane-state/<lane>.json` (lane-keyed pool state) and `.task-meta/<name>/…`
   (task-keyed execution artifacts) are distinct: a FREE lane has a lane-state record
   but no task metadata. Both are dot-prefixed, non-`_`-band directories → invisible
   to `git worktree` and to the positive-filter cleanup sweeps; W11 registers them as
   PROTECTED (owner tag) in M1's `PROTECTED_PREFIXES` as defence-in-depth and the
   orphan-reaper skips them explicitly.
2. **On-mount, not off-mount (parent), for both.** `quarantine_base` deliberately
   sits **off** the mount (in the parent) so the reaper never re-scans it; the
   lane-state and task-meta stores deliberately sit **on** the mount so they share
   the lanes' lifetime and vanish together with the mount (mount-down = lanes gone
   anyway). We reuse `quarantine_base`'s *structural* pattern (a dedicated sibling
   dir, one path-derivation owner) but not its location.
3. **`WarmLanePool` becomes a cache; `LaneLifecycle` is authoritative.** The
   in-memory map is rebuilt from durable records at startup and updated only through
   `transition(...)`. This is the "single source of truth" that ends the
   five-derivable-sources reconciliation.
4. **Illegal transitions escalate, never heal.** Assert + born-at-L2 escalation
   (sentinel role), record unchanged. Silent adoption/cleaning is exactly the
   cross-product the incidents came from.
5. **Quarantine-on-divergence, not adopt-on-doubt.** Startup recovery adopts only on
   an exact record↔git match; any divergence quarantines. This inverts the old
   restore-from-any-`plan.json` default (2098) that re-poisoned lanes every restart.
6. **`release_warm_lane`'s branch-retention semantics (tasks 1912/1914) are
   preserved.** LaneLifecycle records state transitions; it does **not** re-couple the
   branch-ref lifecycle those tasks decoupled. Branch deletion stays behind the
   existing on-main retention guard.
7. **New-then-old compat, old dropped in a gated final leaf.** No flag-day; readers
   tolerate both paths until a full green cycle proves the new path is authoritative,
   then old-path support + gitignore writers are deleted.
8. **Guard deletion is gated on the contamination gate, not co-committed with the
   relocation.** The relocation lands first (metadata simply moves); guards are
   deleted only after the two-way integration gate proves a hostile `git add -A`
   cannot stage anything.

## Pre-conditions for activating

- **M1 α `_prune_registrations`** — task **2185** (in-progress). W11's LaneLifecycle
  folds the pool-storage sentinel and relies on the single prune chokepoint. Wired.
- **M1 δ `_abort_lane_acquisition`** — **not yet filed** (only M1 α is in
  fused-memory as of 2026-07-06). Mechanism 3 (route unification) hard-depends on it.
- **M1 ε `PROTECTED_PREFIXES`** — **not yet filed**. Mechanism 1 registers the
  `.lane-state`/`.task-meta` bands into it. Soft-dep (W11 can self-protect if ε is
  late).
- Substrate in the §Substrate-reality table (all EXISTS rows) verified on main.

Because M1 δ/ε are not yet filed, W11's M1-δ/ε-dependent tasks are **anchored on
2185** and their precise δ/ε dependency edges are recorded as a to-wire item (see
§Open questions). They are placed **late** in the DAG so M1 has time to land.

## Cross-PRD relationship (G4)

| Other stream/PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| M1 `gitops-chokepoints` | W11 consumes | `_prune_registrations` (2185), `_abort_lane_acquisition` (δ), `PROTECTED_PREFIXES` (ε) | **M1** | α wired (2185); δ/ε to-wire when filed |
| M3 `dashboard-alignment` | W11 owns derivation; M3 documents the reader | `.task/` path derivation (`TaskArtifacts`) + dashboard `read_task_artifacts` | **W11** | M3 ζ = doc-only signpost (soft-coord; same file) |
| W10 `harness-supervision` | independent | `proc_supervision` restart plans | W10 | no seam (W11 restart capstone is a plain deterministic task) |
| W7 `verify-plan` | independent | `ephemeral_worktree` verify probes | W7 | no seam (verify probe worktrees out of W11 scope) |

No contested/reciprocal ownership: M3 explicitly cedes `.task/` derivation to W11;
M1 owns its three primitives outright.

## Contract (B+H)

### Lane state transition table

States: `SEED → REGISTERED → ASSIGNED → IN_USE → RELEASED`; terminal side-state
`QUARANTINED`.

| From | To | Trigger (writer) | Record fields set |
|---|---|---|---|
| — | `SEED` | pool seed (`_seed_warm_lane`, GitOps) | `seeded_from_sha`, `updated_at` |
| `SEED` | `REGISTERED` | `git worktree add` success (GitOps) | `branch`, `updated_at` |
| `REGISTERED`/`RELEASED` | `ASSIGNED` | `acquire_warm_lane` binds a task (GitOps) / `restore_assignment` (Harness) | `task_id`, `title`, `updated_at` |
| `ASSIGNED` | `IN_USE` | dispatch begins work (GitOps/workflow) | `updated_at` |
| `IN_USE`/`ASSIGNED` | `RELEASED` | `release_warm_lane` on terminal task (GitOps) | clears `task_id`/`title`, `updated_at` |
| any | `QUARANTINED` | recovery divergence (Harness) | `updated_at` (record preserved beside quarantine dir) |

**Invariants.**
- I1: exactly one durable record per live lane; the in-memory pool is a pure cache of
  the records.
- I2: any `(from,to)` not in the table → `IllegalLaneTransition` + born-at-L2
  escalation; record unchanged.
- I3: a record whose `branch`/`task_id` disagrees with git reality at startup →
  `QUARANTINED`, never `ASSIGNED`.
- I4: the `.pool-root` sentinel is written/read only via `LaneLifecycle`.

### `.task-meta` path-derivation contract

- `meta_root(worktree)` = `<worktree_base>/.task-meta/<worktree.name>` — computed once
  (from config) and handed to `TaskArtifacts`; no other module joins `.task`.
- Reads: new path, then legacy `<worktree>/.task` (compat window only). Writes: new
  path only.
- Post-migration invariant: `git -C <lane> add -A` stages zero task-metadata paths
  (they are not under the worktree root).

## Boundary-test sketch (B+H — the ω integration gate's signal)

| # | Scenario | Preconditions | Postconditions asserted |
|---|---|---|---|
| B1 | Writer→reader round-trip | acquire a lane for task N, then run startup recovery | recovery reads `.lane-state/<lane>.json`, adopts, rebinds N→lane; pool cache matches record |
| B2 | Crash between transitions → quarantine | record says `ASSIGNED:N` but `.git/worktrees/<lane>` admin entry removed (2097/2098 state) | recovery **quarantines** the lane (moved to `quarantine_base`, log line), does **not** re-pin N; next dispatch is clean |
| B3 | Illegal transition escalates | force `RELEASED → IN_USE` | `IllegalLaneTransition` raised; a born-at-L2 escalation is filed; record unchanged |
| B4 | Hostile `git add -A` (contamination) | agent runs `git add -A && git commit` in a lane whose task wrote plan.json/iterations | commit stages **zero** `.task-meta` paths; `git ls-tree` on the commit shows no metadata; main uncontaminable **with the scrub guards removed** |
| B5 | Survives cleaning | `git clean -xfd` + `git checkout -f` in a lane | `.lane-state` record and `.task-meta` artifacts both intact and readable |
| B6 | Dashboard reader | dashboard `read_task_artifacts` over a relocated lane | returns plan/phase/iteration data from `.task-meta` (new-then-old); no regression vs `.task/` layout |
| B7 | Fault-exit teardown (M3) | inject an exception mid-acquire after checkout | fault delegates to `_abort_lane_acquisition`; lane HEAD detached + pool FREE + no `already used by worktree` on re-acquire |

B1–B3 face the **Harness recovery reader** ↔ **GitOps writer** seam both ways; B4–B5
face the contamination seam; B6 the dashboard reader. **B1–B6 are ω's signal**
(mechanisms 1+2). **B7 is η's own signal** (mechanism 3, the M1-`_abort_lane_acquisition`
fault-teardown seam) — deliberately kept off ω's critical path so the single unmet
M1 δ prerequisite gates only η, not the whole integration tail.

## Decomposition plan

Labels are PRD-local; task IDs assigned at decompose. File-lock note: `git_ops.py`
and `harness.py` are the hot serialized files — the plan keeps each on a **linear
chain** (γ→η→θ on git_ops.py; δ on harness.py) to avoid narrow-lock starvation.

| # | Task | Files | Prereqs | Observable signal |
|---|---|---|---|---|
| α | `LaneLifecycle` module: `LaneState`, `LEGAL_TRANSITIONS`, atomic record I/O, `transition()` + illegal→escalate, quarantine helper | `orchestrator/src/orchestrator/lane_lifecycle.py` (new), test | — | Unit test: a legal `SEED→REGISTERED→ASSIGNED→IN_USE→RELEASED` sequence persists+reads back from `.lane-state/<lane>.json`; an illegal `RELEASED→IN_USE` raises `IllegalLaneTransition` and files a born-at-L2 escalation; record unchanged. Unlocks γ, δ. |
| β | `TaskArtifacts` single path owner over `.task-meta/<name>/` + config field + new-then-old read compat | `artifacts.py`, `config.py`, test | — | Unit test: `TaskArtifacts(worktree, meta_root)` writes `plan.json` under `.task-meta/<name>/`; a read returns it, and falls back to legacy `<worktree>/.task/plan.json` when only the old path exists. Unlocks ε1, ε2, δ. |
| γ | Route `acquire_warm_lane`/`release_warm_lane` through `LaneLifecycle` transitions; fold the `.pool-root` sentinel; move git_ops-side `.task` reads (disk-backstop `2952`, interactive stamp `1831`) to `.task-meta` | `git_ops.py`, `lane_lifecycle.py`, test | α, **2185**, M1 ε (to-wire) | Acquiring a lane writes `.lane-state/<lane>.json` `state=ASSIGNED,task_id=N`; release → `RELEASED`; `.pool-root` presence is written/read only via `LaneLifecycle` (grep shows no direct sentinel touch outside the module). Unlocks δ, η. |
| δ | Replace `_recover_crashed_tasks` cross-product with record-driven recovery (read record → verify git → adopt/quarantine); route `restore_assignment`/`note_assignment` through `transition()`; move harness-side `.task` reads to `.task-meta` | `harness.py`, `warm_lane_pool.py`, `lane_lifecycle.py`, test | α, β, γ, **2185** | Crash-recovery test (B2): a lane whose record says `ASSIGNED:N` but whose git registration is gone is **quarantined** (log + moved to `quarantine_base`), **not** re-pinned; a matching lane is adopted (B1). Unlocks ω. |
| ε1 | Hand the `.task-meta` base to workflow + agents + mcp_lifecycle via config; relocate agent-visible session/artifact paths | `workflow.py`, `agents/roles.py`, `orchestrator/.../mcp_lifecycle*`, config plumbing, test | β | A dispatched task's agent writes plan/iterations/session under `.task-meta/<name>/`; `git status` in the lane shows a clean tree (no `.task` entries). Unlocks ω. |
| ε2 | Dashboard `read_task_artifacts` new-then-old path (coordinate M3 ζ format-coupling doc) | `dashboard/src/dashboard/data/orchestrator.py`, test | β (soft-coord M3 ζ) | Dashboard test (B6): `read_task_artifacts` over a relocated lane returns plan/phase/iteration data from `.task-meta`; legacy `.task/` layout still parses during the window. Unlocks ω. |
| ω | **B+H integration gate** (mechanisms 1+2): the two-way boundary suite B1–B6 | `orchestrator/tests/test_lane_lifecycle_integration.py` (new) | γ, δ, ε1, ε2 | `pytest` of the B1–B6 boundary suite is green: writer↔reader round-trip, crash→quarantine, illegal→escalate, hostile-`git add -A` stages nothing, survives-clean, dashboard reader. Unlocks θ. |
| θ | Delete the guard layer (git_ops scrub call sites + `commit()` `:!.task` net; merge_gates ×4 `:!.task/` exclusions), keep `_assert_no_task_dir` as tripwire | `git_ops.py`, `merge_gates.py`, tests | ω | With guards deleted, the ω contamination test (B4) still passes (structural, not guard-defended); no `:!.task` pathspec remains except the retained tripwire; full `pytest` green. Unlocks ι. |
| ι | Final compat-close leaf: drop new-then-old fallback (new-path-only) + delete `.gitignore` writers + `_assert_no_task_dir` tripwire | `git_ops.py`, `artifacts.py`, tests | θ (gated on a full green cycle) | Grep shows zero legacy `<worktree>/.task` reads and zero `.task/.gitignore` writers remain; `pytest` green; a migrated lane has no `.task/` dir at all. |
| η | Unify `acquire_warm_lane`'s 7 routes over `LaneLifecycle` transitions; delegate every fault exit to M1 `_abort_lane_acquisition` (**independent leaf — off ω's critical path**) | `git_ops.py`, `lane_lifecycle.py`, test | γ, **M1 δ (to-wire; anchor 2185)** | Route table test: each of the 7 routes is a named `from→to` transition; fault-injection (B7) delegates to `_abort_lane_acquisition` (lane detached + FREE + no `already used by worktree` on re-acquire). Unlocks κ. |
| κ | Migration adopt + orchestrator restart deploy capstone (deferred-filer, ε2/2233 pattern): commit a one-shot adopt+restart script (adopt writes initial `.lane-state` records for live lanes), then file a `task_kind='deterministic'` self-restart-and-verify task depending on the full W11 batch | `scripts/deploy-w11-lane-lifecycle.sh` (new), deterministic task | ι, η (transitively whole batch) | The committed adopt+restart script exists+executable; a `task_kind='deterministic'` self-restart task is filed (get_task shows it) depending on the full W11 batch; on dispatch the orchestrator restarts and serves LaneLifecycle-backed acquire with `.task-meta` relocation live (fresh-PID verify). |

α, β are foundations; γ, δ, ε1, ε2 are intermediates feeding the ω gate; ω is the
B+H integration-gate leaf for mechanisms 1+2; θ, ι are the gated guard-deletion +
compat-close leaves; η is the independent mechanism-3 leaf (its single M1-δ
prerequisite gates only η); κ is the deploy capstone gated on the whole batch. The
`.task-meta` migration (β→ε1/ε2→ω→θ→ι) and the LaneLifecycle spine (α→γ→δ→ω) share
the ω gate; η hangs off γ in parallel.

## Out of scope

- M1's `_prune_registrations` chokepoint, `_abort_lane_acquisition` primitive, and
  `PROTECTED_PREFIXES` registry themselves (M1 owns; W11 consumes).
- W10 `proc_supervision` / restart-plan machinery (W11's capstone is a plain
  deterministic restart).
- W7 verify-probe `ephemeral_worktree` lifecycle.
- Spec-pool (`_spec-*`), `_merge-verify`/`_offline-deep`, `_solo-*` lane bands in the
  durable-record model (transition table is extensible to them later).
- Re-coupling the branch-ref lifecycle decoupled by tasks 1912/1914.

## Open questions (tactical — surfaced, not blocking)

1. **M1 δ/ε dependency wiring.** Only M1 α (2185) is filed as of 2026-07-06; δ
   (`_abort_lane_acquisition`) and ε (`PROTECTED_PREFIXES`) are not yet in
   fused-memory. **Safe default (taken):** wire γ/η on 2185 as the M1 anchor and place
   η late; when M1's δ/ε leaves are filed, add the precise `add_dependency` edges
   (η→δ, γ→ε). Decide/wire at first W11 dispatch or when M1 completes decompose.
2. **`meta_root` config surface.** Whether the `.task-meta` base is a new
   `config.task_meta_dir` field or derived from `worktree_dir` — either is coherent;
   derive-from-`worktree_dir` (no new config knob) is the suggested default. Decide in
   β.
3. **`lane_lifecycle.py` home vs `warm_lane_pool.py`.** New module (suggested, keeps
   the pool a thin cache) vs folding into `warm_lane_pool.py`. Decide in α.
4. **Escalation sentinel role string** for illegal transitions
   (`harness-lane-lifecycle` suggested) — keep it in the harness-internal allowlist so
   it is born-at-L2, not downgraded. Decide in α.
5. **M3 ζ soft-dep.** ε2 and M3 ζ both edit `dashboard/data/orchestrator.py`; wire ε2
   after M3 ζ if its id is available at decompose, else rely on the file lock. Decide
   at decompose dep-wiring.
6. **`_assert_no_task_dir` tripwire retirement point.** Kept through θ; dropped in ι.
   Confirm the full-green-cycle criterion (one clean merge-queue cycle post-θ) in ι.
