# Capability manifest — worktree-lane-lifecycle (W11)

Mechanizes G3 (assumed-substrate) + G6 (premise validity) per task. Evidence
verified on main 2026-07-06 (post-70f6a0ccbc). Each block: `task → [(capability →
evidence)]`. Any binding resolving to `declared-only | test-only | producer-absent |
producer-downstream | producer-extent-short | rejection-absent` **blocks queueing**
of that task until resolved.

Legend: `grep:<file>:<line>` = wired on main; `producer:task-N upstream` = delivered
by an upstream dep; `EXISTS` = definition present on main; `NEW` = this task creates
it (no prior substrate assumed).

---

## α — LaneLifecycle module (foundation)
- `LaneState` enum, `LEGAL_TRANSITIONS` table, `transition()` → **NEW** (this task creates them; no substrate assumed). PASS.
- atomic record write (`os.replace` tmp-rename) → stdlib `os`/`json`/`pathlib`. PASS.
- born-at-L2 escalation on illegal transition → **rejection-mechanism check**: the escalation client exists — `escalate_blocker` (escalation MCP) + harness escalation path used throughout `harness.py`; born-at-L2 sentinel-role convention documented in `plans/bug-hotspot-remediation-program-2026-07-06.md` D3 and CLAUDE.md ("Born-at-L2 escalations"). The signal AUTHORS an illegal transition and asserts the escalation FIRES — `rejection-check:RELEASED→IN_USE` is satisfied by the test raising `IllegalLaneTransition` + observing the filed escalation. PASS (the mechanism is built by this task; the test binds it).
- `quarantine_worktree` helper reuse → `grep:git_ops.py:6733` EXISTS. PASS.
- **G6:** no numeric/exactness premise. Branch-4 (rejection) bound above.

## β — TaskArtifacts single path owner over `.task-meta/`
- `TaskArtifacts` class to extend → `grep:artifacts.py:160-775` EXISTS (root `= worktree/'.task'` at :163-164). PASS.
- `worktree_base` to derive the `.task-meta` base from → `grep:git_ops.py:850` EXISTS (`project_root/config.worktree_dir`). PASS.
- config field for the base → `config.worktree_dir` `grep:config.py:797` EXISTS; new `.task-meta` derivation is **NEW** on top of it (Open Q2 — derive vs new knob). PASS.
- new-then-old read compat → **NEW** behaviour; legacy path `<worktree>/.task` EXISTS to fall back to. PASS.
- **G6:** the contamination premise ("`git add -A` cannot stage `.task-meta`") is a **structural** claim proved at ω-B4, not β. β only asserts write/read placement. PASS.

## γ — GitOps writer over LaneLifecycle + fold `.pool-root`
- `acquire_warm_lane`/`release_warm_lane` to route → `grep:git_ops.py:2502-3065` (acquire), `release_warm_lane` `grep:git_ops.py:3355` EXIST. PASS.
- `LaneLifecycle.transition()` → `producer:α upstream` (hard dep). PASS.
- `.pool-root` sentinel + `pool_storage_present/mark/_note/_bootstrap_ok` to fold → `grep:git_ops.py:175,930-1046` EXIST. PASS.
- M1 `_prune_registrations` chokepoint γ builds on → `producer:task-2185 upstream` (in-progress; wired). PASS.
- disk-backstop `.task/plan.json` read + interactive `interactive.json` stamp to relocate → `grep:git_ops.py:2952`, `grep:git_ops.py:1831` EXIST. PASS.
- M1 `PROTECTED_PREFIXES` band registration (defense-in-depth) → **producer-absent** (M1 ε unfiled). **Resolved by relaxation:** γ SELF-PROTECTS (dot-prefixed non-`_`-band dirs are invisible to the positive-filter sweeps and `git worktree`); the M1-ε registration is deferred defense-in-depth, NOT load-bearing → not a blocking capability for γ's signal. PASS (relaxed; see Open Q1).

## δ — Harness recovery over the durable record
- `_recover_crashed_tasks` heuristic tree to replace → `grep:harness.py:1948-2287` EXISTS. PASS.
- `restore_assignment`/`note_assignment` to route through `transition()` → `grep:warm_lane_pool.py:246,226` EXIST. PASS.
- durable `.lane-state` records to read → `producer:γ upstream` (γ writes them). PASS.
- `LaneLifecycle` + quarantine → `producer:α upstream`; `quarantine_worktree` `grep:git_ops.py:6733`. PASS.
- harness-side `.task` reads to relocate → `grep:harness.py:2004,2030,2241,2277` EXIST. PASS.
- **G6 (branch-3, end-to-end):** B2's quarantine-not-repin capability is delivered by δ itself + α (quarantine) + γ (records) — all UPSTREAM. No downstream owner. PASS.

## ε1 — agent/workflow/mcp_lifecycle path wiring
- `TaskArtifacts(worktree, meta_root)` new signature → `producer:β upstream`. PASS.
- `workflow.py` TaskArtifacts instantiation → `grep:workflow.py:1590` EXISTS. PASS.
- agent path admonitions / session paths → `grep:agents/roles.py:277,288,342` EXIST (targets to update). PASS.
- **G6:** clean-tree signal is structural (post-relocation). PASS.

## ε2 — dashboard read-path
- `read_task_artifacts(worktree_path)` to update → `grep:dashboard/src/dashboard/data/orchestrator.py:170-236` EXISTS (builds `worktree_path/.task` at :180). PASS.
- new-then-old path → `producer:β upstream` (base derivation). PASS.
- M3 ζ format-coupling doc → soft-coord (same file); NOT a blocking capability (doc-only signpost). PASS.

## ω — B+H integration gate (mechanisms 1+2) — the leaf carrying B1–B6
- writer↔reader round-trip (B1) → `producer:γ,δ upstream`. PASS.
- crash→quarantine (B2) → `producer:δ upstream`; `quarantine_base` `grep:git_ops.py:6687`. PASS.
- illegal→escalate (B3) → `producer:α upstream`. PASS.
- **hostile `git add -A` stages nothing (B4)** → **structural**: `.task-meta` is a `worktree_base` sibling, not under the worktree root (`producer:β,ε1 upstream`). This is the field-population/rejection twin: the claim is "staging PRODUCES zero `.task` entries" — bound by running `git add -A && git commit` and asserting `git ls-tree` shows none. `rejection-check:git-add-A-stages-task-meta` = the metadata is unreachable by the worktree's index by construction. PASS.
- survives-clean (B5) → structural (`producer:β upstream`). PASS.
- dashboard reader (B6) → `producer:ε2 upstream`. PASS.
- **G6:** B4 is the load-bearing premise; it is achievable because relocation is upstream of ω and the assertion is a *structural* absence, not a guard-observed one. PASS.

## θ — delete the guard layer
- scrub call sites / `commit()` `:!.task` net / merge_gates `:!.task/` ×4 to delete → `grep:git_ops.py:3044,4782,5956,1625,3860,3866`; `grep:merge_gates.py:1087,1117,1261,1282` EXIST. PASS.
- safety premise (deletion is safe) → `producer:ω upstream` (B4 proves contamination structural BEFORE guards are removed — DAG-direction correct: ω is upstream of θ). PASS.
- `_assert_no_task_dir` retained tripwire → `grep:git_ops.py:344,5861` EXISTS. PASS.

## ι — final compat-close leaf
- drop new-then-old fallback → `producer:θ upstream` (green cycle). PASS.
- `.gitignore` writers to delete → `grep:git_ops.py:328-341`, `grep:artifacts.py:194-200` EXIST. PASS.
- **G6:** "zero legacy `.task` reads remain" — bound by grep at task time. PASS.

## κ — deploy capstone (deferred-filer)
- committed adopt+restart script must EXIST+executable at submit_task → validated at `tools.py` submit path (deferred-filer pattern; precedent task 2233 = W5 π). PASS.
- deterministic self-restart per CLAUDE.md → orchestrator restart conventions (2064/2105 fixed+deployed), `task_kind='deterministic'`. PASS.
- **G6:** no numeric premise; deploy is deterministic. PASS.

## η — 7-route unification over LaneLifecycle (**FILED DEFERRED — held, NOT flipped**)
- `acquire_warm_lane` 7 routes to unify → `grep:git_ops.py:2502-3065` EXISTS. PASS.
- `LaneLifecycle.transition()` → `producer:γ upstream`. PASS.
- **M1 `_abort_lane_acquisition` (every fault exit delegates to it)** → **producer-absent**:
  `grep _abort_lane_acquisition orchestrator/src` = ZERO matches; M1 δ is not on main
  and is **not a filed task** (only M1 α=2185 exists). **This is a hard FAIL binding.**
  **Resolution (gate-compliant):** η is filed in `planning_mode` but **NOT flipped to
  pending** — held `deferred` so it cannot dispatch against absent substrate. When M1's
  δ leaf is filed: `add_dependency(η → M1-δ)`, then flip η. This is the gate's "queue
  the prerequisite upstream + wire the dep" applied by holding η until the prerequisite
  is fileable. **BLOCKS η's flip; does NOT block the mechanism-1+2 batch.**

---

## Gate summary
- All mechanism-1+2 tasks (α,β,γ,δ,ε1,ε2,ω,θ,ι,κ): **all bindings PASS** → cleared to flip.
- η: one **producer-absent** binding (`_abort_lane_acquisition`, M1 δ) → **held deferred**; cleared to flip only after M1 δ is filed + the η→δ edge wired.
- One relaxed binding (γ's M1-ε `PROTECTED_PREFIXES` registration) → non-load-bearing (γ self-protects); recorded as deferred defense-in-depth.
