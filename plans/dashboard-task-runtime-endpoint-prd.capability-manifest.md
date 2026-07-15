# Capability Manifest — dashboard-task-runtime-endpoint-prd

Mechanizes G3 (substrate exists/**wired**) + G6 (premise valid) per task. One
block per task; each capability bound to on-main evidence. Any **FAIL** binding
blocks queueing.

**Domain flags:** tooling/infra domain — no grammar/DSL → grammar-fixture
checks **N/A**; no numeric accuracy bounds → numeric-floor checks **N/A**.
Live checks: **capability→producer (wired)**, **DAG-direction**,
**field-population**, **rejection-mechanism** (the hand-reader deletion, B7).

All `file:line` evidence re-confirmed on current main during the decompose
re-walk (2026-07-15, `git rev-parse HEAD` = a7f9d75585). This PRD introduces
**no novel substrate** — it is pure infrastructure wiring of existing format
owners (`LaneLifecycle`, `WarmLanePool`, `TaskArtifacts`) behind a new MCP tool.

**Delivered-check gate is LIVE** (tasks 2578 γ-stamper + 2580 δ-gate `done` and
merged on main). `commit_planning` auto-stamps the YAML sidecar and copies each
producer's **mechanical** `delivered_check`s into `metadata.delivered_checks`;
the scheduler then gates a consumer's dispatch on the producer's check passing
on main. Mechanical checks are therefore bound **conservatively** — only robust
`def <name>` presence greps (α/β/γ), which pass byte-for-byte once the producer
lands and cannot false-negative a consumer into a wedge. Deletions (B7),
UI-render assertions, and the boundary suite itself are `kind: manual` in the
sidecar (recorded, excluded from the automated gate — covered by ε's B-suite).

Machine-readable twin: `plans/dashboard-task-runtime-endpoint-prd.capability-manifest.yaml`.

---

## α — `Harness.task_runtime_snapshot()` accessor  *(intermediate → β; foundation, roped into ε)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Harness reaches `WarmLanePool` (task→lane) | capability→producer (wired) | **confirmed** — `warm_lane_pool.py:238` `assignments_snapshot`; reachable via `harness.git_ops.warm_lane_pool` (`harness.py:2175`) | PASS |
| Harness reaches `LaneLifecycle` durable task_id-keyed `.lane-state` (task→lane + lane_state) | capability→producer (wired) | **confirmed** — `lane_lifecycle.py:253` `read`, `:192` `task_id` on `LaneRecord`; reachable via `harness.git_ops._lane_lifecycle` (`harness.py:2177`) | PASS |
| `TaskArtifacts.{read_iteration_log,read_reviews,read_plan,meta_root_for}` for loops/attempts/phase | capability→producer (wired) | **confirmed** — `artifacts.py:486,536,425,204`; imported by harness (`harness.py:29`); `TaskArtifacts(worktree, meta_root)` ctor `:189` | PASS |
| Layout-aware via harness's own config (pooled `.lane-state` vs non-pooled `.worktrees/<id>`) | capability→producer (wired) | **confirmed** — `harness.git_ops.pool_in_use()`/`worktree_base` (`harness.py:2141,2157`) already branch pooled vs per-task | PASS |
| loops/attempts are **real counts, never sentinel**; honest-empty (0) distinguished from read-error | field-population | producer = α — reads `iterations.jsonl` line count + review verdicts via `TaskArtifacts` (real values). Design note carried into α's description: a per-task artifact **read failure** must not be silently coerced to `0` (honors `structured-facts-at-failure`); honest-empty→0 is B4 | PASS (built+bound by α) |

## β — `get_task_runtime_state` MCP tool + declared return schema  *(intermediate → γ, ε; foundation, roped into ε)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Escalation server runs in-process with harness; `@mcp.tool` closures reach `harness` | capability→producer (wired) | **confirmed** — `create_server(harness=…)` booted at `harness.py:7015`; tools reach `harness.git_ops` throughout `server.py` (e.g. `:121,:1574`) exactly as `merge_request`/`get_merge_halt_status` do | PASS |
| α's accessor upstream of the tool | DAG-direction | α **upstream** of β (β deps α) | PASS |
| Return-shape contract is a **declared machine-checked schema**, not prose (single `shared/` model both sides import) | capability→producer (wired) | `shared/` is already a dashboard **and** escalation dependency; the model is authored by β. Design note carried into β's description: declare ONE `shared/` pydantic/TypedDict model imported by both the tool and the dashboard join — no structurally-mirrored second copy (INV-1 `contracts-machine-checked`, INV-5 `no-lockstep-duplication`; resolves Open-Q2) | PASS (built+bound by β) |

## γ — Dashboard consumes MCP runtime; delete the hand reader  *(intermediate → δ, ζ, ε; also carries an API-response signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Per-project MCP fan-out substrate to clone (`mcp_tool_call`, session cache, per-project offline marker) | capability→producer (wired) | **confirmed** — `data/memory.py:176` `mcp_tool_call`; `data/merge_halt.py` clones it (fan-out over `config.escalation_urls`, per-call timeout, `{offline: true}` shape) | PASS |
| `active_tasks._shape_one_project` is the join site sourcing loops/attempts/started | capability→producer (wired) | **confirmed** — `active_tasks.py:211` `_shape_one_project`, `:253` calls `_scan_worktrees` (the call γ replaces with the fan-out map) | PASS |
| β's tool upstream of dashboard consumption | DAG-direction | β **upstream** of γ (γ deps β) | PASS |
| Fields sourced from the map are **non-sentinel** (real loops/attempts/started/lane/phase/lane_state) | field-population | producer = β/α (upstream) return real values per the contract's return-shape invariants; γ joins them | PASS |
| Hand reader (`_scan_worktrees`/`_extract_task_id`/`read_task_artifacts` + `discover_orchestrators` `worktrees` dict) **deleted** — no disk-read fallback survives | rejection-mechanism (B7) | producer = γ — deletion is γ's deliverable; asserted by ε's B7 grep-for-absence. Sidecar binds this `kind: manual` (an `expect: absent` grep would false-positive on a docstring/comment mentioning the retired name — e.g. ζ's retirement note) | PASS (built+bound by γ, verified by ε) |

## δ — OrchTab Lane / Phase / State columns + offline `—` rendering  *(leaf — user-observable; the reported-bug fix)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| OrchTab table exists to extend with columns | capability→producer (wired) | **confirmed** — `dashboard/src/dashboard/frontend/…/tabs.jsx` renders the Orchestrators table (Loops/Tries/Age today) | PASS |
| lane/phase/lane_state + real loops/attempts/started available on the row | capability→producer | producer = γ (**upstream**, δ deps γ) populates the row from the runtime map | PASS |
| Offline project row renders `—`, never `0` | field-population | producer = γ carries the per-project offline marker (upstream); δ renders `—`. Sidecar binds `kind: manual` — a JSX render assertion is not robustly greppable and is exercised by ε's B6 | PASS |

## ε — B+H integration gate: two-way boundary suite B1–B7  *(leaf — G2 top signal; C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Producer legs (accessor α, tool β) upstream | DAG-direction | α, β, γ, δ all **upstream** of ε (ε deps α,β,γ,δ) — no inversion | PASS |
| B1/B2/B4 loop/attempt counts are round-trip identities (fixture N lines → `loops=N`) | field-population | `TaskArtifacts.read_iteration_log` counts `iterations.jsonl` lines; the identity is "count == fixture line count" — no accuracy bound, achievable by construction | PASS |
| B7 hand-reader-gone (grep for absence) | rejection-mechanism | the retired symbols are produced-absent by γ (**upstream**); ε's B7 is the observing check | PASS |
| Integration-test rig (fixture harness + dashboard join/offline mocks) | capability→producer (wired) | **confirmed** — `orchestrator/tests/` scheduler/cross-project integration patterns + dashboard data-layer tests provide the rig shape | PASS |

## ζ — Retire the FORMAT-COUPLING reader marker (companion correction)  *(leaf — docs; `complexity=simple`)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The `orchestrator.py` FORMAT-COUPLING / "FUTURE SINGLE OWNER / W11 greps this marker" doc block exists to retire | capability→producer (wired) | **confirmed** — `dashboard/src/dashboard/data/orchestrator.py` module docstring names itself the hand-maintained twin + the W11-grep marker (`:6-79`) | PASS |
| The deletion this note documents has landed | DAG-direction | producer = γ (**upstream**, ζ deps γ) performs the hand-reader deletion; ζ only retires the now-stale pointer | PASS |
| Both prior PRDs (`worktree-lane-lifecycle-prd.md`, `dashboard-alignment-prd.md`) exist to annotate | capability→producer (wired) | **confirmed** — both committed under `plans/`; each left the seam deferred (PRD §Cross-PRD relationship) | PASS |

---

## Manifest verdict

**All bindings PASS (α, β, γ, δ, ε, ζ).** No novel substrate — every capability
is wired on main today or produced upstream in-batch. No cross-PRD holds:
nothing waits on another PRD's unlanded work (the two referenced PRDs *declined*
this seam; this PRD owns it). The whole batch files, wires intra-batch deps
(β←α, γ←β, δ←γ, ζ←γ, ε←α,β,γ,δ), auto-stamps + commits this manifest and the
YAML sidecar, and flips to **pending** in one `commit_planning` call.
