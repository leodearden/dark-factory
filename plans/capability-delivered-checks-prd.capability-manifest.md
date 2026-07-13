# Capability Manifest — capability-delivered-checks-prd

Mechanizes G3 (substrate exists/**wired**) + G6 (premise valid) per task. One
block per task; each capability bound to on-main evidence. Any **FAIL**
binding blocks queueing.

**Domain flags:** tooling/infra domain — no grammar/DSL → grammar-fixture
checks **N/A**; no numeric accuracy bounds → numeric-floor checks **N/A**.
Live checks: **capability→producer (wired)**, **DAG-direction**,
**field-population**, **rejection-mechanism**.

All `file:line` evidence re-confirmed on current main during the authoring
session (2026-07-13). Machine-readable twin (this PRD's own exemplar):
`plans/capability-delivered-checks-prd.capability-manifest.yaml` — note that
for THIS batch the sidecar is exemplar/fixture only: the γ stamper does not
exist yet, so ids are hand-stamped at decompose and no task carries
`metadata.delivered_checks` (stated in the PRD to avoid a G6 fiction).

---

## α — Shared sidecar schema + delivered-check models  *(intermediate → β, γ, δ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `register_metadata_submodel` exists to register `delivered_checks` | capability→producer (wired) | **confirmed** — `shared/src/shared/task_metadata.py:305`; `BeforeDone`/`Milestone`/`ExternalDep` precedents in the same module | PASS |
| pydantic v2 + pyyaml available in `shared/` | capability→producer (wired) | **confirmed** — `shared/pyproject.toml:14` (pydantic≥2.7), `:20` (pyyaml≥6.0) | PASS |
| An exemplar sidecar exists to validate against | capability→producer | **committed this session** beside the PRD (`…capability-manifest.yaml`) — the CI fixture α's loader parses | PASS |
| Malformed docs rejected with structured errors naming the entry | rejection-mechanism | producer = α — α authors the validator + malformed fixtures and observes each §Contract rule fire; rejection is α's deliverable | PASS (built+bound by α) |

## β — /prd skill emits the YAML sidecar; exemplar committed  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| decompose-mode Step 2.5 exists to extend | capability→producer (wired) | **confirmed** — `skills/prd/references/decompose-mode.md:38-42` (Step 2.5 builds the .md manifest today; repo copy identical to `~/.claude` copy) | PASS |
| gates.md manifest section exists to extend | capability→producer (wired) | **confirmed** — `skills/prd/references/gates.md:162-184` (Capability Manifest section) | PASS |
| Exemplar sidecar validates via α's loader | DAG-direction | α **upstream** of β (β deps α); fixture committed this session | PASS |

## γ — commit_planning stamps sidecar + copies delivered_checks  *(intermediate → ζ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `commit_planning` exists and already batch-reads task records | capability→producer (wired) | **confirmed** — `fused-memory/src/fused_memory/server/tools.py:3289` (def), `:3366` (asyncio.gather get_task over the batch — the stamping hook point) | PASS |
| `metadata.prd_path` + `prd_task_label` populated on decompose batches | field-population | **confirmed** — decompose-mode.md:66-70 metadata block; every recent batch carries them (spot-checked 2338, 2043 descriptions reference PRD labels); THIS batch sets both on all six tasks | PASS |
| Server may touch project-repo files (sidecar write-back) | capability→producer (wired) | **precedent confirmed** — `before_done.script` exists+executable validation at submit_task (`tools.py:3014-3047`, deterministic_task_guard) does project_root filesystem access today | PASS |
| Metadata write is merge-safe (no replace-not-merge clobber) | rejection-mechanism | **hazard known & fixed substrate** — tasks 1827/1828 landed (survey addendum); γ's tests must assert sibling metadata keys survive the stamp | PASS (constraint noted; test bound to γ) |
| α's models importable from fused-memory | DAG-direction | α **upstream** of γ (γ deps α); `shared` already a fused-memory dep (`fused-memory/pyproject.toml` workspace) | PASS |

## δ — Scheduler delivered-check gate + runner + per-SHA cache  *(intermediate → ε, ζ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `_deps_satisfied` pure-predicate cache-param pattern to mirror | capability→producer (wired) | **confirmed** — `orchestrator/src/orchestrator/scheduler.py:2930-3019` (`external_status_cache` arm; "side effects MUST NOT live here" docstring `:2940`) | PASS |
| Per-tick sweep + streak counters + fail-safe precedent | capability→producer (wired) | **confirmed** — external-dep sweep `scheduler.py:2340-2489` (`_streak_external_unresolved`, resolver-degraded no-bump branch) | PASS |
| Dep task records available at the gate (`tasks_by_id`) | capability→producer (wired) | **confirmed** — `_deps_satisfied(tasks_by_id=…)` used by the intra-train allowance `:2970-2979`; dispatch call sites pass the full snapshot | PASS |
| `git grep <pattern> main` evaluates against the committed main tree | capability→producer | git substrate; merge worker already shells git against project_root (`git_ops.py`); no checkout-cleanliness dependence | PASS |
| α's `DeliveredCheck` model importable | DAG-direction | α **upstream** of δ (δ deps α) | PASS |

## ε — Grace-streak escalation + config knobs + ops docs  *(intermediate → ζ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Born-at-L2: severity + sentinel-role contract | capability→producer (wired) | **confirmed** — `escalation/src/escalation/models.py:41` (`BORN_AT_L2_SEVERITIES`), `escalation/server.py:303` (`_is_harness_sentinel_role`, `orchestrator-*` passes), `:333-334` (level=2 stamp) | PASS |
| Callback shape to mirror (`on_external_dep_block`) | capability→producer (wired) | **confirmed** — `orchestrator/harness.py:788` (wiring), `:4244+` (`_block_and_escalate_external_dep`: blocked + dedupe + file) | PASS |
| `category` accepts a new `dependency_capability` string | capability→producer | **confirmed** — free-form `str` field (`escalation/models.py:55`); watcher urgency keys off severity (`watcher.py:110`) | PASS |
| Green-tier hot-reload for `delivered_checks.*` knobs | capability→producer (wired) | **confirmed** — config hot-reload allowlist mechanism landed (`plans/config-hot-reload-prd.md`, `reload_config` tool live); ε adds the keys | PASS |
| A pending L2 names the exact failed check | field-population | producer = ε — summary/detail contract fixed in PRD §Resolved 6 (`DEP_CAPABILITY_NOT_DELIVERED`, check name, pattern, dep id, main SHA) | PASS |
| δ's gate/streak substrate upstream | DAG-direction | δ **upstream** of ε (ε deps δ) | PASS |

## ζ — End-to-end integration gate  *(leaf — G2 top signal; C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Every boundary-sketch capability (stamp, gate, escalate) | DAG-direction | all produced by γ/δ/ε, **upstream** of ζ (ζ deps γ, δ, ε; α transitively) | PASS |
| `metadata.delivered_checks` populated non-sentinel by the stamper | field-population | producer = γ (upstream) copies mechanical checks into producer-task metadata; ζ asserts via `get_task` (the product's own read path) | PASS |
| Dep-done-with-failing-check → withhold → named L2 → land → dispatch, end-to-end | end-to-end capability | trace: stamp ← γ; withhold+cache ← δ; grace+L2+blocked ← ε; re-pend + dispatch ← existing scheduler substrate (manual re-pend contract, CLAUDE.md external-dep precedent). No inversion — every leg upstream | PASS |
| Integration-test rig (planning batch + scheduler tick driving) | capability→producer | **exists** — `orchestrator/tests/` scheduler tick-phase + cross-project dispatch integration tests (`test_cross_project_dispatch_integration.py`, `test_scheduler_tick_phases.py`) provide the rig patterns | PASS |

---

## Manifest verdict

**All bindings PASS (α, β, γ, δ, ε, ζ).** No novel substrate — every capability
is wired on main today or produced upstream in-batch. No cross-PRD holds:
unlike ν/ξ-style pending deps, nothing here waits on another PRD's unlanded
work. The whole batch files, wires intra-batch deps (β←α, γ←α, δ←α, ε←δ,
ζ←γ+δ+ε), commits this manifest + the YAML exemplar, and flips to **pending**
in one `commit_planning` call.
