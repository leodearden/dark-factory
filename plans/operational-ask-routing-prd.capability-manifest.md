# Capability manifest — operational-ask-routing-prd

Mechanizes G3 + G6 per leaf: each task's asserted capabilities bound to
evidence (existing-substrate grep, or `producer:task-<label> upstream` for
batch-built capabilities). All bindings PASS — the batch is self-contained and
every asserted capability is either existing `main` substrate or produced by a
strictly-upstream task in the DAG. Machine-readable twin:
`operational-ask-routing-prd.capability-manifest.yaml`.

DAG: α → β → {γ, δ}; α → ε; {β, γ, δ, ε} → ζ.

## α — `operational_mode` schema field + validation

- **operational_mode-recognized-and-validated** → capability→producer (wired):
  mirrors `execution_class`'s existing handling — `grep` shows `execution_class`
  is a recognized field validated at the write boundary
  (`shared/src/shared/task_metadata.py`, `execution_class_guard.py`). α adds the
  parallel `operational_mode` field. Substrate pattern exists. **PASS.**
- **invalid-value-rejected** (G6 branch 4, rejection): the `ValidationError`
  mechanism exists (`execution_class_error` rejects an invalid class); α binds a
  parallel rejection for `operational_mode ∉ {gate,llm}`, observed to fire in
  α's own tests. **PASS** (rejection-mechanism, built+bound by α).

## β — submit-boundary `inject_operational_routing`

- **inject-runs-before-planning_mode-branch** (G6 branch 3, end-to-end):
  `tools.py:submit_task` runs its inject chain (`inject_task_kind`,
  `inject_execution_class`) before forwarding to `interceptor.submit_task`, whose
  `planning_mode` branch is downstream of the injected metadata — so a boundary
  inject reaches the planning_mode path. Verified on `main`. **PASS.**
- **pure-gate-stamp-reused** → capability→producer (wired):
  `_inject_deterministic_pure_gate` exists
  (`task_interceptor.py`) and is reused by β (no lock-step re-impl, INV-5).
  **PASS.**
- **operational_mode-field** → `producer:task-α` upstream (α delivers the
  validated field β reads). **PASS** (DAG-direction: α is upstream of β).

## γ — `operational_llm` distinct human-gate

- **pure-gate-born-at-L2-escalation-exists** → capability→producer (wired): the
  `DeterministicRunner` files a born-at-L2 escalation for an
  `always_escalates=true` pure-gate with author-influenceable `summary`/`detail`/
  `category` (`orchestrator/src/orchestrator/deterministic_runner.py`). γ routes
  the `llm` marker into a distinguishable reason. Substrate exists. **PASS.**
- **llm-marker** → `producer:task-β` upstream (β's inject stamps the `llm`-gate
  marker γ consumes). **PASS** (DAG-direction: β upstream of γ).
- **not-dispatched-to-architect** (G6 branch 3): a `task_kind=deterministic`
  task is routed to the `DeterministicRunner`, never the architect — the
  deterministic routing exists on `main`; β sets `task_kind=deterministic` for
  the `llm` gate. **PASS.**

## δ — demote curator execution_class axis to legacy fallback

- **execution_class-axis-and-substring-entries-exist** → capability→producer
  (wired): `match_candidate`'s execution_class axis + substring loop both exist
  (`operational_ask_registry.py`). δ narrows the axis to untagged-only; substring
  fallback retained. Substrate exists. **PASS.**
- **boundary-owns-tagged-routing** → `producer:task-β` upstream (β must own
  tagged routing before the curator axis can be safely demoted). **PASS**
  (DAG-direction: β upstream of δ).

## ε — recon source-completion brief

- **recon-stage-render-and-prompts-exist** → capability→producer (wired):
  `recon_self_model.render_*` + Stage 1/2 prompt modules exist and are rendered
  into the stage prompts (`reconciliation/prompts/`). ε edits the guidance text.
  Substrate exists. **PASS.**
- **stage-1-2-hold-memory-mutation-tools** (G6 branch 3): Stage 1/2 have
  `add_memory`/`delete_memory`/`merge_entities` (`cli_stage_runner.py`
  `DISALLOW_MEMORY_WRITES` scopes only Stage 3) — the inline-merge instruction is
  executable by the stage it targets. Verified on `main`. **PASS.**
- **operational_mode-declared-on-recon-submits** → `producer:task-α` upstream
  (the field ε instructs stages to declare is delivered by α). **PASS.**

## ζ — boundary integration gate (B+H)

- **boundary-matrix-green** → DAG-direction: every leg of the boundary-test
  matrix is produced strictly upstream (α field, β boundary inject, γ llm-gate,
  δ curator demotion, ε recon brief). ζ is the leaf whose signal is the green
  matrix. **PASS.**
