# Capability manifest — task-metadata-schema (W3)

Mechanizes G3 + G6 per leaf. PRD: `plans/task-metadata-schema-prd.md`. Verified 2026-07-06.
Leaves: **ζ, θ2**. Deterministic deploy **θ1** is an intermediate but asserts substrate, so
its bindings are included. Intermediates α–ε are the producers the leaves bind to.

Evidence vocabulary: `producer:task-<label> upstream` (deliverable covers the specific
extent, and the producer is upstream in the DAG) · `grep:<file>` (wired/exists on main) ·
`substrate:<doc>` (documented existing capability). Any FAIL value
(`declared-only`/`test-only`/`producer-absent`/`producer-extent-short`/`producer-downstream`/
`rejection-absent`) blocks the batch. **No FAIL bindings below.**

---

## Leaf ζ — two-way boundary-test suite (CI-green)

| Capability the signal asserts | Check | Evidence | Verdict |
|---|---|---|---|
| `shared.task_metadata.parse_metadata` (one parser, one policy) | capability→producer | `producer:task-α upstream` (α delivers `parse_metadata`) | PASS |
| `TaskMetadata` + `BeforeDone/DoneProvenance/MemoryHints/ExternalDep/RetryLedger` sub-models | capability→producer | `producer:task-α upstream` | PASS |
| Shared `DoneProvenance.kind` Literal enum (the ONE valid-kinds decl, I2) | capability→producer | `producer:task-α upstream` (defined), `producer:task-γ upstream` (backend `_VALID_PROVENANCE_KINDS` deleted → imports it) | PASS |
| Backend `add_task`/`update_task` validate via `parse_metadata` (post-merge) | capability→producer | `producer:task-β upstream` | PASS |
| Orchestrator constructs `DoneProvenance` from the shared model | capability→producer | `producer:task-δ upstream` | PASS |
| Workflow writes typed `RetryLedger`; persist-fail escalates (row 4) | capability→producer | `producer:task-ε upstream` | PASS |
| `task_metadata.enforce` flag toggles warn/reject (rows 5-6) | capability→producer | `producer:task-β upstream` | PASS |
| pydantic v2 `extra='allow'` retains + re-serialises unknown keys (I1 round-trip) | substrate exists | `substrate:pydantic>=2.7` in all three pyprojects; `extra='allow'` populates `model_extra` and re-emits via `model_dump()` (pydantic v2 documented) | PASS |
| **Rejection fires** on `kind='bogus'` / malformed write in enforce-mode (rows 3,5,6 — G6 branch-4) | rejection-mechanism | mechanism = `Literal` + `@model_validator` (α) raising `ValidationError` surfaced at the backend write boundary (β); both **upstream** of ζ; **ζ is itself the binding test that observes the diagnostic fire** | PASS |
| DAG-direction (anti-inversion) | dag-direction | all producers α,β,γ,δ,ε are **upstream** of ζ (ζ deps each) | PASS |
| Field-population (DoneProvenance/RetryLedger fields sampled) | field-population | δ/ε write **real non-sentinel** values (constructed kinds, live counters), not placeholders | PASS |

*Note on the rejection bindings:* the rejection mechanism is not pre-existing substrate — it
is produced by α (`Literal`+validator) and β (backend enforcement), both upstream. Per the
manifest's rejection rule this is the analog of "a named producer task is upstream"; ζ is the
test that authors the malformed input and observes the `ValidationError`. Legitimate — not
`rejection-absent`.

## Leaf θ2 — enforce-flip capstone (malformed write rejected via MCP)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `task_metadata.enforce` config key exists to flip | capability→producer | `producer:task-β upstream` (via ζ→β) | PASS |
| `task_metadata.schema_warning` census line greppable in journal | capability→producer | `producer:task-β upstream` | PASS |
| Deterministic **pure-gate** (absent `before_done` + `always_escalates=true`) | substrate exists | `substrate:CLAUDE.md` "Deterministic task kind" field-combo preset (absent/true = pure gate) | PASS |
| Born-at-L2 escalation from a pure gate | substrate exists | `substrate:CLAUDE.md` born-at-L2, `agent_role='orchestrator-deterministic'` | PASS |
| Out-of-cgroup `systemctl --user restart fused-memory.service` | substrate exists | `substrate:resolved decision #6` (program doc); NOT `restart-fused-memory.sh --drain` (hung, task 2090) | PASS |
| **Rejection fires** post-flip (signal is a negative assertion — G6 branch-4) | rejection-mechanism | mechanism produced by α+β+γ, all transitively upstream (θ2 deps ζ deps α,β,γ); θ2's live signal observes it | PASS |
| DAG-direction | dag-direction | ζ, θ1 upstream of θ2 | PASS |

## Intermediate deploy θ1 — warn-mode go-live (substrate bindings)

| Capability | Check | Evidence | Verdict |
|---|---|---|---|
| `scripts/restart-fused-memory.sh` exists + executable, no-`--drain` path = `systemctl --user restart fused-memory` + health-wait | substrate exists | `grep:scripts/restart-fused-memory.sh` (`-rwxrwxr-x`; body at lines 52-66 does `systemctl --user restart "$SERVICE"` + `curl localhost:8002/health`) | PASS |
| Deterministic `before_done` runner path (blocking, cross-unit, fresh-MainPID verify) | substrate exists | `substrate:CLAUDE.md` "Blocking vs detached self-kill" (`target_unit != own unit` → blocking + fresh MainPID) | PASS |
| `before_done.script` validated exists+executable+`timeout_secs` at `submit_task` | substrate exists | `grep:deterministic_task_guard.py:157` `_validate_before_done` (path-containment, `os.X_OK`, positive-int `timeout_secs`) | PASS |
| Warn-mode `task_metadata.schema_warning` emitted post-restart | capability→producer | `producer:task-β upstream` | PASS |

---

## Substrate re-verification log (G3, 2026-07-06)

- `pydantic>=2.7` — direct dep in `shared/pyproject.toml:14`, `fused-memory/pyproject.toml:17`,
  `orchestrator/pyproject.toml:17`. **No dataclass fallback needed** (retires the brief's G3 open item).
- `dark-factory-shared = { workspace = true }` — `fused-memory/pyproject.toml:27`,
  `orchestrator/pyproject.toml:26`; orchestrator already imports `shared.usage_gate`,
  `shared.locking`, `shared.cost_store`, `shared.safe_io`, `shared.config_models`. No new coupling.
- `shared/src/shared/config_models.py` — existing shared-pydantic `BaseModel` precedent.
- 8 parser sites re-verified by symbol (lines drift, symbols stable): `_parse_metadata`
  (interceptor:977), `_extract_metadata_dict` (interceptor:1428), `_merged_audit_metadata`
  (interceptor:3845), `deterministic_task_guard._parse_metadata` (:48),
  `lock_charter_guard.extract_files` (:147/179), `tools.py` `json.loads` (:2343/2359),
  `_row_to_task` (backend:274), `_merge_metadata` (backend:1354).
- `_VALID_PROVENANCE_KINDS` (interceptor:3548-3553) currently lists all four kinds incl.
  `deterministic-deploy-scheduled` — confirms the 1902/1976/1982 incident patch landed
  post-hoc (G6 premise real; the recurrence is the point).
- Scheduler untyped read `(task.get('metadata') or {}).get('external_deps') or []` at
  `scheduler.py:1996`. Workflow counters at `workflow.py` 1030/2323/3305/3329/3424/3453,
  `_merge_fresh_metadata:3233`, `_normalize_cause_hint:367`, `_compute_merge_outcome_signature:394`.
