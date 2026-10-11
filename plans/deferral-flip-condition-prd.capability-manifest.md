# Capability manifest — deferral records ("every `deferred` task says how it ends")

Binds each task's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `deferral-flip-condition-prd.capability-manifest.yaml`.
PRD: `plans/deferral-flip-condition-prd.md`.
As of main `fc55c9c7c8`, 2026-10-08.

**Verdict summary: 63 bindings, all PASS; 25 mechanical checks, zero polarity findings at `main`.**

The manifest seat's first pass found two `producer-downstream` FAILs with one root. The
"expiry lapse left unflipped for more than two sweep intervals" rule, used by α's
`needs_human` and ε's census `--check`, needed a sweep interval that only γ's config key
owned, and γ is downstream of α and parallel to ε. The lead resolved it at decompose (PRD §12
item 3): α owns `DEFAULT_SWEEP_INTERVAL_SECS` and `LAPSE_OVERDUE_AFTER_SECS` in
`shared/src/shared/task_deferral.py`, and γ's config default reads the former. Both rows are
now PASS, and α's mechanical check on `LAPSE_OVERDUE_AFTER_SECS` gates ε and δ.

The decompose re-walk also moved β's, γ's and δ's post-restart live checks into ζ (PRD §12
item 1), because no task in the batch delivers a service restart. γ and δ are therefore
intermediates that unlock ζ. Every other binding holds. None is declared-only, test-only,
producer-absent or producer-extent-short, and the DAG (α→β; α,β→γ; α,β→ε; α,β,ε→δ; γ,δ,ε→ζ)
has no inversion.

## Substrate findings that shaped the bindings

| Capability | Status at decompose (re-verified at `fc55c9c7c8`) | Consequence |
|---|---|---|
| No deferral symbols on main | **CONFIRMED** — `DeferralKind`, `CALLER_KINDS`, `ProcessIdentity`, `deferral_required`, `deferral_gate`, `deferral_sweep`, `deferral_census`, `migrate_deferrals`, `expected_stamped_at` and `held_by_session` have zero matches in any `.py`, `.jsx`, `.js` or `.yaml` on main | every mechanical check below fails today, as required |
| Status choke point | **CONFIRMED** — `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._apply_status_transition` | β wires one gate call there |
| Atomic status + metadata write | **CONFIRMED** — `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend.set_status_and_stamp_audit` takes `audit_fields`; it has no key-removal semantics | β adds the narrow removal (PRD open question 1) |
| Write floor to follow (not clone) | **CONFIRMED** — `sqlite_task_backend.py::_assert_done_provenance_passthrough`, `fused-memory/src/fused_memory/backends/task_backend_errors.py::DoneProvenanceWriteAuthorityError` | β adds `DeferralWriteAuthorityError` beside it |
| Sub-model registry and its load path | **CONFIRMED** — `shared/src/shared/task_metadata.py::register_metadata_submodel`; the interceptor loads `shared.deploy_state` by a bare side-effect import | α follows that mechanism (PRD open question 3) |
| Sweep host that survives a halt | **CONFIRMED** — `orchestrator/src/orchestrator/background_service.py::BackgroundService`, registered for `stranded-reconcile` in `orchestrator/src/orchestrator/harness.py` | γ adds one sibling registration |
| Dead-holder naming | **CONFIRMED** — `orchestrator/src/orchestrator/session_registry.py::resolve_session_slug_for_pid` returns a slug or `None` | notice names the pid always, the slug when known |
| Sentinel born-at-L2 precedent | **CONFIRMED** — `harness.py::_DIRTY_TREE_ESCALATION_SENTINEL`; role prefix `orchestrator-` is in `escalation/src/escalation/server.py::_HARNESS_SENTINEL_ROLE_PREFIXES` | `orchestrator-deferral-sweep` is an admitted sentinel role |
| Live-claimant predicate | **CONFIRMED** — `shared/src/shared/task_claimant.py::has_live_claimant(task, now, ttl)`; the same-named `fused-memory/src/fused_memory/middleware/live_task_write_guard.py::has_live_claimant` is a different function | γ must import the `shared` one |
| Merge-in-flight lookup | **CONFIRMED** — `orchestrator/src/orchestrator/merge_lane/worker.py::SpeculativeMergeWorker.snapshot` returns `entries` whose dicts carry `task_id` | composes without new substrate; accessor left to γ (PRD open question 5) |
| Pause predicate | **CONFIRMED** — `orchestrator/src/orchestrator/scheduler.py::Scheduler.is_paused` | single "is dispatch halted" question |
| Default grace matches the lease TTL | **CONFIRMED** — `session_registry.py::LEASE_HEARTBEAT_TTL` is two hours, equal to `DEFAULT_GRACE_SECS = 7200` | precedent only; not imported |
| `/proc` readable by the reading processes | **CONFIRMED** — `systemctl --user show` reports `ProtectProc=default`, `ProcSubset=all` for `fused-memory.service`, `orchestrator-dark-factory.service` and `dark-factory-dashboard.service` (the dashboard unit is an addition to PRD §7) | α/β/γ/δ can read `/proc` on the host |
| Single planning-birth path | **CONFIRMED** — `task_interceptor.py::TaskInterceptor._submit_task_planning_mode` holds the only `add_task(status='deferred')` call | β stamps `planning` there |
| `commit_planning` today | **CONFIRMED** — `fused-memory/src/fused_memory/server/tools.py::commit_planning` has no `agent_id` and no `deferral` argument; `set_task_status` has neither `deferral` nor `expected_stamped_at` | β's arguments are genuinely new |
| Dashboard depends on `shared` | **CONFIRMED** — `dashboard/pyproject.toml` lists `dark-factory-shared` | δ imports the shared model |
| Both file memories exist | **CONFIRMED** — `procedural_defer_a_pinned_task_to_guard_a_hand_carry.md` and `feedback_hand_carry_procedure.md` under the project memory directory | ζ can update them |

## Bindings

### α — deferral record and process identity (shared) *(intermediate)*

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `deferral-kind-vocabulary` | capability→producer — `DeferralKind` in `shared/src/shared/task_deferral.py` (PRD §5.1) | PASS |
| `caller-kinds-subset` | capability→producer — `CALLER_KINDS` beside it (PRD §5.1) | PASS |
| `deferral-submodel-registered` | capability→producer — `register_metadata_submodel('deferral', …, cardinality='dict')`; pattern accepts the wrapped-call form | PASS |
| `process-identity-type` | capability→producer — frozen `ProcessIdentity` in `shared/src/shared/process_identity.py` (PRD §5.2) | PASS |
| `liveness-predicate` | capability→producer — `liveness(identity)` in `process_identity.py`, the one liveness test | PASS |
| `lapse-cause-predicate` | capability→producer — pure `lapse_cause(record, now, dead_since)`; γ's `decide` calls it | PASS |
| `needs-human-predicate` | capability→producer — pure `needs_human(record, now, carrier_status, liveness)`; δ and ε call it | PASS |
| `process-identity-capture-and-command-name` | capability→producer — `capture` and `command_name`; manual, two names cannot be ANDed in one grep; boundary row 12 | PASS |
| `registration-load-path` | capability→producer — α makes `shared/src/shared/task_metadata.py` load the `deferral` registration, so `parse_metadata` knows the key in every process (PRD §5.1, amended); RootModel splat verified | PASS |
| `process-identity-unreadable-is-not-dead` | capability→producer — `ProcessIdentityUnreadable`; `DEAD` needs `ENOENT` plus a self-check (decision 6); boundary row 26; manual | PASS |
| `needs-human-expiry-overrun-threshold` | capability→producer — α owns `LAPSE_OVERDUE_AFTER_SECS` (PRD §5.1, amended); formerly `producer-downstream` | PASS |

### β — server enforcement *(intermediate)*

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `deferral-gate-plan-entrypoint` | capability→producer, α upstream — `plan()` in `fused-memory/src/fused_memory/middleware/deferral_gate.py` | PASS |
| `deferral-gate-wired-into-choke-point` | capability→producer (wired) — `task_interceptor.py::TaskInterceptor._apply_status_transition` references the gate module | PASS |
| `deferral-required-refusal-code` | capability→producer — `deferral_required` (PRD §5.3 table) | PASS |
| `deferral-invalid-refusal-code` | capability→producer — `deferral_invalid` with `reason_code`, `field`, `value` | PASS |
| `deferral-changed-refusal-code` | capability→producer — `deferral_changed`, the INV-3 compare-and-set γ relies on | PASS |
| `deferral-write-authority-refusal-code` | capability→producer — `deferral_write_authority`, the fourth code γ maps | PASS |
| `set-task-status-takes-expected-stamped-at` | capability→producer — `server/tools.py::set_task_status` gains the argument | PASS |
| `set-task-status-takes-restamp-only` | capability→producer — the guarded re-stamp ε's migration uses (decision 3, §5.3, amended) | PASS |
| `response-echo-and-restamp-early-return` | capability→producer — success echoes the record; a re-stamp returns before targeted reconciliation (rows 6, 27); manual | PASS |
| `rewrite-audit-trail-carries-deferral` | capability→producer — `SqliteTaskBackend.rewrite_audit_trail` (task 5771) passes `deferral` through; manual | PASS |
| `refusal-before-mutation-at-choke-point` | substrate CONFIRMED — validators run under the write lock before any state change | PASS |
| `atomic-record-and-status-write` | substrate CONFIRMED — `set_status_and_stamp_audit`; removal extension is β's | PASS |
| `clear-on-exit-removes-record` | capability→producer — every exit through the choke point, plus the self-heal's raw-SQL cancel (`auto_cancelled_by_self_heal` path) | PASS |
| `update-task-cannot-bypass-the-floor` | substrate CONFIRMED (to follow, not clone) — the `done_provenance` floor passes an echo only in `replace`; β's compares against the stored row in all three modes | PASS |
| `planning-births-stamped` | substrate CONFIRMED — `_submit_task_planning_mode` is the only deferred birth | PASS |
| `commit-planning-scope-and-agent-id` | capability→producer — `tools.py::commit_planning` gains `agent_id` and `deferral`; `not_planning_hold` | PASS |
| `holder-corroboration-proc-reads` | producer:task-α upstream + substrate CONFIRMED (`ProtectProc=default`) | PASS |
| `repo-tracked-writers-instructed` | capability→producer — six instruction files, re-grep of `skills/` and `docs/`, 15-file cap | PASS |

### γ — expiry sweep, client mapping, de-flake scope *(intermediate: unlocks ζ)*

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `sweep-pass-function` | capability→producer, α and β upstream — `run_deferral_sweep_pass` in `orchestrator/src/orchestrator/deferral_sweep.py` | PASS |
| `sweep-decide-function` | capability→producer — pure `decide(...)` in the same module | PASS |
| `sweep-service-registered` | capability→producer (wired) — `BackgroundService` named `deferral-sweep` | PASS |
| `deferral-rejection-client-error` | capability→producer — `DeferralRejection(SetTaskStatusRejected)` in `orchestrator/src/orchestrator/scheduler.py` | PASS |
| `scheduler-forwards-expected-stamped-at` | capability→producer, β upstream — `Scheduler.set_task_status` forwards it, with optional `agent_id` and `client_op_id` (§5.4, amended) | PASS |
| `sweep-config-keys` | capability→producer — three restart-only keys (not green-tier: `config.py::RELOADABLE_FIELDS` holds no sweep interval); the interval default reads α's constant; manual | PASS |
| `dead-holder-named-in-notice` | substrate CONFIRMED — `resolve_session_slug_for_pid` | PASS |
| `notice-reaches-human-queue-without-pinning` | substrate CONFIRMED — `_DIRTY_TREE_ESCALATION_SENTINEL` precedent; level 2, severity `info` (Leo, 2026-10-09; a documented exception to `BORN_AT_L2_SEVERITIES`, which no reader breaks on), category `deferral_holder_lost`; archive dedup seeded once off-loop | PASS |
| `live-claimant-guard` | substrate CONFIRMED — `shared/src/shared/task_claimant.py::has_live_claimant` | PASS |
| `merge-in-flight-lookup` | substrate CONFIRMED — `SpeculativeMergeWorker.snapshot` entries by `task_id` | PASS |
| `pause-predicate` | substrate CONFIRMED — `Scheduler.is_paused` | PASS |
| `deferred-rows-readable-by-status` | substrate CONFIRMED — `tools.py::get_tasks` `statuses` filter | PASS |
| `de-flake-owner-completed-only-when-planning` | capability→producer, β upstream — `orchestrator/src/orchestrator/flake_ledger.py`, `planning` or unrecorded (same scope as `commit_planning`) | PASS |

### δ — dashboard: draw the hold *(intermediate: unlocks ζ)*

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `dashboard-reads-shared-deferral-model` | capability→producer, α upstream — dashboard imports `shared.task_deferral` | PASS |
| `no-deferral-recorded-state` | capability→producer — the distinct "no deferral recorded" state (INV-13) | PASS |
| `real-records-to-draw` | producer:task-β upstream; δ completes on row 24, and the live drawing moved to ζ step 5; manual | PASS |
| `row-shaping-site` | substrate CONFIRMED — `dashboard/src/dashboard/data/active_tasks.py::_build_task_row` | PASS |
| `dashboard-can-read-holder-liveness` | substrate CONFIRMED — `ProtectProc=default` on the dashboard unit | PASS |
| `needs-human-count-matches-census` | producer:task-ε and α upstream — one `needs_human` feeds both counts; manual | PASS |

### ε — census, migration script, check 6 *(intermediate)*

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `census-script-check-flag` | capability→producer, α and β upstream — `scripts/deferral_census.py` with `--check` | PASS |
| `migration-script-apply-flag` | capability→producer — `scripts/migrate_deferrals.py` with `--apply` | PASS |
| `census-expiry-lapse-threshold` | producer:task-α upstream — `LAPSE_OVERDUE_AFTER_SECS`; formerly `producer-downstream` | PASS |
| `migration-writes-are-guarded-restamps` | producer:task-β upstream — every write is a `restamp_only` guarded re-stamp (row 25) | PASS |
| `migration-mapping-substrate` | substrate CONFIRMED — three source shapes, re-stamped through β's gate; counts cross-checked per `CLAUDE.md` forensic-read guidance | PASS |
| `deferred-read-transport` | substrate CONFIRMED — `get_tasks(statuses=…)` via `scripts/legibility/census_trigger.py::post_mcp_tool_call` | PASS |
| `ownership-reads-carrier-field` | capability→producer — `scripts/sitting/ownership.py::COALESCE_KEY` is today `x_coalesced_into`; ε reads `deferral.carrier_task_id` | PASS |
| `review-briefing-check-6-runs-census` | capability→producer — `skills/review-briefing/SKILL.md` check 6 | PASS |

### ζ — human gate: restarts, live checks, migration, triage *(leaf, pure gate, integration gate)*

Every capability is manual: ζ has no dependents, so a mechanical check would gate
nothing, and its signal is an operator action against live stores.

| Capability | Binding (evidence) | Verdict |
|---|---|---|
| `census-check-exit-code-oracle` | producer:task-ε upstream — the exit-0 oracle and per-project `legacy_unknown` count | PASS |
| `migration-apply-available` | producer:task-ε and β upstream — `--apply` through the enforced gate | PASS |
| `dashboard-needs-human-count-to-compare` | producer:task-δ upstream — the header count the signal compares | PASS |
| `sweep-running-for-lapse-arm` | producer:task-γ upstream — the census lapse arm presumes the sweep is live | PASS |
| `file-memories-to-update-exist` | substrate CONFIRMED — both files exist outside the repo | PASS |
| `post-restart-live-checks` | producer:task-β, γ, δ upstream — ζ steps 1, 4 and 5 after the operator's restarts | PASS |
| `successor-owner-for-remaining-holds` | ζ step 7 — a successor pure gate with a 30-day delayed milestone per project with needs-human rows left (INV-7) | PASS |

## Which checks are mechanical, and why each name is fixed

Of 63 capabilities, 25 carry a mechanical `grep` check and 38 are `manual`
(including all of ζ). Mechanical counts per producer: α 8, β 8, γ 5, δ 2, ε 2, ζ 0.
α, β, γ, δ and ε all have dependents (α gates β, γ, δ, ε; β gates γ, δ, ε, ζ; ε gates δ
and ζ; γ and δ gate ζ), so every mechanical check gates something. ζ's are recorded
as manual.

Every pattern is a `git grep -E` expression scoped by `paths` to production code,
so it cannot be satisfied by the PRD, this manifest or the sidecar, and not by a
test. Each fails on main today (evaluated at `main`, outcome `fail` for all 25) and
none is a filename pattern. Each name below appears verbatim in the PRD contract
or file plan, so an implementer has no legitimate alternative:

- **α.** `DeferralKind`, `CALLER_KINDS`, `lapse_cause`, `needs_human` (PRD §5.1 and
  decisions 6 and 10), `LAPSE_OVERDUE_AFTER_SECS` (§5.1 as amended at decompose), `ProcessIdentity`, `liveness` (PRD §5.2) and the key
  `deferral` in `register_metadata_submodel` (PRD decision 1) are all named, with
  their module paths. Module paths are fixed by §5.1 and §5.2.
- **β.** `deferral_gate.py::plan` (decision 3 and §13) and the four error codes of
  the §5.3 table, `deferral_required`, `deferral_invalid`, `deferral_changed` and
  `deferral_write_authority`. The code checks are scoped to both
  `fused-memory/src/fused_memory/` and `shared/src/shared/`, because the
  orchestrator client cannot import `fused_memory` and an implementer may
  legitimately home the code strings in `shared/`. `expected_stamped_at` and
  `restamp_only` on `tools.py::set_task_status` are in the §5.3 signature (the second
  added at decompose). The wiring check keys on
  the module name `deferral_gate`, which the file path fixes.
- **γ.** `run_deferral_sweep_pass` and `decide` (§5.4), the `deferral-sweep`
  service name (decision 7; the quoted form excludes the `orchestrator-deferral-sweep`
  agent id), `DeferralRejection` (§5.4) and the `expected_stamped_at` forward.
- **δ.** The shared module name `task_deferral` and the display state "no deferral
  recorded" (decision 10 and the δ signal), scoped to the dashboard package because
  the PRD does not fix which file composes the text.
- **ε.** The `--check` flag (goal and decision 10) and the `--apply` flag
  (decision 9), each scoped to its fixed script path.

Deliberately not pinned, and therefore manual: `capture` and `command_name` (two
names, one grep cannot AND them), the `Liveness` enum (its home is fixed in
`process_identity.py` by the amended §5.2 and §12 item 7; it is left manual because the
`liveness` check already covers the producer), `ProcessIdentityUnreadable` and the
`proc_root` seam (behaviour proved by row 26), the
`DeferralWriteAuthorityError` class (no dependent imports it; the code string is
bound instead), `deferral` argument names on `set_task_status` and `commit_planning`
(an argument name is too generic to grep without false matches), the merge-queue
accessor (open question 5 leaves it to γ), the config keys, the census and
dashboard needs-human strings, and every skill or doc edit.

Polarity: all 25 mechanical checks pass `lint_delivered_checks` at `main` with zero
findings. They also evaluate green against a synthetic tree carrying plausible
implementations (one-line and wrapped registration, `StrEnum` and annotated
`CALLER_KINDS`, async and sync pass functions), so none can wedge on spacing or
quote style.
