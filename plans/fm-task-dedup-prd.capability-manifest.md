# Capability manifest — fm-task-dedup (W8)

Mechanizes G3 (assumed-substrate) + G6 (premise validity) per task: each signal's
asserted capabilities bound to evidence. Substrate re-verified against `main`
2026-07-06. Labels are PRD-local (`plans/fm-task-dedup-prd.md` §8); task IDs
assigned at decompose. **No binding resolves to a FAIL value** → batch clears the
gate.

Evidence vocabulary: `substrate:<file>:<line>` (exists on main today),
`self:<deliverable>` (this task builds it, wired into its own production path),
`producer:<label> upstream` (delivered by a task in the transitive dependency
closure that is upstream of this one).

---

## A1 — candidate_key column + compute-on-insert + backfill + report

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| SQLite supports adding a nullable column + partial-index-capable schema | `substrate:` SQLite 3.45.1 (partial idx since 3.8.0); `_migrate`/`PRAGMA user_version` mechanism `substrate:fused-memory/src/fused_memory/backends/sqlite_task_backend.py:165` | PASS |
| Normalization definition (`sha256_16(title\|sorted(files))`) exists to reuse | `substrate:.../task_curator.py:663` (`_normalize_key`), `:453` (`normalize_title`) — A1 extracts to a low-dep leaf + delegates | PASS |
| `add_task` is the single store-level INSERT chokepoint reachable by all paths | `substrate:.../sqlite_task_backend.py:694` (add_task INSERT :749); planning_mode reaches it `substrate:.../task_interceptor.py:1716` | PASS |
| `_row_to_task` can expose the new column | `self:` A1 adds `candidate_key` to `_row_to_task` `substrate:.../sqlite_task_backend.py:274` | PASS |
| Migration report count may be **0** (no numeric premise) | signal explicitly asserts "count may be 0" — **no floor asserted** (G6 branch-1 N/A) | PASS |

## A2 — self-gating partial UNIQUE index + collision→combined

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Partial UNIQUE index over non-cancelled non-null rows | `self:` `CREATE UNIQUE INDEX … WHERE candidate_key IS NOT NULL AND status != 'cancelled'`; column from `producer:A1 upstream` | PASS |
| Backend surfaces IntegrityError as `DuplicateCandidateKeyError(existing_id)` | `self:` new typed error + existing-row SELECT in `add_task` (`producer:A1` populated the key) | PASS |
| Interceptor create-dispatch resolves `combined` | `self:` catch in `_dispatch_ticket_decision` create branch `substrate:.../task_interceptor.py:2960-3097` (mark_resolved status='combined' pattern already exists `:2789`) | PASS |
| planning_mode resolves `combined` (reintroduction guard) | `self:` catch in `_submit_task_planning_mode` `substrate:.../task_interceptor.py:1712-1741` | PASS |
| Rejection premise (duplicate creation is prevented) — mechanism exists + fires | G6 branch-4: A2 **builds** the index + collision path and the signal binds it by authoring a duplicate and observing the `combined` outcome; capability is self-produced, not downstream | PASS |
| Fail-safe skip on residual dups (no service-fatal raise) | `self:` connection-open detect+log+escalate+skip (decision #4); escalator substrate `substrate:.../task_interceptor.py:1117` (scope_violation escalator pattern) | PASS |

## A3 — candidate_key B+H boundary-test gate

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Real submit→resolve / planning_mode / crash-injection / restart paths exist to drive | `producer:A2 upstream` (enforcement) + `substrate:` two-phase submit `.../task_interceptor.py:1543`, planning_mode `:1653` | PASS |
| Crash between INSERT and COMMIT rolls back (no orphan row) | `substrate:` `add_task` runs INSERT under `self._txn` `.../sqlite_task_backend.py:728`; A3 injects the fault via a test hook | PASS |
| Property survives restart (in-memory caches cold) | durability is a property of the DB index (`producer:A2`), not the in-memory layers (`substrate:.../task_curator.py:501` `_recent_creates`) | PASS |
| Signal is not a synthetic-input unit (G2) | drives the production submit/resolve/planning_mode read paths end-to-end — C-as-integration-gate | PASS |

## B1 — per-ticket lifecycle dataclass

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The parallel arrays + two index-spaces exist to replace | `substrate:.../task_interceptor.py:2814` (`non_none_to_ticket_data`), `:2832` (`curator_degrade_reasons`), `:2972` (`resolved_task_ids`), `:2843-2884` (index-space remap) | PASS |
| Mixed-batch terminal correctness observable via resolve_ticket | `substrate:` `_persist_worker_terminal` + `_signal_ticket_event` `.../task_interceptor.py:3056,3100`; behavior-preserving refactor | PASS |

## C1 — backend privileged write-authority seam

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Backend `update_task` can host a done_provenance floor | `substrate:.../sqlite_task_backend.py:804` (status floor already there — extend) | PASS |
| Canonical rejection-shape home exists | `substrate:.../backends/task_backend_errors.py` (TaskmasterError `:20`) — C1 adds the two `*_error()` + typed subclasses | PASS |
| `stamp_audit_metadata` can do privileged RMW off-protocol | `self:` new non-protocol method; not added to `TaskBackendProtocol` (`substrate:.../backends/task_backend_protocol.py`) | PASS |
| Rejection assertion (`update_task(done_provenance)` rejected) fires | G6 branch-4: C1 builds the floor; signal authors the call and observes the canonical rejection — self-produced | PASS |

## C2 — rewire sanctioned writer + delete guard copies (C integration-gate)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `stamp_audit_metadata` available to the sanctioned writer | `producer:C1 upstream` | PASS |
| Sanctioned writer currently persists via public update_task (to rewire) | `substrate:.../task_interceptor.py:719,782` (reopen + done_provenance via `tm.update_task`) | PASS |
| The three guard copies exist to delete | `substrate:.../server/tools.py:2314,2342`; `.../task_interceptor.py:3872,3900`; backend floor `.../sqlite_task_backend.py:804` | PASS |
| tools.py surfaces backend typed errors as structured dicts | `substrate:.../server/tools.py:1178-1190` (`except … return {'error',…'error_type'}`) — canonical `.to_error_dict()` passthrough (Open Q #2) | PASS |
| `interceptor_write_succeeded` collapses to one shape | `substrate:.../task_interceptor.py:3958` (currently enumerates shapes) — C2 reduces to canonical | PASS |
| "Byte-identical rejection across surfaces" premise | both surfaces route to the one `task_backend_errors.py` definition (`producer:C1`) — achievable by construction | PASS |

## D1 — structured routing (reject-on-files, advise-on-prose)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Registry can resolve a concrete file path → owner (exact, no regex) | `substrate:.../project_prefix_registry.py:168-188` (`all_prefixes`/`project_for_prefix`) — D1 adds `project_for_path` on the same class | PASS |
| Interceptor path-guard decision point exists to split | `substrate:.../task_interceptor.py:1057` (`_path_guard_check`), `:1117` (scope escalator) | PASS |
| Advisory metadata write path exists | `substrate:` metadata is file-level per Lock-charter; `_inject_routing_override` warns-and-writes pattern `substrate:.../task_interceptor.py:1451` | PASS |
| Reject premise (files-owner-mismatch rejected) fires | G6 branch-4: D1 builds `project_for_path` + files-reject; signal authors a cross-project-files task and observes the structured reject — self-produced | PASS |
| Prose-hit is created not rejected (advisory) — capability is in D1's own set | `self:` D1 delivers the advisory branch; not owned downstream | PASS |

## D2 — delete dark_factory_path_guard shim

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Shim is the current interceptor import (to switch) | `substrate:.../task_interceptor.py:34-38` (`from …dark_factory_path_guard import …`) | PASS |
| Registry default can carry the hard-coded prefixes | `substrate:.../dark_factory_path_guard.py:37` (`DARK_FACTORY_PATH_PREFIXES`) → fold into registry default (`producer:D1` established `project_for_path`) | PASS |
| Post-delete grep-empty is observable | `self:` D2 removes the module; `grep dark_factory_path_guard fused-memory/src` = ∅ | PASS |
| dark-factory mis-file still rejected via registry default | `producer:D1 upstream` (files-reject over the registry) | PASS |

## Z — deterministic deploy capstone (fused-memory restart)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `before_done.script` exists + executable (validated at submit_task) | `substrate:scripts/restart-fused-memory.sh` (executable; no-arg = `systemctl --user restart` :48 + health wait, **no --drain**) | PASS |
| Cross-unit deterministic deploy path (blocking + fresh-PID verify) | `substrate:` CLAUDE.md deterministic task-kind conventions; `target_unit=None` ⇒ cross-unit blocking | PASS |
| Schema migrations run on restart | `substrate:.../sqlite_task_backend.py:165` `_migrate` at connection-open — restart re-opens | PASS |
| Enforcement-live OR fail-safe-escalate outcome (honest signal) | `producer:A2` (self-gating index); both outcomes are successful deploys (decision #4) | PASS |
| fused-memory is the only process needing restart | orchestrator consumes none of these mechanisms cross-process (candidate_key internal; write-authority + routing are fused-memory-side) | PASS |

---

**Gate result: PASS.** Every asserted capability binds to substrate-on-main,
a self-deliverable wired into its own production path, or an upstream producer in
the dependency closure. No `declared-only` / `test-only` / `producer-downstream`
/ `producer-absent` / `producer-extent-short` / `fixture-ERROR` / `bound≤floor` /
`rejection-absent` bindings. Two tactical items tracked in the PRD's Open
Questions (§10 #2 tools.py passthrough, #5 files-key name) — neither is a
substrate FAIL.
