# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

#### MCP write tools now reject leaked tool-call XML (task 3083)

`submit_task`, `update_task`, `add_memory`, and `add_episode` now **reject** a
call whose text fields carry a leaked serialized tool-call XML fragment,
returning `error_type = 'ToolCallXmlLeakError'` **before anything is
persisted**. Scanned fields: `title`, `description`, `prompt`, and `details`
for the task tools; `content` for the memory tools. Each field is scanned
independently, so a match cannot straddle two clean fields.

**Why this is a behaviour change and not just a validation tweak.** These
fragments never originate in this repo — there is no XML parser anywhere in
`fused-memory/src/`. They are evidence that the harness's tool-call parser
terminated a string argument early at a literal closing tag inside that
argument's value, which also **silently swallows the sibling arguments that
followed**. A leaked fragment in `description` means `priority` may never have
reached the MCP boundary at all, in which case
`sqlite_task_backend.py`'s `priority or 'medium'` substituted a plausible
wrong value with no log. The rejection message says so explicitly: that
sentence is what converts an invisible wrong-value failure into a visible one.

The guard **rejects rather than sanitizes** — silently stripping the fragment
would be a second silent mutation of user content layered on the first. It runs
at the top of the write prologue, ahead of `inject_execution_class` /
`inject_operational_routing`, so routing is never derived from already-truncated
text.

**Opt-out:** `metadata={'allow_toolcall_xml': True}`, mirroring the existing
`allow_near_duplicate` convention. Required for any write that legitimately
quotes the fragment — task 3083's own description, and
`docs/mcp-toolcall-xml-leak.md`, would otherwise be unfileable.

**Also added:** `scan_memory_content` (read-only literal substring scan over
Qdrant payload text — semantic `search` provably cannot find these),
`redact_episode_content` (neutralise a leaked Graphiti episode without
`delete_episode(cascade=True)` destroying its valid extracted edges), and
`fused-memory/scripts/sweep_toolcall_xml_leak.py` (the corpus sweep; dry-run by
default).

**What the sweep's repair preserves.** The memory **text** in full, plus the
**payload metadata** that is the record's only metadata-scoped retrieval axis —
carry-over rule `payload keys - _MEM0_OWNED_KEYS`, with the scope identities
threaded back as `agent_id=` / `session_id=` arguments and anything carried
nowhere *named* per-record in the report's `metadata_dropped`. Without that,
a repaired record would silently vanish from `get_memories_by_metadata` /
`count_memories_by_metadata`, which match payload keys by equality.

**The sweep verifies the re-add persisted** rather than trusting a non-raising
`add_memory`: the service swallows a Mem0 write failure into
`AddMemoryResponse.message` as `[mem0_error: ...]` and returns normally, so a
returned response is not evidence of a write. Three per-record outcomes exit
non-zero and need a human — `content_lost_in_flight` (the delete landed, the
re-add did not persist; the original text now exists only in the printed report,
restore it from there before re-running), `skipped_not_mem0_routed` (a
repairable record whose category does not route to mem0, left entirely untouched
because neither a plain re-add nor `dual_write=True` is safe), and
`record_error` (that record's repair aborted on an unexpected error, so whether
its delete landed is unknown).

**The report always survives an abort**, since for a `content_lost_in_flight`
record it is the only remaining copy of the original text. Each record enters
the report *before* any store mutation is attempted and its repair runs under
its own `try`, so one record's transport failure is recorded as `record_error`
on that record and the sweep continues instead of unwinding and discarding every
earlier entry. If anything escapes anyway, the CLI prints the **partial** report
— same shape, plus `"aborted": true` — before exiting `2`.

**Full root cause, evidence, and operator runbook:**
[`docs/mcp-toolcall-xml-leak.md`](docs/mcp-toolcall-xml-leak.md). Run the sweep
**before** any further large consolidation pass — consolidation deletes
corrupted entries as a merge side effect and destroys the specimens.

### Changed (BREAKING)

#### `reservation_installed` (reason=reserve_now) → `reserve_now_consumed` (task 1230)

The reserve-now short-circuit path in the scheduler **no longer emits**
`reservation_installed` with `data.reason == 'reserve_now'`.  It now emits the
dedicated `reserve_now_consumed` event instead.

**Old behaviour (pre-task-1230):**

```
event_type : reservation_installed
data       : {modules, priority, reason='reserve_now'}
```

**New behaviour:**

```
event_type : reserve_now_consumed
data       : {modules, priority}
```

**Commits:** `4d45eecd9b` (add `reserve_now_armed`/`reserve_now_consumed`
`EventType` members) · `deb8f426ab` (replace `reservation_installed` with
`reserve_now_consumed` at the scheduler short-circuit emit site, steps 5-6).

**Migration note:**  Any downstream consumer (dashboard query, log filter,
reconciliation tooling, external subscriber) that was filtering on
`event_type = 'reservation_installed' AND data->>'reason' = 'reserve_now'`
must migrate to `event_type = 'reserve_now_consumed'`.

**Threshold-based reservation path is UNCHANGED** — the scheduler still emits
`reservation_installed` when the skip-count threshold is exceeded
(`scheduler.py:1593`); that event's data payload has never contained a
`reason` key (`data = {modules, skip_count, priority}`).  The two events are
now discriminated by **event name**, not by a `reason` field:

| Event | Path | Data keys |
|---|---|---|
| `reservation_installed` | threshold (skip_count ≥ threshold) | `modules`, `skip_count`, `priority` |
| `reserve_now_consumed` | reserve-now short-circuit | `modules`, `priority` |

**Dual-emit rejected:** Option (b) — emitting both `reservation_installed` and
`reserve_now_consumed` during a deprecation window — was evaluated and
rejected.  It would protect zero in-repo consumers (see audit below) and would
require reverting the deliberately-added, merged locked-in regression test
`TestReserveNowConsumedShortCircuit` (part b, `test_scheduler_state.py:191-197`)
from task 1230, contradicting a merged decision for no benefit.

**Audit result (task 1333) — in-repo blast radius: ZERO:**

An exhaustive search across all `*.py`, `*.md`, `*.sql`, `*.js`, `*.ts`,
`*.html`, `*.yaml`, `*.yml`, `*.json` files (excluding `.venv/`, `.git/`,
`uv.lock`) found the following `reservation_installed` sites:

- `orchestrator/src/orchestrator/event_store.py:75` — enum definition only
- `orchestrator/src/orchestrator/scheduler.py:1593` — **threshold-path emit**
  (unrelated to reserve_now; no `reason` key; UNCHANGED)
- `orchestrator/src/orchestrator/scheduler.py:2030` — code comment only
- `orchestrator/tests/test_scheduler_state.py:155,191-197` — task-1230
  locked-in regression test (asserts short-circuit emits `reserve_now_consumed`
  and no legacy `reservation_installed` with `reason=='reserve_now'`)
- `orchestrator/tests/test_scheduler.py:2019,2206,2230-2234,3899,3952-3955,4022-4025`
  — tests of the threshold `reservation_installed` path (unaffected)

Dashboard (`dashboard/`): all `reserve_now` references concern the override
*flag* (UI badges, clear-fields allow-list, POST body, scheduler-snapshot
reads) — the dashboard does **not** consume `reservation_installed` events from
the event_store.  No `scripts/`, SQL, JS/TS, or documentation file consumes or
filters the event.

No emit site in the current (post-task-1230) tree sets `data.reason` on
`reservation_installed`.  (Before task 1230 the reserve-now short-circuit did
emit `reservation_installed` with `reason='reserve_now'`; removed in
`deb8f426ab` — historical event-store rows predating task 1230 may still
contain it.)  The renamed event's pre-1230 form (`reservation_installed` +
`data.reason='reserve_now'`) had zero in-repo consumers.
