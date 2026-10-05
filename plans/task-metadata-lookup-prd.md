# Task metadata lookup: `find_tasks_by_metadata`

**Status:** authored 2026-10-04 (seat S6 of the quality-skills alignment team), not yet
decomposed.
**Type:** one new read path over the existing task store. No new store, no schema change.
**Approach:** B+H, small — one seam, three consuming plans (G5 "cross-PRD consumers ≥ 2").
**Code anchors** verified against `92af0716e8` (worktree `quality-skills-alignment`,
2026-10-04). Cite-by-symbol; re-locate at implementation time.
**Ruling (Leo, 2026-10-04):** add a `find_tasks_by_metadata` fused-memory MCP tool for
finding-key lookups; the task store is the disposition home (no new store).

## Goal

A caller holding a finding key, a run id, or any other top-level `metadata` value can
ask fused-memory "which tasks carry this?" and get an exact, complete answer over every
task in every status, including `deferred` and `cancelled`. Observable when it lands:

- `find_tasks_by_metadata(project_root="/home/leo/src/dark-factory", key="files",
  value="scripts/merge_lane_metrics.py")` over MCP returns task 5414 among 15 matches
  (count measured 2026-10-04; a list-valued key matched by membership).
- `docs/quality-findings-contract.md` §8 step 1 and the triage text of `/review`,
  `/review-all` and `/hotspot-survey` call the tool, with no interim forensic-sqlite
  fallback left in them.
- `scripts/check_run_completion.py` (owned by `plans/completion-driven-triggers-prd.md`)
  reads a run's filed tasks with `key="x_finding_run"` through the same tool.

## Background

`docs/quality-findings-contract.md` §7 makes the task store the home of every
disposition that is somebody's decision; §8 step 1 deduplicates a finding by looking
its key up across *all* statuses, because step 1b needs `cancelled`-with-reason and
`search_tasks`' curator corpus excludes `deferred` (`docs/task-authoring.md` §9,
"Verify a batch is filed with `get_task`"). No tool answers "tasks whose metadata key K
has value V": `get_tasks` returns the whole tree, `search_tasks` is semantic and
corpus-scoped, `get_statuses` carries no metadata. The interim answer in the contract
and in `skills/review/references/phase3-triage.md` is a read-only `sqlite3` query
against `.taskmaster/tasks/tasks.db` — the forensic path `CLAUDE.md` §"Forensic reads
of tasks.db" reserves for forensics, not for an operating protocol.

Measured on the live store, read-only (`mode=ro`), 2026-10-04 at load average 52–175 on
32 cores, Python sqlite 3.45.1, median of 9–15 repeats:

| Measure | Value |
|---|---|
| Rows / tags / `metadata` bytes | 6,281 / 1 (`master`) / 13.2 MB |
| Status mix | done 4,389 · pending 1,238 · deferred 339 · cancelled 278 · in-progress 25 · blocked 10 · merge-deferred 2 |
| Rows whose `metadata` fails `json_valid` | 0 |
| Rows carrying any `x_finding*` key | 0 (no producer has written one yet) |
| Top-level keys holding a JSON list (rows) | `files` 5,297 · `modules` 1,591 · `dry_run_proposals` 589 · `related_tasks` 531 · `delivered_checks` 440 · `x_coalesced_from` 124 |
| Full-table `json_extract`/`json_each` match, scalar-or-membership | 58–91 ms (scalar key `source`, list key `files`, absent key) |
| `SELECT metadata` with no JSON work | 46 ms |
| `SELECT *` + `json.loads` of every row (what `get_tasks` pays) | 275 ms |
| One row in `search_tasks`' shape (description clipped at the corpus's 1,000 chars) | median 1,420 B, p95 1,954 B, max 8,931 B; worst 50 rows 160 KB |
| One row without `description` / `files_to_modify` | median 252 B, max 2,072 B; worst 50 rows 25.6 KB |

The MCP envelope documented safe at ~62 KB (`fused-memory/src/fused_memory/server/tools.py::get_statuses`
docstring) is exceeded by 50 median rows in `search_tasks`' full shape, which decides
decision 4 below.

## Sketch of approach

One SQL method on the store owner, passed through the interceptor's pure-read section,
exposed as one MCP tool, documented where task tools are listed. Callers in another
process reach it over MCP; nothing outside fused-memory opens the store for this.

## Resolved design decisions

1. **One implementation, at the store owner.** The query lives in
   `SqliteTaskBackend.find_tasks_by_metadata` (`fused-memory/src/fused_memory/backends/sqlite_task_backend.py`),
   declared on `TaskBackendProtocol`, and reads through `SqliteTaskBackend._fresh_read_conn`
   exactly as `SqliteTaskBackend.get_task` does. `TaskInterceptor.find_tasks_by_metadata`
   (`fused-memory/src/fused_memory/middleware/task_interceptor.py`, "Pure reads" section)
   is a pass-through like `TaskInterceptor.get_task`: no event, no journal. The MCP tool in
   `fused-memory/src/fused_memory/server/tools.py` is the only out-of-process entry. This
   places the helper one layer below where the team brief of 2026-10-04 suggested ("the
   interceptor"): the interceptor holds no SQL for any read and should not start (heuristic 9, *deep
   modules with appropriate nesting* — the backend hides the store, the interceptor
   hides the backend).
2. **Scripts call the MCP tool; nothing imports the helper.** A predicate or census
   script runs as its own process. Importing the backend would open the SQLite file from
   a second process outside the per-project read lock, re-derive the per-project store
   path the server already resolves from `project_root` (reify's store included), pull
   the `fused_memory` package into scripts that do not depend on it, and turn the
   forensic exception of `CLAUDE.md` into an operating path (heuristic 11, SPOT; heuristic
   7, *stateless interactions* with an explicitly owned store). Reachability:
   `orchestrator/src/orchestrator/deterministic_runner.py::DeterministicRunner` spawns a
   `before_done` script with `os.environ` merged under its own env, on the host where
   fused-memory listens; `scripts/legibility/census_trigger.py::post_mcp_tool_call` is
   the existing transport scripts use to call fused-memory tools at
   `FUSED_MEMORY_MCP_URL` (default `http://localhost:8002`). A store or transport
   failure surfaces as an error payload, which `check_run_completion.py` maps to an exit
   other than 0/75 and so escalates (contract §11) — never as "not yet landed".
3. **`json_extract` scan, no index.** The tool is generic over the key, so an
   expression index serves only the one key it names; list membership (contract §8's
   `x_finding_key` may be a list) cannot use a B-tree expression index at all and would
   need a side table, which is a second store for the same fact (ruled out). The scan
   costs 58–91 ms on 6,281 rows under heavy load, a third of what `get_tasks` already
   pays per call; call volume is one lookup per finding per attended run plus one per
   completion-gate recheck. The `statuses` filter narrows the scan through the existing
   `ix_tasks_status (tag, status)` index. Revisit only when a measured median exceeds
   500 ms (≈ 40k rows at linear scaling); the remedy then is an expression index on
   `x_finding_key` alone, with membership still scanned. `candidate_key` is not the
   precedent here: it is a server-computed scalar with a uniqueness invariant, not a
   caller-chosen key.
4. **Row shape: `search_tasks`' row minus the bulky fields.** Each row is
   `{task_id, title, status, priority, updated_at, matched_value}` — `search_tasks`'
   fields minus `score` (no ranking), `description` and `files_to_modify`. Including
   the last two puts 50 median rows past the documented-safe MCP envelope (Background);
   without them the worst 50 rows measured 25.6 KB. A caller that needs the record
   (contract §8 step 1b reads a `cancelled` task's stated reason) calls `get_task` on the
   one id, which it must do anyway to read `details`.
5. **Match semantics.** `key` is one top-level `metadata` key; nested paths are out of
   scope. `value=None` matches every task where the key is present (including a JSON
   `null`). A `str` or `int` value matches a scalar by equality and a JSON list by
   membership of an element. `bool`, float, dict and list values are rejected: SQLite's
   `json_extract` returns `1`/`0` for JSON booleans, so a bool would silently match an
   integer (heuristic 12, *structured data*; INV-2 names the offending value). `key` must
   match `^[A-Za-z0-9_][A-Za-z0-9_-]*$` and is spliced only as a quoted JSON path
   (`'$."' || key || '"'`), never into SQL text.
6. **Completeness is in the result, not a log line** (INV-11). The response always
   carries `pagination` from `tools.py::_pagination_meta` (`total`, `offset`,
   `page_size`, `returned`, `has_more`), so a truncated page cannot read as complete.
   `limit` defaults to 50 and is clamped to [1, 100] like `search_tasks`; the worst 100
   rows of the reduced shape stay under the envelope (≤ 51 KB: the worst 50 total
   25.6 KB, so each of the next 50 is at most the 50th's ~0.5 KB). A row whose `metadata` fails
   `json_valid` is skipped by a `CASE` guard (so one corrupt row cannot fail the whole
   query) and its id is reported under `malformed_metadata_ids`, present only when
   non-empty.
7. **An empty answer is distinguishable from an absent key** (INV-13). The response
   carries `tasks_with_key`: how many tasks in the filtered corpus carry `key` at all,
   from the same scan. `results: []` with `tasks_with_key: 0` says "no task in this
   corpus carries this key" (a producer that has not written yet, or a misspelt key);
   `tasks_with_key > 0` says the key is live and nothing matches this value.
8. **`tag` is accepted for parity** with `get_task`/`get_tasks` (default `master`); the
   signature in the team brief omitted it.

## Pre-conditions for activating

None. Every substrate the leaf needs exists at `92af0716e8` (G3 bindings below). The live
signal becomes observable when the running fused-memory service restarts onto the
landed code.

## Cross-PRD relationship

| Other PRD / surface | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/quality-findings-contract.md` §8 step 1 | consumes | the tool's name, signature and all-status corpus. §8 lets `x_finding_key` be a list, so step 1 relies on the tool's list-membership semantics (decision 5): a task filed with several keys must be found by any one of them | **this PRD** implements; the contract names it. L1 deletes the contract's "until it lands" fallback clause | contract in authoring 2026-10-04 |
| `plans/completion-driven-triggers-prd.md` leaf γ (in-process) | consumes | the one-open-chain guard `fused-memory/src/fused_memory/middleware/trigger_chain_guard.py::trigger_chain_open_error`, wired into `tools.py::submit_task`, calls `TaskInterceptor.find_tasks_by_metadata(key='trigger_chain', value=None, statuses=ACTIVE)` — a presence query. It filters each row's `matched_value["skill"]` itself, which is why `matched_value` carries the whole top-level value (decision 5, nested paths out of scope). The in-process signature is the Contract's, minus the MCP-only `project_id`/`project_root` echo | **this PRD** owns the interceptor method; γ depends on L1. γ's own PRD decides that the guard fails open on a read error; this PRD's method raises and never returns an empty result on failure, so the guard can tell the two apart | that PRD in authoring 2026-10-04 |
| `plans/completion-driven-triggers-prd.md` | consumes | `find_tasks_by_metadata(key="x_finding_run", value=<run_id>, limit=100)` paged on `has_more`, called from `scripts/check_run_completion.py` via `scripts/legibility/census_trigger.py::post_mcp_tool_call`; needs `status` and `priority` per row for the §11 weighted share | **this PRD** owns the tool; that PRD owns the script and its task depends on L1 | that PRD in authoring 2026-10-04; dependency wired at its decompose |
| `plans/census-incremental-prd.md` | consumes | L4: contract §8 step 1 on `key="x_finding_key"` plus the backfill keyed on `source`, over `census_trigger.py::post_mcp_tool_call`. L6 consumes L4's back-links (a fixed entry re-observed after its fix), so it reaches the tool only through L4. The trigger's weighted completion reads `x_finding_run` inside `check_run_completion.py` (its L8b), not in census code | **this PRD** owns the tool; that PRD's L4 hard-depends on L1 (its own R7), and L6 follows L4. A store or transport failure reaches it as an `error` payload, not a raised exception — its boundary row 8 must test for the payload | that PRD in authoring 2026-10-04 |
| `skills/review/references/phase3-triage.md`, `skills/review-all/**`, `skills/hotspot-survey/**`, `skills/_shared/filing-the-trigger-chain.md` | consume | contract §8 step 1 | the skills' seats own their text; L1 only removes the interim forensic-query sentences once the tool is live | skill edits in flight 2026-10-04 |

No reciprocal ambiguity: every row names this PRD as the tool's owner and the other side
as the caller. If the completion-triggers PRD finds a census-named module the wrong home
for a general MCP transport, moving `post_mcp_tool_call` is that PRD's call, not this one's.

## Contract (B+H)

```
find_tasks_by_metadata(
    project_root: str,
    key: str,
    value: str | int | None = None,
    statuses: list[str] | None = None,     # None = every status
    limit: int = 50,                       # clamped to [1, 100]
    offset: int = 0,
    tag: str | None = None,
) -> {
    'project_id', 'project_root', 'key', 'value', 'statuses',
    'results': [{'task_id': int, 'title', 'status', 'priority', 'updated_at', 'matched_value'}],
    'tasks_with_key': int,
    'pagination': {'total', 'offset', 'page_size', 'returned', 'has_more'},
    'malformed_metadata_ids': [int],       # only when non-empty
}
```

- Results are ordered by task id ascending, so pages tile the match set.
- `matched_value` is the task's whole `metadata[key]` (the list, when it is a list).
- Validation errors return `{'error', 'error_type': 'ValidationError'}` naming the
  offending argument and value, before the store is touched (`tools.py::_validate_paging`
  for `limit`/`offset`; bare-string `statuses` rejected as in `get_tasks`).
- Store errors propagate as the `mcp_tool_errors` payload. Never `{}`, never
  `results: []` on failure: the contract §8 rule "a run that cannot perform step 1 files
  nothing" depends on telling the two apart.
- Pure read: no reconciliation event, no journal row; one `_log_read` entry like
  `get_task`.

## Boundary-test sketch

All rows run against a real SQLite task store built by the backend in a temp
`project_root`, driven through the MCP tool function, never by patching backend
internals (Tests stance, `docs/code-quality.md`).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Scalar match across statuses | three tasks with `x_finding_key="fk-aaa…"` in `pending`, `deferred`, `cancelled` | all three returned, ids ascending; `tasks_with_key == 3` |
| 2 | List membership | task with `x_finding_key=["fk-a","fk-b"]` | `value="fk-b"` returns it; `matched_value` is the list |
| 3 | Presence | `value=None`, one task with `x_k: null`, one without `x_k` | the first is returned, the second is not |
| 4 | Int value against an int list | `related_tasks=[12, 40]` | `value=40` matches; `value="40"` does not |
| 5 | Bool rejected | `value=True` | `ValidationError` naming `value=True`; store not read |
| 6 | Key grammar | `key='a"; DROP'` | `ValidationError`; no SQL executed |
| 7 | Paging | 120 matches, `limit=100` | first page `returned=100, has_more=True`; second page with `offset=100` returns 20, no overlap |
| 8 | Clamp | `limit=500` | `page_size == 100` in `pagination` |
| 9 | Corrupt row | one row with non-JSON `metadata` | query succeeds; its id in `malformed_metadata_ids`; other matches returned |
| 10 | Status filter | `statuses=['cancelled']` | only `cancelled` rows; `tasks_with_key` counted within the filter |
| 11 | Unknown key | key nobody carries | `results == []`, `tasks_with_key == 0` |
| 12 | Consumer face | `scripts/legibility/census_trigger.py::post_mcp_tool_call` against a live server in the fused-memory MCP integration harness | the unwrapped payload has the Contract shape |

## Decomposition plan

One leaf. A separate documentation leaf would sit under the overlay's ~100 LOC floor
(`.claude/skills/prd/project.md` §Task sizing bands), so the doc and skill-text edits ride
with the code.

- **L1 — `find_tasks_by_metadata`: backend query, protocol, interceptor pass-through, MCP
  tool, docs.** [high; `task_kind='normal'`; ~500–700 LOC with tests; ~12 files]
  Files: `fused-memory/src/fused_memory/backends/sqlite_task_backend.py`,
  `fused-memory/src/fused_memory/backends/task_backend_protocol.py`,
  `fused-memory/src/fused_memory/middleware/task_interceptor.py`,
  `fused-memory/src/fused_memory/server/tools.py` (tool +
  `FUSED_MEMORY_INSTRUCTIONS` "Task operations" line),
  `fused-memory/tests/test_find_tasks_by_metadata_tool.py` (new),
  `docs/task-authoring.md` (a §9 recipe "Look a task up by a metadata key", and one
  sentence under §8 Tier-C saying `x_` keys are findable through it),
  and every interim forensic-sqlite fallback, each replaced by the tool call:
  `docs/quality-findings-contract.md` §8 step 1 ("until it lands" clause);
  `skills/_shared/filing-the-trigger-chain.md` §1 (the `trigger_chain` presence check);
  `skills/review/references/phase3-triage.md` §8 key lookup, including its
  `json_each(t.metadata, '$.x_finding_key')` snippet; `skills/review-all/SKILL.md`
  Phase 5 filing; and `skills/hotspot-survey/SKILL.md` refresh step 2. L1 re-greps for
  `Forensic reads of tasks.db` / `json_extract(metadata` / `json_each(t.metadata` across
  `skills/` and `docs/` at implementation time and removes any further fallback it finds.
  **Signal:** after the service restarts onto the landed code, an MCP call
  `find_tasks_by_metadata(project_root="/home/leo/src/dark-factory", key="files",
  value="scripts/merge_lane_metrics.py")` returns task 5414 with `status`, `priority`
  and `matched_value`, and `pagination.total` equals the count from the forensic query in
  `CLAUDE.md` §"Forensic reads of tasks.db" at the same moment; boundary rows 1–11 (and
  row 12, subject to open question 2) are green in the fused-memory suite. Consumers: the completion-triggers and
  census-incremental tasks (dependency wired at their decompose), and every instrument's
  §8 step 1.

### Capability bindings (draft for the decompose manifest)

| Capability | Evidence at `92af0716e8` |
|---|---|
| Per-project read guard | `sqlite_task_backend.py::SqliteTaskBackend._fresh_read_conn`, used by `SqliteTaskBackend.get_task` |
| Status index for the filter | `ix_tasks_status ON tasks (tag, status)` in `sqlite_task_backend.py::_SCHEMA_SQL` |
| JSON1 functions (`json_valid`, `json_type`, `json_extract`, `json_each`) | live store answered all four read-only; workspace venv sqlite 3.50.4 |
| Paging envelope + validation | `tools.py::_pagination_meta`, `tools.py::_validate_paging` |
| Tool plumbing | `tools.py::_normalize_project_root`, `tools.py::_log_read`, `fused-memory/src/fused_memory/server/tool_errors.py::mcp_tool_errors` |
| Pass-through precedent | `task_interceptor.py::TaskInterceptor.get_task` |
| Script transport | `scripts/legibility/census_trigger.py::post_mcp_tool_call` |
| Live list-valued key for the signal | `files` on 5,297 rows; task 5414 carries `scripts/merge_lane_metrics.py` |

G7 walk (advisory at author time): INV-11 `no-silent-fail-soft` — decision 6 and the
Contract's error rule; INV-13 `readers-prove-their-producer` — decision 7 (no `x_finding*` producer
has written yet, and the response says so); INV-9 `one-fact-one-home` — decisions 1–2;
INV-2 `structured-facts-at-failure` — validation errors name the value. No waiver.

## Out of scope

- Nested-path or range queries; full-text search over metadata (that is `search_tasks`).
- Any index or schema change (decision 3 states the revisit trigger).
- Writing `x_finding_*` keys — the instruments do that at filing (contract §8).
- A read-only fallback script: `CLAUDE.md` §"Forensic reads of tasks.db" already covers
  forensics.
- Adding the tool to any orchestrator role's `allowed_tools`; no dispatched role is a
  consumer.

## Open questions (tactical)

1. Test-file placement: a new `fused-memory/tests/test_find_tasks_by_metadata_tool.py`
   beside `test_search_tasks_tool.py` is suggested; decide in L1.
2. Boundary row 12 needs a live-server harness; if the fused-memory suite has none that
   reaches `post_mcp_tool_call`, the row moves to the consuming completion-triggers task
   and L1 keeps rows 1–11.
