# Capability manifest — task-metadata-lookup-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized) for
`plans/task-metadata-lookup-prd.md`. Evidence verified on branch
`wip/quality-rulings-1005` @ `2bdfa20ce2` (main is an ancestor; none of the files cited
below differ from main), and against the live task store read-only (`mode=ro`) at
2026-10-05T16:43Z. Cited by `path::symbol`; no line anchors.
Machine-readable twin: `plans/task-metadata-lookup-prd.capability-manifest.yaml`.

One leaf, L1. It has no in-batch or out-of-batch prerequisite. Downstream consumers are
filed by other PRDs and depend on L1: `plans/completion-driven-triggers-prd.md` γ, δ, ε
and `plans/census-incremental-prd.md` L4.

## L1 — `find_tasks_by_metadata`: backend query, protocol, interceptor pass-through, MCP tool, docs

| Capability asserted by the signal or Contract | Evidence | Verdict |
|---|---|---|
| Per-project read guard for a new read method | `fused-memory/src/fused_memory/backends/sqlite_task_backend.py::SqliteTaskBackend._fresh_read_conn`, used on the production path by `SqliteTaskBackend.get_task` and `SqliteTaskBackend._get_tasks_internal` | PASS wired |
| Status index for the `statuses` filter | `CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status)` in `sqlite_task_backend.py::_SCHEMA_SQL` | PASS wired |
| JSON1 (`json_valid`, `json_type`, `json_extract`, `json_each`) | The live store ran a membership query using all four, read-only, 2026-10-05. The workspace venv has sqlite 3.50.4 and the system python 3.45.1 | PASS |
| Backend method on the store owner | NEW, produced by L1: `SqliteTaskBackend.find_tasks_by_metadata`. Absent on main today | PASS producer:L1 |
| Protocol declaration | NEW, produced by L1, on `fused-memory/src/fused_memory/backends/task_backend_protocol.py::TaskBackendProtocol`. `SqliteTaskBackend` is its only production implementer (grep over `fused-memory/src`) | PASS producer:L1 |
| Interceptor pass-through (in-process consumer: completion-triggers γ) | NEW, produced by L1. Precedent: `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor.get_task` in the "Pure reads (direct pass-through)" section | PASS producer:L1 |
| MCP tool (consumers: completion-triggers δ/ε, census L4, contract §8 step 1) | NEW, produced by L1, in `fused-memory/src/fused_memory/server/tools.py`. Plumbing exists: `tools.py::_normalize_project_root`, `tools.py::_log_read`, `fused-memory/src/fused_memory/server/tool_errors.py::mcp_tool_errors`. The `FUSED_MEMORY_INSTRUCTIONS` "Task operations" block exists | PASS producer:L1 |
| Paging envelope and validation | `tools.py::_pagination_meta` (five keys, `has_more` derived), `tools.py::_validate_paging` (rejects bool, ≤0 and negative offset). The limit clamp mirrors `tools.py::search_tasks` (reject ≤0, cap at 100) | PASS wired |
| Bare-string `statuses` rejected | `tools.py::get_tasks` rejects a bare string with `'statuses must be a list of status strings'` | PASS wired |
| A store error reaches the caller as an error payload, never as `results: []` (INV-11) | `tool_errors.py::mcp_tool_errors` converts a raised store error into the error payload. The backend method must raise, not return empty. L1's tests bind this | PASS mechanism-with-precedent |
| Rejection of `value=True` and of a malformed `key` (G6 branch 4) | Built by L1 and bound by its boundary rows 5–6. Precedent that fires today: `_validate_paging` rejects `page_size=True` with a `ValidationError` | PASS mechanism-with-precedent |
| Live producer for the signal's key (INV-13) | `files` is written by `submit_task` in production on 5,356 rows. Task 5021 has been `done` since 2026-09-11 and its `files` contains `scripts/merge_lane_metrics.py`. A read-only membership query at 2026-10-05T16:43Z returned 14 ids, 5021 among them. The first exemplar, task 5414, was wrong: it was open when chosen and landed with its `files` rewritten (this decompose re-measured it and amended the PRD). The signal asserts membership and same-moment equality, not a count | PASS |
| No `x_finding*` or `trigger_chain` producer has written yet, and the reader says so | 0 rows carry either key, measured 2026-10-05. Decision 7's `tasks_with_key == 0` is the "no producer" state (INV-13) | PASS |
| Page fits the ~62 KB MCP envelope (decision 6, as amended 2026-10-05) | Default `include_value=False`, measured read-only 2026-10-05T16:57Z on 6,340 rows: worst row 2,036 B, worst 100 rows 40,798 B, so the `limit` cap of 100 bounds the page. With `include_value=True`, `RESULT_BYTE_BUDGET = 48,000` B bounds `results`. The largest single live row over any top-level key is 44,375 B, so none is elided today. The elided path is reachable and pinned by boundary row 14. The wall is the `tools.py` comment on the `get_statuses` cap (task 3064): the transport rejects past ~62–80 KB | PASS floor: 40,798 < 62,000; 48,000 + envelope < 62,000 |
| Opt-in value and byte budget (`include_value`, `truncated_value`, `{"_elided": true, "bytes": N}`) | NEW, produced by L1 and bound by boundary rows 13–15. Completion-triggers γ and ε must pass `include_value=True` (seam row) | PASS producer:L1 |
| Interim forensic fallbacks removed | `skills/review/references/phase3-triage.md` (the `json_each(t.metadata, '$.x_finding_key')` snippet, and the §1 `json_extract(metadata, '$.trigger_chain.skill')` lookup line), `skills/review-all/SKILL.md` Phase 5, `skills/_shared/filing-the-trigger-chain.md` §1, `docs/quality-findings-contract.md` §8 step 1. `skills/hotspot-survey/SKILL.md` refresh step 2 already defers to contract §8 step 1 and has no fallback text, so it needs no edit unless one appears. Non-metadata forensic reads stay: the deferred listing in `skills/review-briefing/SKILL.md` and corruption detection in `skills/escalation-watcher/SKILL.md` | PASS (N→0 at L1) |
| Boundary row 12 (consumer face over `post_mcp_tool_call` against a live server) | Transport exists: `scripts/legibility/census_trigger.py::post_mcp_tool_call`. No fused-memory test reaches it today (grep over `fused-memory/tests`). Open question 2 hands the row to completion-triggers λ when there is no harness, and that PRD's seam table accepts | OPEN (decided in L1) |

### Resolved at decompose

The envelope binding first FAILED. As authored, decision 6 bounded the worst 100 rows
at ≤ 51 KB, but measured that without `matched_value`, which the Contract then returned
whole. With the value included, the measurements on 2026-10-05 were:

- presence on `files`: worst 50 = 113,201 B, worst 100 = 184,760 B
- `delivered_checks`: worst 100 = 118,614 B
- `dry_run_proposals`: one row up to 41,415 B; worst 50 = 1,096,640 B

The lead ruled on 2026-10-05 to adopt both opt-in `include_value` and a 48,000 B
result budget with lone-row elision. The PRD's decisions 4 and 6, the Contract and
boundary rows 13–15 were amended to match, and the binding now passes on the basis
above.

No FAIL, `declared-only`, `test-only`, `producer-downstream`, `producer-absent`,
`producer-extent-short`, `fixture-ERROR` or `rejection-absent` binding.
