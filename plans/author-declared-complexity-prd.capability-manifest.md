# Capability manifest — author-declared `complexity` field PRD

Mechanizes G3 + G6 per leaf. Each asserted capability is bound to evidence on
`main`. Any FAIL value blocks the batch. **All bindings PASS** — no novel
substrate, no numeric floors, no field-population, no grammar fixtures.

PRD: `plans/author-declared-complexity-prd.md`

## α — author-controlled simple-path routing (intermediate: unlocks β; also user-observable)

Signal: a `complexity='simple'` task (no blocker tokens) is routed to the
single-agent SIMPLE_TASK path (emits `phase_skipped` / `architect_skipped_simple_task`,
no architect plan phase); a fieldless task runs the full path; a declared-simple
task with a blocker token in its description runs the full path (veto).

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `phase_skipped` event with `reason='architect_skipped_simple_task'` is emitted on the simple path | capability→producer (wired) | `grep:orchestrator/src/orchestrator/workflow.py:2819-2835` — `EventType.phase_skipped` emitted with that exact reason inside `_run_simple_task` (production path) | PASS |
| single-agent SIMPLE_TASK path exists and is invoked from the gate | capability→producer (wired) | `_run_simple_task` `grep:workflow.py:2709`; `SIMPLE_TASK` role `grep:agents/roles.py:1049`; gate invokes it `grep:workflow.py:1522-1523` (α rewrites the trigger, keeps the invocation) | PASS |
| blocker-veto rejection mechanism exists and fires on blocker tokens | rejection-check (anti-silent-accept) | `_SIMPLE_TASK_HARD_BLOCKERS_RE` exists `grep:agents/triage.py:37-43`; α wires it as the gate veto; α's test authors a declared-simple+"migration" task and observes it take the FULL path (no `phase_skipped`) — rejection observed to fire | PASS |
| `metadata.complexity` is persisted and readable at the gate | capability→producer (wired) | metadata stored as JSON TEXT `grep:fused-memory/src/fused_memory/backends/sqlite_task_backend.py:64`; existing keys (`files`, `force_full_path`) already flow through the same path; gate reads `self.task['metadata']` `grep:workflow.py:1517-1518` | PASS |
| `simple_task_enabled` kill switch retained | capability→producer (wired) | `grep:orchestrator/src/orchestrator/config.py:1415` | PASS |
| DAG-direction (α upstream of β) | anti-inversion | α has no prereqs; β `depends_on` α | PASS |

## β — publish `complexity` rubric on the `submit_task` tool surface (leaf)

Signal: the live `submit_task` MCP tool description advertises the `complexity`
field + "when to declare simple" rubric.

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `submit_task` tool has an agent-facing description that can carry the field doc | capability→producer (wired) | `async def submit_task` with docstring/description surfaced as the MCP tool description `grep:fused-memory/src/fused_memory/server/tools.py:2421` | PASS |
| the `complexity` field it documents is honoured by a real consumer | capability→producer (upstream) | `producer:task-α` — α implements the gate that reads `complexity`; α is in β's transitive dependency closure and covers the exact extent (the routing trigger) | PASS |

**No FAIL bindings — batch clears the manifest gate.**
