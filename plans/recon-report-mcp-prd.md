# PRD — Recon agent diagnostic-output channel via dedicated MCP

**Status:** active (greenfield design), 2026-05-28
**Slug:** `recon-report-mcp`
**Approach:** B + H (contract + two-way boundary-tests). See § Contract and § Boundary-test sketch.

## 1. Goal

The fused-memory reconciliation Stage 1 / Stage 2 / Stage 3 LLM agents produce diagnostic reports through a **dedicated MCP namespace** (`recon_report.*`) instead of free-form structured JSON output. Each citation — entity, edge, task, memory — is validated at tool-call time against the live system; the LLM never types or recalls a UUID into the report. The structural property that the report channel enforces eliminates four observed failure modes that prompt-only discipline could not.

User-observable surface, in one sentence: across a 24h soak, the recon-escalation-watcher's close-manifests show **zero** typo'd entity UUIDs, **zero** malformed `affected_ids` entries, **zero** count-snapshot citations, and **zero** cross-project task-id collisions emitted by the recon stages.

## 2. Background — Issue #8 evidence

Recon-escalation-watcher session 2026-05-28 documented (`recon-watch/close-manifest-3.json`):

1. **Typo'd entity UUIDs.** A finding cited entity `371b46ea-...-7e79f6...`; the real entity is `...7e79d6...` (one hex-digit drift, classic LLM hallucination). The agent's follow-up `refresh_entity_summary` returned `NodeNotFoundError`, which the agent then mis-diagnosed as "the entity may not exist."
2. **Truncated edge UUIDs in `affected_ids`.** Sentinels like `96cddd4d-edge` (not a 36-char UUID). `update_edge` returned `EdgeNotFoundError`.
3. **Contradictory count snapshots within a single run.** One finding said `1505 done`, another `3355 done`, another `3358 done` for the same project at the same moment — the agent was quoting historical entries from a polluted entity summary as if they were current facts. The pollution itself is artefact of an upstream truncation bug (df:1516, DONE) but the legacy edges still exist and the LLM still reads them.
4. **Cross-project task-id collision.** "Bootstrap monorepo" (know-live task 1) and the current project's "Wire bundle adapters" (also task 1) were treated by the agent as colliding. Bare-integer `task_id` strings in `affected_ids` carry no project context.
5. **(Out of scope)** Harmful `submit_task` recommendations on existing tasks. The task curator now has semantic-similarity dedup; no LLM-side pre-check is needed.

### 2.1 Why prompt-only discipline failed

Stage 1's `## UUID Resolution Discipline` (prompts/stage1.py:68–81) requires verbatim UUIDs **for `delete_memory` tool calls** but does not extend to the structured `flagged_items` report channel. `STAGE_REPORT_SCHEMA` (cli_stage_runner.py:60–122) accepts any string in `affected_ids`. `_normalize_report` (cli_stage_runner.py:229–262) has no UUID format check. The Snapshot Discipline section (prompts/stage1.py:200–238) forbids new count-snapshot writes but neither cleans up legacy text nor strips it from the payload before the LLM reads it. The LLM treats prior-turn text as substitutable for fresh tool output — UUIDs and counts get quoted from working memory, not from the structured field of the most recent tool response.

### 2.2 Precedent — architect plan-MCP

The same failure class was solved for the architect's `plan.json` output by moving from free-form structured output to a dedicated MCP whose tools build the plan incrementally (create / add steps / stamp complete). Each tool validates its input at call time; the LLM cannot smuggle a malformed step into the final artefact. This PRD applies the same shape to the recon report channel.

## 3. Non-goals and constraints

- **Do NOT touch task-curator semantic dedup.** Already shipped; LLM-side absence pre-check before `submit_task` is **out of scope**.
- **Do NOT weaken Stage 3's read-only constraint.** `recon_report.*` is exempt from `DISALLOW_MEMORY_WRITES` but writes to its own in-process state only, not to Graphiti/Mem0/Taskmaster.
- **Degrade gracefully.** A stage agent that fails to emit a terminal `recon_report.complete()` produces an empty report — same outcome as today's malformed-JSON path.
- **No data-format break** for downstream consumers. `StageReport.items_flagged` and `StageReport.stats` retain their existing field names; only the producer side changes.

## 4. Sketch of approach

### 4.1 The dedicated MCP endpoint

A second `FastMCP` instance (`Recon Report`) is constructed inside the existing fused-memory server process, bound to a separate HTTP port (default `8003`). It serves tools only — same uvicorn shape as the primary endpoint at `streamable_http_app()`, gathered alongside the primary `uvicorn.Server` via `asyncio.gather`. Recon stages get **only** the recon_report endpoint added to their `_build_mcp_config` (stages/base.py); the primary fused-memory endpoint's tool surface is unchanged for all other clients.

The recon_report server holds in-process state keyed by reconciliation `run_id`: each stage's in-progress findings, stats, and summary live in a dict that the tools manipulate. The terminal `complete()` tool finalises the report; `cli_stage_runner._extract_report` reads the assembled report from server state via an internal accessor (in-process function call, not an MCP tool — the runner is in the same Python process).

### 4.2 Validation discipline

Every `cite_*` tool resolves or validates its argument against the live system by calling the existing memory/task services in-process (no network hop):

- `cite_entity(name)` → server calls `memory_service.get_entity(name, project_id)`; resolves to the canonical 36-char `entity_uuid`; attaches uuid + canonical name to the finding. The LLM passes a name, never a UUID.
- `cite_edge(edge_uuid)` → server validates 36-char shape, looks up the edge, attaches uuid + fact-text snapshot. Sentinel strings like `96cddd4d-edge` rejected with structured error.
- `cite_task(project_id, task_id)` → both fields required; server validates the (project, id) pair via `get_task`; rejects bare ints, validates project_id against `known_projects`, attaches task title.
- `cite_memory(memory_id, store)` → validates 36-char shape, looks up memory in the named store, attaches metadata fingerprint.

Validation failures return a structured error response that the LLM sees on its next turn. The LLM has the opportunity to self-correct (look up the entity by name, copy the freshly-returned UUID). Persistent failure simply means the finding never includes that citation — the report is empty of bad data, not corrupted by it.

### 4.3 Stage-prompt rewrite

Stage 1/2/3 system prompts replace the "produce your final structured JSON report" instruction with "build the report via `recon_report.*` tools and call `recon_report.complete(summary)` when done." `AgentLoop.terminal_tool` (already configurable, defaults to `stage_complete`) flips to `recon_report.complete`. The existing prompt sections that govern memory/task mutations (UUID Resolution Discipline, Terminal-State Pre-Check Discipline, etc.) are unchanged — they govern the **action** channel, which still uses the primary fused-memory tools. Only the **report** channel moves to recon_report.

### 4.4 Legacy-snapshot defence-in-depth

Independent of the channel change, the data-layer hazard that produced contradictory counts (legacy count-snapshot temporal_facts edges) is closed with three measures:

- **Write-gate** in `add_memory(category='temporal_facts')`: reject content matching count-snapshot patterns when the writer's `agent_id` matches `recon-stage-*`. Defence-in-depth backing Stage 1's existing Snapshot Discipline prompt rule.
- **Payload-side filter** in `MemoryConsolidator.assemble_payload`: strip lines matching the count-snapshot pattern from injected entity summaries before sending to the LLM. Reports the strip count via `stats.entity_summary_snapshot_lines_stripped`.
- **One-time cleanup** script: scan every entity summary across every known project in Graphiti for count-snapshot patterns, invalidate matching edges, emit per-edge `observations_and_summaries` audit entries. Sweep scope is **all entities / all projects** — the polluted `reify project` entity is the documented case, but the same pattern may exist elsewhere.

## 5. Resolved design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Internal to fused-memory process, separate HTTP port.** | Same Python process → cite_* tools call memory/task services in-process (no hop, no auth concerns). Separate port → primary fused-memory tool surface stays untouched for general clients. Two FastMCP instances + two uvicorn.Server runs is a standard ASGI pattern. |
| D2 | **In-process state keyed by reconciliation run_id.** | Run_id is already threaded through every stage; cli_stage_runner can read assembled state by run_id at terminal. Concurrent stages (across projects) don't collide. Process restart blow-away matches existing fused-memory restart blast radius (recon cycle aborts; not a new failure mode). |
| D3 | **LLM passes entity names, server resolves to UUIDs.** | The structural fix for Issue #1. The LLM has no UUID typing surface. The architect plan-MCP precedent follows the same pattern (steps are added by content, plan ids generated server-side). |
| D4 | **`cite_task` requires both `project_id` and `task_id` as separate parameters.** | Bare task-id strings are structurally impossible. Cross-project collision (Issue #4) cannot occur. |
| D5 | **`affected_ids` retired; replaced by typed citation lists.** | The assembled report exposes `cited_entities`, `cited_edges`, `cited_tasks`, `cited_memories` to downstream consumers. Existing `items_flagged` consumers (judge.py, stats_verifier.py, flag_dedup.py) receive findings whose typed citation fields replace the opaque-string list. |
| D6 | **In-run signature dedup at `add_finding` call time.** | Same `(task_id, flag_type)` pair within a single stage rejected with a structured error. Cross-cycle dedup post-processor in `flag_dedup.py` stays untouched. |
| D7 | **Cleanup script sweeps all entities across all projects.** | The reify pollution is documented but the same pattern may exist on other entities. Wider sweep is safer than guessing. Per-edge audit observations provide rollback evidence. |
| D8 | **Write-gate fires on `agent_id` match (`recon-stage-*`), not on global content match.** | A genuine non-recon write that happens to mention "done" / "cancelled" / "total" is not the failure mode. The Snapshot Discipline rule is recon-specific; the gate scope matches. |

## 6. Pre-conditions for activating

- df:1516 is DONE (commit `b489bd72`) — the upstream truncation bug is fixed. Without it, even a correct report channel would still receive bad counts from a truncated tree.
- FastMCP, uvicorn, Starlette versions in `fused-memory/pyproject.toml` already support multiple parallel ASGI apps in one process. No substrate work needed.
- `AgentLoop.terminal_tool` (agent_loop.py:80) already supports terminal-tool selection. No structural change to the agent loop.
- `cli_stage_runner._extract_report` is the only consumer of the report blob today; no other readers need migrating.

No external prerequisites.

## 7. Cross-PRD relationship

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| df:1516 (DONE) | upstream context | Stage 1 task-tree truncation fix + Stage 2 absence guard | df:1516 | done — no active seam |

No cross-PRD seams. This PRD is internally complete to the fused-memory package.

## 8. Out of scope for this PRD

- LLM-side absence pre-check before `submit_task` (task curator semantic dedup already handles this).
- Migration of the orchestrator's task-pipeline escalation channel (port 8100/8102) to a similar tool-driven shape — separate concern, separate decision.
- Migration of the Stage 3 finding schema beyond what the cite_* tools require — extensions to `FINDING_ITEM_SCHEMA` enums stay in the same place they live today (cli_stage_runner.py:80–122).
- The judge's stat-verification logic (`stats_verifier.py`) — unchanged except where the new typed citation shape is mechanically substituted for `affected_ids` reads.

## 9. § Contract (B+H)

### 9.1 Recon-report MCP endpoint

- **Process:** fused-memory server (single Python process).
- **Transport:** HTTP, separate uvicorn.Server bound to `config.server.recon_report_port` (default `8003`, configurable).
- **Discovery:** stages/base.py `_build_mcp_config` injects `{'recon-report': {'type': 'http', 'url': 'http://127.0.0.1:8003/mcp/'}}` into the per-stage MCP config alongside `fused-memory` and `escalation`.
- **Allow-list:** `recon_report.*` tools are exempt from `DISALLOW_MEMORY_WRITES` and `DISALLOW_TASK_WRITES` in `cli_stage_runner.py`. Stage 3's read-only contract is preserved because recon_report writes only to in-process state, not to memory/task stores.

### 9.2 Tool signatures

All tools require `run_id: str` to scope the call to the current reconciliation cycle. Mismatch returns `{"error": "run_id_unknown", "error_type": "ReconReportRunUnknown"}`.

```
recon_report.start_report(run_id: str, stage: str, project_id: str) -> {"ok": True}
    Idempotent. Called implicitly by cli_stage_runner before the stage agent
    starts; the agent itself does not need to invoke this. State scaffold for
    findings / stats / summary is allocated for (run_id, stage).

recon_report.add_finding(
    run_id: str,
    severity: Literal["minor", "moderate", "serious"],
    category: str,        # validated against STAGE3 enum when stage == "integrity_check"
    description: str,
    suggested_action: str,
    task_id: str | None = None,      # optional; for in-run dedup signature
    flag_type: str | None = None,    # optional; for in-run dedup signature
    actionable: bool = True,
) -> {"finding_id": str}             # 36-char UUID assigned server-side

    Allocates a new finding row. If (task_id, flag_type) matches a prior
    finding in this (run_id, stage), returns
    {"error": "duplicate_finding", "error_type": "ReconReportDuplicateFinding",
     "existing_finding_id": "..."} so the LLM can attach citations to the
    existing finding instead.

recon_report.cite_entity(run_id: str, finding_id: str, name: str)
    -> {"entity_uuid": "<36-char>", "canonical_name": "..."}

    Resolves name → uuid via memory_service.get_entity(name, project_id).
    Rejects when the entity is not found (NodeNotFound). The LLM never types
    a UUID. Multiple cite_entity calls on the same finding accumulate.

recon_report.cite_edge(run_id: str, finding_id: str, edge_uuid: str)
    -> {"edge_uuid": "<36-char>", "fact_text_snapshot": "..."}

    Validates the 36-char shape (regex enforced), looks up the edge via
    memory_service.get_edge(edge_uuid). Rejects truncated sentinels like
    "96cddd4d-edge". Returns the fact-text at lookup time as a snapshot.

recon_report.cite_task(
    run_id: str, finding_id: str, project_id: str, task_id: str,
) -> {"project_id": "...", "task_id": "...", "title": "..."}

    Both project_id and task_id REQUIRED. Validates project_id against
    known_projects; validates task existence via task_interceptor.get_task.
    Rejects bare integers without project_id. The dual-required shape makes
    cross-project collision (Issue #4) structurally impossible.

recon_report.cite_memory(
    run_id: str, finding_id: str, memory_id: str, store: Literal["graphiti", "mem0"],
) -> {"memory_id": "<36-char>", "metadata_fingerprint": {...}}

    Validates 36-char shape, looks up memory in the named store. Rejects
    8-char hex prefixes and missing memories.

recon_report.set_stat(run_id: str, key: str, value: int | float | str) -> {"ok": True}
recon_report.inc_stat(run_id: str, key: str, by: int = 1) -> {"value": int}
    Typed counter manipulation. The judge's stat-verification logic reads
    these by exact name.

recon_report.complete(run_id: str, summary: str) -> {"flagged_count": int, "stats": {...}}
    Terminal tool. Stamps the report as complete. cli_stage_runner reads
    the assembled state from its internal accessor immediately after.
    Idempotent within a run_id: a second complete() with the same summary
    is a no-op; a different summary appends a warning to the report but
    does not overwrite.
```

### 9.3 Assembled report shape (read by cli_stage_runner)

```python
{
    "summary": str,
    "stats": dict[str, int | float | str],
    "flagged_items": [
        {
            "finding_id": str,           # server-assigned UUID
            "severity": str,
            "category": str,
            "description": str,
            "suggested_action": str,
            "actionable": bool,
            "task_id": str | None,
            "flag_type": str | None,
            "cited_entities": [{"entity_uuid": str, "canonical_name": str}, ...],
            "cited_edges":    [{"edge_uuid": str, "fact_text_snapshot": str}, ...],
            "cited_tasks":    [{"project_id": str, "task_id": str, "title": str}, ...],
            "cited_memories": [{"memory_id": str, "store": str, "metadata_fingerprint": dict}, ...],
        },
        ...
    ],
}
```

The existing `StageReport.items_flagged: list[dict]` field shape is preserved (each item is still a dict); the keys are the new typed-citation set instead of `affected_ids`. Downstream consumers (judge.py, flag_dedup.py, stats_verifier.py) read the citation lists by name.

### 9.4 State lifecycle

- **start_report** is invoked by `cli_stage_runner.run_stage_via_cli` before the agent subprocess launches. The scaffold is keyed by `(run_id, stage)`.
- **Validity window:** state persists for `config.reconciliation.recon_report_state_ttl_seconds` after `complete()` (default `300` seconds — long enough for cli_stage_runner to read).
- **Reaper:** a periodic asyncio task in the fused-memory server sweeps expired state every 60s. Bounded memory.
- **Process restart:** all in-progress state is lost. The orphaned stage's cli_stage_runner read returns an empty report — same outcome today as a malformed-JSON parse failure.

### 9.5 In-run dedup semantics

`add_finding` computes a signature from `(task_id, flag_type)`. If a prior finding in this `(run_id, stage)` has the same signature, the call returns an error pointing the LLM at the existing `finding_id` so citations can attach to it instead of creating a duplicate. The post-processor cross-cycle dedup (`flag_dedup.py`) continues to fold same-signature flags **across** runs; the in-run gate handles within-run duplicates that the post-processor would have folded anyway.

## 10. § Boundary-test sketch (B+H)

Two-way: producer side (LLM-as-recon-stage exercising the cite_* tools) and consumer side (cli_stage_runner consuming the assembled report).

### 10.1 Producer-side scenarios (LLM-facing)

| Scenario | Precondition | Postcondition |
|---|---|---|
| **P1: clean cite cycle** | Live entity `foo` exists in Graphiti for project_id `bar`. | `cite_entity(name="foo")` returns canonical uuid + name. Subsequent `complete()` produces a report whose finding has `cited_entities: [{entity_uuid: <real>, canonical_name: "foo"}]`. |
| **P2: typo'd-uuid impossible** | Stage agent's prompt forbids UUID typing. | The agent has no tool that accepts an entity UUID parameter. Any attempt to call `cite_edge` with a non-36-char string returns a structured rejection. No path exists by which a typo'd UUID enters the report. |
| **P3: truncated edge sentinel rejected** | Agent attempts `cite_edge(edge_uuid="96cddd4d-edge")`. | Tool returns `{"error": "invalid_uuid_shape", "error_type": "ReconReportInvalidUuid"}`; no edge attached; finding's `cited_edges` is unaffected. Agent's next turn sees the error and either looks up the real UUID or moves on. |
| **P4: bare task_id rejected** | Agent attempts `cite_task(project_id="dark_factory", task_id="")`. | Validation error. Bare ints without `project_id` are structurally impossible: the tool's parameter list requires both. |
| **P5: cross-project collision impossible** | Agent works on dark_factory; sees know-live task 1 in a search result. | `cite_task` requires the explicit project_id; the agent must declare which project task 1 belongs to. Two findings citing task 1 from different projects produce two distinct cited_tasks entries with different `project_id` values. |
| **P6: in-run dedup** | Agent calls `add_finding(task_id="42", flag_type="orphaned_knowledge")` twice in the same stage. | Second call returns `{"error": "duplicate_finding", "existing_finding_id": "<id>"}`. Agent may attach further citations to the existing finding. |
| **P7: terminal idempotence** | Agent calls `complete(summary="...")`, then calls it again with the same summary. | Second call returns the same response; report is unchanged. |
| **P8: graceful empty report** | Agent crashes / times out before calling `complete()`. | cli_stage_runner's read sees an in-progress (not completed) state; `_extract_report` falls back to the "empty stage report" shape, same as today's parse-failure path. The stage is logged as failed; no malformed report enters StageReport. |

### 10.2 Consumer-side scenarios (cli_stage_runner / downstream)

| Scenario | Precondition | Postcondition |
|---|---|---|
| **C1: read assembled report** | Stage agent has called `complete()` for `(run_id="r1", stage="memory_consolidator")`. | `cli_stage_runner._extract_report(run_id, stage)` returns the assembled dict matching § 9.3 shape. |
| **C2: typed citations downstream** | Report contains a finding with `cited_tasks: [{project_id: "x", task_id: "5"}]`. | `flag_dedup.dedup_flags` reads the task_id from `cited_tasks[0].task_id`, computes signature `(x, 5, flag_type)`, dedups correctly. |
| **C3: judge stat read** | Stage emitted `inc_stat("findings_added")` 3 times. | `verify_and_rewrite_stats` reads `stats["findings_added"] == 3` from the assembled report. |
| **C4: state isolation across runs** | Two concurrent stages, run_ids `r1` and `r2`. | Each stage's calls scoped by `run_id`; cli_stage_runner reads only the matching run_id's state. No cross-run leakage. |
| **C5: run_id mismatch rejected** | Stage agent (somehow) calls a tool with a stale `run_id` from a prior cycle. | Tool returns `{"error": "run_id_unknown"}`. The current run's report is unaffected. |
| **C6: state TTL** | Stage completed 6 minutes ago, TTL = 300s. | Reaper has evicted the state; a late read returns `None` and cli_stage_runner falls back to the empty-report path. Acceptable; cli_stage_runner reads immediately after `complete()` in normal operation. |
| **C7: process restart mid-cycle** | fused-memory restarts after Stage 1 emitted findings but before `complete()`. | State is lost. cli_stage_runner's read returns empty. The stage is logged as failed (same outcome as malformed JSON today). |
| **C8: 24h soak** | Full reconciliation pipeline runs continuously for 24h with the new channel active. | recon-watch close-manifests over the 24h window show: zero typo'd UUIDs, zero malformed `affected_ids`, zero count-snapshot citations, zero cross-project task-id collisions. |

The **integration-gate task** (decomposition § 11 task **γ**) names C8 as its observable signal.

## 11. Decomposition plan

| Label | Title | Modules touched | Observable signal | Prereqs |
|---|---|---|---|---|
| **α** | `recon_report` MCP scaffold + state + add_finding/complete | `fused-memory/src/fused_memory/server/` (new module `recon_report.py`), `fused-memory/src/fused_memory/server/main.py` (second uvicorn.Server boot), `fused-memory/src/fused_memory/config/schema.py` (`server.recon_report_port`, `reconciliation.recon_report_state_ttl_seconds`) | Manual probe via `curl` to `http://127.0.0.1:8003/mcp/`: `start_report` → `add_finding` → `complete` → cli_stage_runner accessor returns the assembled report dict matching § 9.3 shape. Unit-tested in `fused-memory/tests/server/test_recon_report.py`. | — |
| **β** | `recon_report` cite_* validation tools | `fused-memory/src/fused_memory/server/recon_report.py` (extends α) | Unit tests in `tests/server/test_recon_report_citations.py` covering P3 (cite_edge sentinel rejection), P4 (cite_task bare-id rejection), P5 (cross-project collision), P1 (clean cycle name→uuid resolution). | α |
| **γ** | Cutover: Stage 1/2/3 prompts + `cli_stage_runner` + downstream schema consumers + 24h soak | `fused-memory/src/fused_memory/reconciliation/prompts/stage{1,2,3}.py`, `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py`, `fused-memory/src/fused_memory/reconciliation/agent_loop.py` (terminal_tool default), `fused-memory/src/fused_memory/reconciliation/stages/base.py` (`_build_mcp_config`), `fused-memory/src/fused_memory/reconciliation/judge.py`, `fused-memory/src/fused_memory/reconciliation/flag_dedup.py`, `fused-memory/src/fused_memory/reconciliation/stats_verifier.py` | Boundary-test sketch § 10.2 C8: 24h soak of the live reconciliation pipeline shows the four metrics zeroed out, observable in recon-watch close-manifests. Integration test in CI exercises one full Stage 1→2→3 cycle end-to-end. | α, β |
| **δ** | Count-snapshot write-gate + payload-side filter | `fused-memory/src/fused_memory/server/tools.py` (`add_memory` gate), `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py` (`assemble_payload` line filter), `fused-memory/src/fused_memory/reconciliation/task_filter.py` (pattern shared constant) | Unit tests: `add_memory(category="temporal_facts", content="...3355 done, 290 cancelled, 3358 total...", agent_id="recon-stage-task_knowledge_sync")` returns rejection; `assemble_payload` strips matching lines from entity summaries; `stats.entity_summary_snapshot_lines_stripped` reports the strip count. | — |
| **ε** | One-time legacy count-snapshot cleanup across all entities / all projects | `fused-memory/scripts/cleanup_count_snapshots.py` (new), audit observations in Mem0 | Script run produces a per-edge audit report (entities scanned, edges invalidated, projects covered); post-run `search(query="1505 done")` returns zero hits in entity summaries across all known projects; rollback evidence preserved in `observations_and_summaries`. | δ |

Five tasks total. α/β/γ form the report-channel chain (R1+R2 from the brief); δ/ε form the data-layer hardening (B). γ is the integration-gate task per the B+H pattern.

## 12. Open questions (tactical, surfaced but not decided)

1. **Default port number for recon_report endpoint.** PRD assumes `8003`; final number to be confirmed against any existing assignments in `config/config.yaml`. Decide during α.
2. **`recon_report` state TTL.** PRD suggests 300s; calibrate against observed Stage 3 → next-stage handoff latency. Decide during γ.
3. **Cleanup-script execution model.** PRD lands the script under `scripts/`; whether to run it once as an operator action or wire into a systemd timer (one-shot) is an operational call. Decide during ε.
4. **In-run dedup error vs warning.** PRD specifies the duplicate-finding call returns an error pointing at the existing `finding_id`. An alternative is a warning that still allocates the second finding. The error shape is stricter and forces the agent to use the citation-attachment path. Decide during β.
5. **Stage-2 `cross_project_findings` representation.** Today Stage 2's prompt instructs the LLM to add a `cross_project_findings` top-level entry (with `summary`, `target_project_hint`, `evidence`) when a finding's scope belongs to an unknown project (prompts/stage2.py:77). No code reads it back today; under recon_report it should land either as a typed `recon_report.add_cross_project_finding` tool or as a category-coded finding in `flagged_items`. Decide during γ. **Suggested resolution:** dedicated tool, mirror tool-driven discipline.
6. **Cleanup script dry-run mode.** ε is large-blast-radius (all entities / all projects); a `--dry-run` mode that prints the per-edge audit report without invalidating anything should be the default invocation, with `--apply` required for the destructive path. Decide during ε. **Suggested resolution:** dry-run as default, `--apply` required.

## 13. Notes on coordination with df:1516

df:1516 (DONE, commit `b489bd72`) fixed the Stage 1 task-tree truncation that was the **source** of the bad count snapshots, and added a guard in Stage 2 against deleting knowledge for a task it cannot positively confirm absent. This PRD addresses the **legacy data** that df:1516 deliberately left in place, plus the **report channel** that df:1516 did not touch. The two fixes are orthogonal; this PRD is the natural follow-on.

## 14. References

- Live evidence: `recon-watch/close-manifest-3.json` (entries `esc-recon-a29272d5-23..27`).
- Upstream fix: df task 1516 (commit `b489bd72ab7c983ab36d3a160e64a18a1385066f`).
- Existing prompts: `fused-memory/src/fused_memory/reconciliation/prompts/stage{1,2,3}.py`.
- Existing schema: `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py:60–122`.
- Existing agent loop: `fused-memory/src/fused_memory/reconciliation/agent_loop.py:80` (`terminal_tool`).
- Precedent for tool-driven artefact construction: the architect plan-MCP pattern.
