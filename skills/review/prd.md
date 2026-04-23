# Deep Review Skill — PRD

## Overview

The `/review` skill performs deep, multi-phase code review of a project (or scoped subset), going beyond per-task verification to catch integration gaps, stubbed pipelines, broken wiring, and architectural drift. It is the primary mechanism for closing the systematic gap between "all tasks complete" and "software is actually finished enough to use."

The skill reads a **review briefing** (a versioned, project-specific file describing smoke tests, critical paths, and architectural invariants) and uses it alongside project memory to drive three review phases: integration verification, architectural coherence analysis, and triage/task creation.

## Motivation

The orchestrator's per-task verification is vertical — scoped to each task's own diff and tests. This leaves a class of issues that only surface when the assembled system is exercised end-to-end:

- Modules that pass their own tests but aren't wired to each other
- Rendering/processing pipelines with plausible-looking stubs that do nothing
- Integration points where types or protocols don't actually align
- Import chains that break at runtime despite passing static analysis
- Test suites that mock boundaries so thoroughly they never test real integration

These issues are consistently found during manual review, and consistently missed by per-task automation. This skill systematises that manual review.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Agent model | Opus (max effort) | Architectural reasoning requires deep context and nuanced judgment |
| Subordinate agents | Sonnet (high effort) | Phase 1 automated checks, stub scanning, test execution |
| Memory | fused-memory MCP | Project context, prior decisions, known invariants |
| Task management | fused-memory MCP (proxied Taskmaster) | Issue creation, task tree inspection |
| Test execution | Project-specific commands from review briefing | Smoke tests, integration tests, e2e tests |

## Inputs

### Review Briefing (required for full effectiveness, gracefully degraded without)

Location: `review/briefing.yaml` in the project root (or path specified in orchestrator config).

```yaml
# review/briefing.yaml
project: dark-factory
subprojects:
  fused-memory:
    root: fused-memory/
    smoke_tests:
      - name: "Server starts"
        command: "uv run --project fused-memory python -m fused_memory.server --help"
        expect: "exit 0"
      - name: "Health endpoint"
        command: "curl -s http://localhost:8002/health"
        expect: "json_field: status = ok"
    critical_paths:
      - name: "Memory write → read round-trip"
        trace:
          - "add_memory() → classifier routes to Mem0 or Graphiti"
          - "search() → retrieves the written memory"
        key_modules: [fused_memory/mcp_tools.py, fused_memory/classifier.py, fused_memory/mem0_client.py, fused_memory/graphiti_client.py]
    invariants:
      - "All MCP tools validate project_id is non-empty"
      - "Write operations always include agent_id attribution"
    known_gaps:
      - "Graphiti bulk import not implemented — tracked in task 112"

  orchestrator:
    root: orchestrator/
    smoke_tests:
      - name: "CLI loads"
        command: "uv run --project orchestrator orchestrator --help"
        expect: "exit 0"
    critical_paths:
      - name: "Task lifecycle"
        trace:
          - "PRD parse → task tree creation"
          - "Scheduler acquires task → workflow starts"
          - "PLAN → EXECUTE → VERIFY → REVIEW → MERGE"
        key_modules: [orchestrator/src/orchestrator/harness.py, orchestrator/src/orchestrator/workflow.py, orchestrator/src/orchestrator/scheduler.py]
    invariants:
      - "Terminal task states (done, cancelled) cannot be downgraded"
      - "Module locks prevent conflicting concurrent writes"
    known_gaps: []
```

If no briefing exists, the skill should:
1. Warn the user that review effectiveness will be limited
2. Suggest running `/review-briefing` first
3. Proceed with best-effort review using only code inspection and project memory

### Scope Parameter

```
/review                          # Full review of entire project
/review --scope fused-memory     # Scoped to one subproject
/review --focused <modules...>   # Focused review of specific modules (within-run mode)
/review --phase integration      # Run only Phase 1
/review --phase architecture     # Run only Phase 2
/review --phase triage           # Run only Phase 3 (on saved Phase 1+2 output)
```

## Review Phases

### Phase 1: Integration Verification

**Goal:** Answer "does the software actually run and do what it claims?"

**Process:**

1. **Load context**
   - Read the review briefing for the target scope
   - Search project memory for recent decisions, known issues, active conventions
   - If focused mode: identify which briefing sections are relevant to the changed modules

2. **Full test suite execution**
   - Run the complete test suite (not task-scoped) — unit, integration, e2e
   - Parse results, classify failures:
     - Pre-existing (known) vs new
     - Unit vs integration vs e2e
     - Flaky (check memory for known flakes) vs deterministic

3. **Smoke test execution**
   - Run each smoke test from the review briefing
   - Record pass/fail with stdout/stderr capture
   - For failures: attempt basic diagnosis (missing import, port conflict, missing config)

4. **Lint and type-check (full scope)**
   - Run linter and type-checker across the entire project (not task-scoped)
   - Classify results: new issues vs pre-existing baseline

5. **Compile output report**
   - Structured data: test results, smoke test results, lint/type issues
   - Each issue tagged with severity (blocking, warning, info) and affected modules
   - Save to `review/reports/phase1-<timestamp>.json` for Phase 3 consumption

**Agent allocation:** Sonnet agents for test execution and result parsing. Parallel execution where possible (e.g., tests and lint run concurrently).

### Phase 2: Architectural Coherence

**Goal:** Answer "is the codebase internally consistent and complete?"

This is the expensive, high-value phase. An Opus agent (or coordinated team) reads the codebase holistically.

**Process:**

1. **Stub and placeholder audit**
   - Scan for: `TODO`, `FIXME`, `HACK`, `NotImplementedError`, `pass` in non-trivial function bodies, hardcoded return values, `...` (Ellipsis) as implementation
   - Cross-reference each finding against:
     - The task tree: was this supposed to be implemented by a completed task?
     - Known gaps in the review briefing: is this intentionally deferred?
     - Project memory: is there a decision record explaining this?
   - Classify: **unintended stub** (task claims done but implementation is placeholder) vs **known gap** (documented) vs **acceptable** (e.g., abstract base class)

2. **Critical path tracing**
   - For each critical path in the review briefing, trace the actual code execution:
     - Does each stage call the next with real data?
     - Are there runtime dependencies (imports, config, env vars) that could fail?
     - Are types compatible at each boundary?
     - Is error handling present and consistent?
   - For paths without explicit briefing coverage: use heuristics to identify likely end-to-end flows (entry points → terminal operations) and trace those

3. **Cross-module consistency**
   - API surface: do public interfaces across modules use consistent naming, typing, error patterns?
   - Data flow: do data structures transform correctly across module boundaries?
   - Configuration: are config keys referenced in code actually defined in config files?
   - Imports: are there circular dependencies, missing transitive dependencies, or orphan modules?

4. **Dead code and orphan detection**
   - Modules/functions that exist but are never imported or called
   - Test files that don't test anything currently active
   - Config entries that nothing reads

5. **Test coverage analysis (qualitative)**
   - Not line-coverage metrics, but: are the critical paths tested end-to-end?
   - Are integration boundaries tested with real implementations (not mocks)?
   - Are there areas with only unit tests that need integration coverage?

6. **Compile output report**
   - Each finding with: location, severity, category, evidence, suggested fix
   - Save to `review/reports/phase2-<timestamp>.json`

**Agent allocation:** Primary Opus agent for architectural reasoning. May spawn Sonnet sub-agents for mechanical tasks (grep for stubs, map import graphs, enumerate dead code). The Opus agent synthesises findings.

### Phase 3: Triage and Task Creation

**Goal:** Convert review findings into actionable work items or escalations.

**Process:**

1. **Load Phase 1 + Phase 2 reports**

2. **Classify each finding:**

   | Classification | Criteria | Action |
   |---------------|----------|--------|
   | **Auto-fix** | Trivially fixable (missing import, type annotation, lint fix) | Create task, priority: medium |
   | **Clear-cut issue** | Unambiguous bug or gap with obvious fix path | Create task, priority: high |
   | **Design question** | Ambiguous, multiple valid approaches, or architectural implications | Escalate to user |
   | **Known/accepted** | Matches a known gap or explicit decision | Skip (log for audit trail) |
   | **Stale task** | Existing pending task addresses a now-irrelevant concern | Flag for user review |

3. **Task tree health check**
   - Review pending tasks against current state of main
   - Flag tasks whose assumptions have been invalidated by merged work
   - Flag tasks that duplicate review findings (avoid double-creation)
   - Flag tasks that are blocked on something that no longer exists

4. **Create tasks**
   - Use `submit_task` + `resolve_ticket` via fused-memory MCP (two-phase pattern): call `submit_task(...)` with the metadata below to receive a `ticket`, then call `resolve_ticket(ticket=ticket, project_root=...)` to block until the curator decides — `created` or `combined` → `task_id` is the new or merged task id; `failed` → if `reason` in {`server_restart`, `timeout`}, retry the `submit_task`+`resolve_ticket` pair once with the same metadata; if `reason` in {`unknown_ticket`, `server_closed`, `expired`}, record the reason in the review report and skip the finding. See `references/phase3-triage.md` for the full snippet.
   - The legacy `add_task` facade is deprecated and slated for removal; all new triage tasks must use `submit_task` + `resolve_ticket`
   - Each task tagged with `metadata.source: "review-cycle"` and `metadata.review_id: "<timestamp>"`
   - Include `memory_hints` pointing to the review findings and relevant briefing sections
   - Set dependencies appropriately (fix-up tasks may depend on each other)

5. **Escalate ambiguous findings**
   - Present to user as a structured summary:
     - Finding, evidence, options, recommendation
   - Wait for user decision before creating tasks for these

6. **Write review summary to memory**
   - `add_memory` with category `observations_and_summaries`
   - What was reviewed, what was found, what was created
   - Key metrics: issues found by severity, tasks created, areas of concern

**Agent allocation:** Opus for triage decisions (severity classification requires judgment). Sonnet for mechanical task creation.

## Output

### Interactive output (shown to user)

```
## Review Complete: fused-memory

### Phase 1: Integration Verification
- Test suite: 142 passed, 3 failed (2 new, 1 known flake)
- Smoke tests: 4/5 passed — FAILED: "Health endpoint" (server not running)
- Lint: 2 new issues (ruff E501, E712)
- Type-check: clean

### Phase 2: Architectural Coherence
- 3 unintended stubs found (tasks claimed done)
- 1 critical path broken: memory write round-trip — classifier returns but Mem0 client never called
- 2 orphan modules: fused_memory/legacy_router.py, fused_memory/old_schema.py
- Integration test gap: no tests exercise Graphiti → Mem0 cross-store search

### Phase 3: Triage
Created 5 tasks:
  - Task 120: Fix Mem0 client wiring in classifier (high)
  - Task 121: Remove orphan modules (medium)
  - Task 122: Add cross-store search integration test (high)
  - Task 123: Fix ruff E501 violations (low)
  - Task 124: Fix ruff E712 violation (low)

Escalated 2 findings for your review:
  1. Health endpoint assumes server is running — should smoke tests start services?
  2. Legacy router appears unused but has a comment "needed for migration" — safe to remove?
```

### Persistent output

- `review/reports/phase1-<timestamp>.json` — Phase 1 raw results
- `review/reports/phase2-<timestamp>.json` — Phase 2 raw results
- `review/reports/summary-<timestamp>.md` — Human-readable summary
- Memory entries via fused-memory
- Tasks via fused-memory (proxied Taskmaster)

## Configuration

Extends the orchestrator config with a `review` section:

```yaml
review:
  briefing_path: "review/briefing.yaml"    # Path to review briefing
  reports_dir: "review/reports"             # Where to save reports
  focused_review_interval: 5               # Trigger focused review every N merges
  models:
    integration: "sonnet"                   # Phase 1 agent
    architecture: "opus"                    # Phase 2 agent
    triage: "opus"                          # Phase 3 agent
  effort:
    integration: "high"
    architecture: "max"
    triage: "max"
  budgets:
    integration: 5.0
    architecture: 15.0
    triage: 5.0
```

## Skill Invocation Modes

| Mode | Trigger | Scope | Phases |
|------|---------|-------|--------|
| **Interactive full** | User runs `/review` | Entire project | All 3 |
| **Interactive scoped** | User runs `/review --scope X` | One subproject | All 3 |
| **Interactive phase** | User runs `/review --phase X` | As configured | Single phase |
| **Within-run focused** | Orchestrator triggers after N merges | Recently merged modules | All 3, narrowed scope |
| **Within-run final** | Orchestrator triggers at end of run | Entire project | All 3 |

The within-run modes are not part of this skill — they are orchestrator infrastructure that invokes the same review phases. The skill handles the interactive modes. The orchestrator integration is a separate piece of work (review checkpoint injection).

## Non-Goals

- **Line-level code coverage metrics** — we care about qualitative coverage of critical paths, not percentages
- **Style/formatting review** — linters handle this; the review focuses on correctness and architecture
- **Performance review** — not in scope for v1 (could be a future phase)
- **Security audit** — deserves its own dedicated skill with different expertise
- **Memory/knowledge audit** — handled by fused-memory reconciliation
- **Replacing per-task reviews** — this complements them, doesn't replace them

## Verification

### The skill works when:
1. Running `/review` on a project with a review briefing produces a structured report identifying real issues
2. Smoke test failures are detected and correctly diagnosed
3. Stubs are found and correctly classified (unintended vs known vs acceptable)
4. Critical path tracing catches broken wiring that per-task reviews missed
5. Created tasks are well-formed, have correct dependencies, and include memory hints
6. Escalated findings include enough context for the user to make a decision
7. Running `/review` without a briefing degrades gracefully with a suggestion to create one
8. Focused mode correctly narrows scope to the specified modules
9. Review findings are written to memory for future reference
10. The skill loads and uses project memory to avoid flagging known/accepted issues
