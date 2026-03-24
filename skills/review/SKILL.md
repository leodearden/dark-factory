---
name: review
description: "Deep, multi-phase code review — runs tests/lint/typecheck, audits architecture for stubs and broken wiring, then triages findings into tasks. ALWAYS use this skill for: /review commands, requests to verify software actually works end-to-end, post-orchestrator quality checks ('the tasks are done but does it actually work?'), finding stubs/NotImplementedError/placeholders, checking integration health across modules, finding dead code or orphan modules, auditing cross-module consistency after merges, running the full test suite with analysis (not just 'run pytest'), and qualitative test coverage analysis ('are the tests just mocking everything?'). This is NOT for: single-file PR reviews, creating review briefings (/review-briefing), unblocking tasks, explaining code, fixing lint, running tests without analysis, or implementing tasks. When in doubt about whether the user wants a deep project-wide review vs a simpler action, USE this skill — it handles scoping internally."
---

# Deep Review

Three-phase review that answers: does the software actually run, is it internally consistent, and what needs fixing?

- **Phase 1 — Integration Verification**: run tests, smoke tests, lint, typecheck. Mechanical, parallelisable.
- **Phase 2 — Architectural Coherence**: trace critical paths, find stubs, check cross-module consistency. Deep reasoning.
- **Phase 3 — Triage**: classify findings, create tasks, escalate ambiguities. Judgment calls.

The review is driven by a **review briefing** (`review/briefing.yaml`) — a project-specific file describing smoke tests, critical paths, invariants, and known gaps. Without it, the review still works but is less targeted. If no briefing exists, suggest `/review-briefing` and proceed with best-effort code inspection.

## Parse invocation

The user may pass arguments. Parse them:

```
/review                          → full review, all phases, entire project
/review --scope fused-memory     → scoped to one subproject
/review --focused mod1 mod2      → focused on specific modules
/review --phase integration      → Phase 1 only
/review --phase architecture     → Phase 2 only
/review --phase triage           → Phase 3 only (reads saved Phase 1+2 reports)
```

Multiple flags can combine: `/review --scope fused-memory --phase integration`.

## Before you begin

### 1. Load the review briefing

Check for `review/briefing.yaml` (or the path from orchestrator config's `review.briefing_path`).

**If it exists:** parse it. Extract the sections relevant to the current scope:
- If `--scope X`: load only `subprojects.X`
- If `--focused mod1 mod2`: load subproject sections whose `root` or `critical_paths[].key_modules` overlap with the specified modules
- If no scope: load everything

Keep the briefing data in working memory — you'll reference smoke tests, critical paths, invariants, and known gaps throughout all three phases.

**If it doesn't exist:**
1. Tell the user: "No review briefing found at `review/briefing.yaml`. Review effectiveness will be limited without one — consider running `/review-briefing` first. Proceeding with best-effort review using code inspection and project memory."
2. Continue without briefing data. Skip smoke tests (Phase 1 step 3) and critical path tracing (Phase 2 step 2) — these require briefing input. Everything else works.

### 2. Load project context from memory

Search fused-memory for context that informs the review:

```
search(query="recent decisions, known issues, active conventions", project_id="dark_factory")
search(query="known test flakes and pre-existing failures", project_id="dark_factory")
```

This context helps you:
- Distinguish new issues from known ones
- Avoid flagging intentional design choices as problems
- Understand why certain patterns exist

### 3. Determine which phases to run

| Invocation | Phases |
|-----------|--------|
| No `--phase` flag | All three, sequentially |
| `--phase integration` | Phase 1 only |
| `--phase architecture` | Phase 2 only |
| `--phase triage` | Phase 3 only — load saved reports from `review/reports/` |

For Phase 3 alone, check for existing `phase1-*.json` and `phase2-*.json` reports. If none exist, warn the user and offer to run the earlier phases first.

---

## Phase 1: Integration Verification

**Goal:** Does the software actually run and do what it claims?

This phase is mechanical — test execution, smoke tests, lint, typecheck. Parallelise where possible using Sonnet sub-agents.

Read `references/phase1-integration.md` for the detailed step-by-step process.

**Summary of steps:**
1. Run the full test suite (unit + integration + e2e), parse and classify results
2. Run lint (`ruff check`) and typecheck (`pyright`) across the full scope
3. Run smoke tests from the review briefing (if available)
4. Compile a structured Phase 1 report

**Agent allocation:** Spawn Sonnet sub-agents for parallel execution:
- One agent for test suite execution and result parsing
- One agent for lint + typecheck
- One agent for smoke tests (sequential, since they may have setup/teardown)

Collect all results and compile the Phase 1 report yourself (the coordinating agent).

**Output:** Save report to `review/reports/phase1-{timestamp}.json` and display a summary to the user before proceeding.

---

## Phase 2: Architectural Coherence

**Goal:** Is the codebase internally consistent and complete?

This is the expensive, high-value phase. You (Opus) do the architectural reasoning. Spawn Sonnet sub-agents for mechanical scanning tasks, then synthesise.

Read `references/phase2-architecture.md` for the detailed step-by-step process.

**Summary of steps:**
1. Stub and placeholder audit — find TODOs, NotImplementedError, pass bodies, etc. Cross-reference against task tree, briefing known gaps, and memory
2. Critical path tracing — follow each critical path from the briefing through actual code, checking wiring at every module boundary
3. Deep read of high-risk modules — read server startup, config loading, pipeline stages, Dockerfiles, and shared utilities line by line. This is where you find behavioral bugs that structural scanning misses: wrong variables passed, hardcoded assumptions, missing initialization, port/path drift
4. Cross-module consistency — API naming, data flow across boundaries, config coherence, import health
5. Dead code and orphan detection
6. Test coverage analysis (qualitative — are critical paths tested end-to-end?)

**Agent allocation:**
- Spawn Sonnet agents for: stub scanning (grep patterns), import graph mapping, dead code enumeration
- Do the reasoning yourself: path tracing, deep read, cross-module analysis, coverage judgment

**Output:** Save report to `review/reports/phase2-{timestamp}.json` and display a summary before proceeding.

---

## Phase 3: Triage and Task Creation

**Goal:** Convert findings into actionable work items or escalations.

This phase requires judgment — classifying severity, deciding what's a task vs an escalation, avoiding duplicate tasks.

Read `references/phase3-triage.md` for the detailed step-by-step process.

**Summary of steps:**
1. Load Phase 1 + Phase 2 reports (from files if running `--phase triage`, from memory if running all phases)
2. Classify each finding: auto-fix, clear-cut issue, design question, known/accepted, or stale task
3. Task tree health check — review pending tasks against current codebase state
4. Create tasks via fused-memory MCP for actionable findings
5. Escalate ambiguous findings to the user with options and recommendations
6. Write review summary to memory

**Output:** Display the full review summary (all three phases) and list created tasks and escalations.

---

## Coordinating the phases

When running all three phases:

1. **Run Phase 1.** Display summary. If there are blocking failures (e.g., nothing compiles), ask the user whether to proceed to Phase 2 or stop and fix first.

2. **Run Phase 2.** Display summary. Proceed to Phase 3.

3. **Run Phase 3.** Display the full review report, created tasks, and escalations.

Between phases, save intermediate reports to `review/reports/` so they can be consumed independently (e.g., `--phase triage` later).

### Report directory setup

```bash
mkdir -p review/reports
```

Use ISO 8601 timestamps in filenames: `phase1-20260324T143000.json`, `phase2-20260324T144500.json`, `summary-20260324T150000.md`.

---

## Output format

### Interactive summary (shown to user)

After all phases complete, display a structured summary:

```markdown
## Review Complete: {scope}

### Phase 1: Integration Verification
- Test suite: {passed} passed, {failed} failed ({new_failures} new, {known} known)
- Smoke tests: {pass_count}/{total} passed — FAILED: {list of failures}
- Lint: {new_count} new issues
- Type-check: {status}

### Phase 2: Architectural Coherence
- {n} unintended stubs found (tasks claimed done)
- {n} critical path issues: {brief descriptions}
- {n} orphan modules
- {n} integration test gaps

### Phase 3: Triage
Created {n} tasks:
  - Task {id}: {title} ({priority})
  ...

Escalated {n} findings for your review:
  1. {finding} — {question}
  ...
```

### Persistent outputs

| File | Content |
|------|---------|
| `review/reports/phase1-{ts}.json` | Structured test/lint/smoke results |
| `review/reports/phase2-{ts}.json` | Architectural findings with evidence |
| `review/reports/summary-{ts}.md` | Human-readable full summary |
| fused-memory entries | Review observations and session summary |
| Taskmaster tasks | Created via fused-memory MCP with `metadata.source: "review-cycle"` |

---

## Graceful degradation

The skill should work across a range of project states:

| Missing | Impact | Behaviour |
|---------|--------|-----------|
| Review briefing | No smoke tests, no critical path tracing | Warn, suggest `/review-briefing`, continue with code inspection |
| fused-memory | No memory context, no task creation | Warn, produce report only (no tasks), suggest checking MCP connection |
| Test suite | No Phase 1 test results | Run lint/typecheck only, note missing tests in report |
| Lint/typecheck config | No static analysis | Skip, note in report |

Never fail silently — always tell the user what's missing and how it limits the review.

---

## Writing to memory

At the end of a review (after Phase 3, or after whichever phase is the last one run), write to fused-memory:

```
add_memory(
  content="Review completed for {scope}: {key findings summary}. Created {n} tasks. Key concerns: {list}.",
  category="observations_and_summaries",
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```

If significant architectural issues were found, write those as separate memories:

```
add_memory(
  content="{specific architectural finding and its implications}",
  category="decisions_and_rationale",  # or observations_and_summaries
  project_id="dark_factory",
  agent_id="claude-interactive"
)
```
