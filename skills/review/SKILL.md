---
name: review
description: "Deep, multi-phase code review — runs tests/lint/typecheck, audits architecture for stubs, broken wiring and quality-heuristic violations against docs/code-quality.md, then routes findings into tasks or deliberation. Incremental by default: re-reviews what changed since the last committed review report and re-verifies that report's open findings. ALWAYS use this skill for: /review commands, requests to verify software actually works end-to-end, post-orchestrator quality checks ('the tasks are done but does it actually work?'), finding stubs/NotImplementedError/placeholders, checking integration health across modules, finding dead code or orphan modules, auditing cross-module consistency after merges, running the full test suite with analysis (not just 'run pytest'), qualitative test coverage analysis ('are the tests just mocking everything?'), and a dispatched 'Run /review on <project>' human gate. This is NOT for: single-file PR reviews, creating review briefings (/review-briefing), historical bug-hotspot mining (/hotspot-survey), deciding which quality instrument to run (/review-all), unblocking tasks, explaining code, fixing lint, running tests without analysis, or implementing tasks. When in doubt about whether the user wants a deep project-wide review vs a simpler action, USE this skill — it handles scoping internally."
---

# Deep Review

Three-phase review that answers: does the software actually run, is it internally consistent, and what needs fixing?

- **Phase 1 — Integration Verification**: tests, smoke tests, lint, typecheck. Mechanical, always the whole scope.
- **Phase 2 — Architectural Coherence**: trace critical paths, find stubs, judge the code against the quality definition. Incremental by default.
- **Phase 3 — Triage**: dedup, route each finding by class, schedule the next run.

Two documents are normative and are cited, never restated:

- **The quality definition** (`$QUALITY_DOC`). Every prompt that judges code carries it and tags findings with the heuristics they rest on (contract §10), one of two ways: when the project carries `docs/code-quality.md`, the prompt says `Read docs/code-quality.md`; otherwise (reify today) the prompt embeds the block `orchestrator/src/orchestrator/agents/code_quality.py::guidance` renders — print it from a dark-factory checkout with `python -c 'from orchestrator.agents.code_quality import guidance; print(guidance())'`. Never restate it by hand.
- **The findings contract** (`$CONTRACT`) — `docs/quality-findings-contract.md`. Finding fields, keys, severity, the method header, dispositions, dedup and the trigger chain all come from it; this skill cites it by section (`§n`). A project without its own copy uses the dark-factory checkout's.

The review is driven by the **review briefing** (`review/briefing.yaml`, or `review.briefing_path` in the project's orchestrator config). Without it the review still runs, best-effort; suggest `/review-briefing`.

## Parse invocation

```
/review                          → incremental: Phase 2 since the latest committed report, all phases
/review --full                   → Phase 2 over the whole scope
/review --since <sha>            → incremental from an explicit commit (`--since none` = `--full`)
/review --scope <area>           → one briefing subproject
/review --focused <path> <path>  → specific modules
/review --phase integration      → Phase 1 only
/review --phase architecture     → Phase 2 only
/review --phase triage           → Phase 3 only, completing the latest report that has no Phase 3
/review --as-of <sha>            → pin the run to this main commit instead of main's tip
/review --findings-only          → Phases 1–2, write the report, stop: no Phase 3 filing, deliberation or trigger chain
/review --launched-from <esc id> → the escalation that launched this run (a review_due human gate)
```

Flags combine: `/review --scope fused-memory --phase integration`. `--full` and `--since` are mutually exclusive. `/review-all` runs `/review --as-of <its as_of_sha> --since <sha> --findings-only` and triages the merged findings itself.

## Before you begin

### 0. Pin the run

Every anchor in a run is verified against one tree (§5).

1. **Project.** `project_root` is the main checkout root — `git worktree list | head -1 | awk '{print $1}'` (the derivation `skills/do/SKILL.md` owns; `git rev-parse --show-toplevel` answers the worktree you stand in). `project_id` is `fused_memory.project_id` in `<project_root>/dark-factory-orchestrator.yaml`. Never assume either.
2. **`as_of_sha`** = the `--as-of` commit when given (it must be an ancestor of main), else `git -C <project_root> rev-parse main`. `--launched-from <esc id>` is recorded as `method.extra.launched_from`.
3. **`run_id`** = `review-<project_id>-<YYYYMMDD>`, with `-<n>` when `review/reports/` already holds that day's id.
4. **Pinned tree.** `git -C <project_root> worktree add --detach <project_root>/.claude/worktrees/review-<run_id> <as_of_sha>`, then run the project's `verify_cold_preprovision_command` (orchestrator config) inside it. Every phase reads and runs there. Reports are written to `<project_root>/review/reports/`, not the pinned tree.
5. **`since`.** `--since <sha>` wins. Otherwise read the `method` of the newest committed report, `git -C <project_root> ls-files 'review/reports/review-*.json'`, whose `extra.phases_run` includes 2 and whose scope covers this run's (a `full` report covers everything; an `<area>` report covers that area). Its `as_of_sha` is `since` and its findings are the carry set. No such report, `--full`, `since` not an ancestor of `as_of_sha`, or a diff `since..as_of_sha` that touches `docs/code-quality.md` or `docs/legibility/design-invariants.md` → `mode: full`, with the reason in `extra.full_reason`.

### 1. Load and validate the briefing

Parse the briefing and keep it in working memory. Extract by scope: `--scope X` loads `subprojects.X` plus top-level `conventions`; `--focused` loads the subprojects that own the named paths; no scope loads everything.

- **Prioritise** — `key_scenarios` and `stability_concerns` say what to trace.
- **Contextualise** — `key_decisions` explain choices that would otherwise look wrong.
- **Construct verification** — `what_working_means` is intent; build commands from the code.
- **Conventions** are rules with teeth: a violation is a finding tagged `kind:convention` plus the heuristic it breaks.
- **Known gaps** are accepted dispositions, each pointing at its accepting task (§7). They suppress a finding only through that pointer (Phase 3).
- **Scope** — `exclude` and `global_exclude` are skipped.

Then run the `/review-briefing --validate` checks in report-only mode (no fix offers) and copy every defect into `method.extra.briefing_defects`. They never block the run. There is no commit-count staleness rule: staleness is what the validate checks find.

No briefing: say so, suggest `/review-briefing`, skip smoke tests and critical-path tracing, use workspace member names as areas.

### 2. Load project context from memory

```
search(query="recent decisions, known issues, active conventions", project_id="<project_id>")
search(query="known test flakes and pre-existing failures", project_id="<project_id>")
```

### 3. Determine phases

| Invocation | Phases |
|-----------|--------|
| no `--phase` | 1, 2, 3 in order |
| `--phase integration` | 1 |
| `--phase architecture` | 2 (Steps 0–8 of the Phase 2 guide) |
| `--phase triage` | 3, on the newest report whose `extra.phases_run` lacks 3; none → offer to run the earlier phases |
| `--findings-only` | 1, 2, then write and commit the report and stop |

## Incremental mode

Phase 1 always runs in full: it is cheap, mechanical, and main may be red. Phase 2 narrows to the **changed scope** — files in `git diff --name-only <since> <as_of_sha>` inside the run's scope, minus excludes, plus the modules that import them — and re-verifies the **carry set**: every finding in the `since` report with disposition `open`, `filed:<id>` or `accepted:<id>`. How each is computed: `references/phase2-architecture.md` Step 0.

| Item | `mode: since` | `mode: full` |
|---|---|---|
| Phase 1 tests, lint, typecheck, smoke | recomputed | recomputed |
| Briefing validate checks | recomputed | recomputed |
| Inputs (hotspot, `/review-all`, codebook) | re-read | re-read |
| Project `/audit` | recomputed from `since`'s committer date | recomputed over `audit.window_days` |
| Stub scan, invariants audit, deep read, cross-module, coverage | changed scope only | whole scope |
| Critical-path tracing | paths whose trace touches the changed scope | every path |
| Dead-code scan | mechanical scan whole tree; judged only for candidates whose key is not in the carry set | whole scope |
| Carry set (prior `open`/`filed`/`accepted`) | re-verified at `as_of_sha`: still present → kept, `last_seen` appended; gone → `fixed:<as_of_sha>`; anchor moved → new key with `supersedes` | same |
| Judgment of unchanged code holding no carried finding | reused: not re-read | recomputed |
| Phase 3 dedup and routing | every finding in this report | every finding in this report |

## Phase 1: Integration Verification

Does the software actually run? Detail: `references/phase1-integration.md`.

1. Run the project's configured `test_command`, `lint_command` and `type_check_command` in the pinned tree (scoped per member when `--scope` is set; `uv run --directory <member>`, never `--project`).
2. Classify failures (new, known flake, pre-existing) and find each one's owner task.
3. Run smoke checks built from the briefing's `what_working_means`.
4. Write the `phase1` block of the report (`references/run-record.md`) and show a summary.

Blocking failures (nothing compiles, most tests red): ask whether to continue to Phase 2.

## Phase 2: Architectural Coherence

Is the codebase internally consistent, complete, and cheap to change? Detail: `references/phase2-architecture.md`.

0. Compute the changed scope (incremental) and re-verify the carry set (both modes).
1. Project `/audit`, when the project ships one.
1.5. Read the inputs: latest hotspot and `/review-all` reports, the confusion codebook (§9), and the latest committed metrics snapshot — the newest `plans/quality-metrics/*.json` by commit (`git log -1 --diff-filter=A --name-only --format= -- plans/quality-metrics/`), rendered with `scripts/quality_metrics_snapshot.py --summary`. The snapshot is context for choosing Step 4's modules, never a ranking (contract §10).
2. Stub and placeholder audit.
3. Critical-path tracing (requires briefing).
4. Deep read of high-risk modules, including the heuristic look-fors.
5. Cross-module consistency.
5.5. Design-invariants audit, tags `inv-<n>`.
6. Dead code and orphans.
7. Test coverage and the Tests stance.
8. Write every finding into the one `findings` list (`references/run-record.md`).

## Phase 3: Triage and Routing

Detail: `references/phase3-triage.md`.

1. Load the report.
2. Phase 1 failures: find or file an owner (operational, outside the finding schema).
3. Class each confirmed or weakened finding: mechanical or structural (§6).
4. Dedup every finding by the §8 protocol and set its disposition.
5. File mechanical findings as curator tickets carrying the §8 metadata.
6. Take structural findings to deliberation with the user; acceptances become owned tasks.
7. File the trigger chain for the next run (§11) by `skills/_shared/filing-the-trigger-chain.md`, superseding any open `/review` chain and resolving the `--launched-from` escalation.
8. Write the review summary to memory.

## Routing: who does what

The coordinator is the session running this skill. Seats are `Agent` calls; pass `model` on the call and state the effort in the seat prompt (and set it where the runtime exposes an effort control).

| Work | Done by | Model | Effort |
|---|---|---|---|
| Pinning, scope computation, report assembly | coordinator | the session's model (an Opus-class model is recommended) | high |
| Phase 1 test, lint+typecheck, smoke seats (parallel) | seats | sonnet | low |
| Stub grep, import graph, importer lookup, dead-code enumeration, codebook digest | seats | sonnet | low |
| Carried-finding re-verification, ≤10 findings per seat | seats | sonnet | medium |
| Steps 3, 4, 5, 5.5, 7 judgment | coordinator; split by area into opus seats when the Phase 2 file set exceeds ~40 files | opus | high |
| Phase 3 classing, dedup decisions, deliberation | coordinator | — | high |

Every judging seat's prompt opens with `$QUALITY_DOC` — "Read `docs/code-quality.md` in full", or the embedded `guidance()` block — followed by: "Tag every finding with the heuristics it rests on (`h1`..`h14`, `comments`, `tests`, `inv-<n>`), lens first." (§10)

## Output

### Interactive summary

```markdown
## Review complete: {scope} — {run_id}
as_of {as_of_sha:10} · since {since:10 | none} · mode {since|full}

### Phase 1
- Tests: {passed} passed, {failed} failed ({new} new — owners: {task ids or "filed {id}"})
- Lint: {status} · Type-check: {status} · Smoke: {n}/{total}

### Phase 2
- Scope: {changed} changed files + {importers} importers; {carried} carried findings re-verified
- Findings: {high} high · {medium} medium · {low} low — by lens: {h7: 3, tests: 2, …}
- Fixed since last run: {keys}
- Briefing defects: {n}

### Phase 3
Filed {n} (mechanical): Task {id}: {title} ({priority}) …
Already owned {n}: {key} → filed:{id} …
For deliberation {n} (structural): 1. {statement} — {question}
Next run: {chain form and task ids}
```

### Persistent outputs

| Output | Content |
|------|---------|
| `review/reports/<run_id>.json` | the run record: `method`, `phase1`, `findings` (`references/run-record.md`) |
| `review/reports/<run_id>.md` | rendering of the same |
| tasks | mechanical findings, acceptances, the trigger chain — all through fused-memory MCP with `metadata.source: "review"` |
| fused-memory | review summary and pattern observations |

### Committing the report

After the last phase run, write both files under `<project_root>/review/reports/` and commit only them to main by the project's direct-commit rule (in dark-factory: `git add` the two paths, then `git commit --only` them, never while a merge verify is in flight — `CLAUDE.md` §"Working in the main checkout"). A project that forbids direct commits gets them through `/merge-queue`. An uncommitted report is invisible to the next run's `since`. Then `git worktree remove` the pinned tree.

## Graceful degradation

| Missing | Behaviour |
|---------|-----------|
| Review briefing | Warn, suggest `/review-briefing`, skip smoke and path tracing |
| Prior committed report | `mode: full`, `since: none` |
| Task store unreachable (MCP and read-only sqlite) | Report only; file nothing, say so (§8) |
| fused-memory MCP down | No memory context, no filing; report says so |
| Test suite / lint / typecheck config | Skip that command, note it in `phase1` |
| `$QUALITY_DOC` (no `docs/code-quality.md` and no `guidance()` render) | Stop Phase 2: judging without the definition is not this skill |

Never fail silently — say what is missing and what it costs the review.

## Writing to memory

After the last phase run:

```
add_memory(
  content="Review {run_id} for {scope} at {as_of_sha:10}: {n} findings ({high} high), {filed} filed, {structural} for deliberation, {fixed} fixed since {since:10}. Key concerns: {list}.",
  category="observations_and_summaries",
  project_id="<project_id>",
  agent_id="claude-review",
  entities=[]
)
```

A cross-cutting pattern (several findings sharing one cause) is a separate `observations_and_summaries` write.
