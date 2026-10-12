# Phase 1: Integration Verification — Detailed Guide

This phase answers: **does the software actually run and do what it claims?**

Everything here is mechanical and parallelisable: one sonnet seat (low effort) per command group, the coordinator compiles. It always covers the whole run scope, in incremental mode too, and it runs in the pinned tree built from `as_of_sha` (SKILL.md "Pin the run").

Phase 1 produces the report's `phase1` block, not findings (`references/run-record.md`). A failing test on main is operational breakage (contract §4), handled in Phase 3 Step 2.

## Step 1: Run the full test suite

Use the project's configured commands, top-level keys of `<project_root>/dark-factory-orchestrator.yaml`:

- `test_command` — the whole suite, as the merge gate runs it.
- Scoped (`--scope <area>` or `--focused`): run the configured command's segment for that member. If you must construct one, `uv run --directory <member> pytest tests/` — `--directory`, never `--project`: `--project` leaves cwd at the root, so pyright reads the root config instead of the member's (false GREEN; dark-factory's briefing records the measurement under its `uv run --directory` convention).
- No configured command: `pytest` at the root, and say so in `phase1`.

Run the full suite, not task-scoped: the point is what per-task verification missed.

### Classify each failure

| Classification | How to determine |
|---------------|------------------|
| **New** | Not a known flake, and it fails at `as_of_sha` but passed at `since`; with `since: none`, every failure that is not a known flake |
| **Known flake** | Memory search "flaky test {name}", or the project's flake ledger, names it |
| **Pre-existing** | Fails at `since` too — check the previous report's `phase1.failures`, or re-run that one test at `since` |

For each failure capture: test id, member, the first relevant error line, classification, affected modules. Phase 3 looks up its owner.

## Step 2: Lint and type-check

Run `lint_command` and `type_check_command` as configured. Scoped: the member's segment, or `uv run --directory <member> ruff check .` and the member's configured type checker from that directory (cwd is load-bearing for pyright's config).

A gate that is clean at the merge lane should be clean here; anything it reports is either drift between the configured command and what the gate runs, or a red main. Record both cases; do not triage individual lint codes.

## Step 3: Smoke checks (requires briefing)

No briefing: skip, and say so.

For each subproject in scope, turn its `what_working_means` lines into concrete checks from the code (server starts and answers its health route, CLI parses `--help`, a unit's timer is active). For each:

1. **Setup** if the check needs it.
2. **Execute** and evaluate: exit code, JSON field, stdout substring or regex.
3. **Teardown** regardless of outcome.
4. **Record** the command as constructed, pass/fail, and a diagnosis on failure.

A check against a running service exercises what is deployed, not the pinned tree; say which in the record.

### Failure diagnosis

- `ModuleNotFoundError` → missing dependency, or the wrong venv (ask the interpreter, never guess a path)
- `ImportError` → broken import chain
- Connection refused → service not running (a live-service check is not a code failure)
- `FileNotFoundError` → missing config or data file
- Timeout → service hanging on startup

## Step 4: Write the `phase1` block

Shape: `references/run-record.md` §"Phase 1 block". Then show:

```markdown
### Phase 1: Integration Verification
- Test suite: 7387/7389 passed (1 new failure, 1 known flake)
- Lint: clean · Type-check: clean
- Smoke: 6/7 — FAILED: dashboard health (service not running)
```

Flag blocking failures clearly (nothing imports, most of a member red): the user may want to stop before Phase 2.
