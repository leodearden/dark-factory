# Phase 1: Integration Verification — Detailed Guide

This phase answers: **does the software actually run and do what it claims?**

Everything here is mechanical and parallelisable. Spawn Sonnet sub-agents for the heavy lifting, then compile results yourself.

## Step 1: Run the full test suite

Determine the test command for the target scope:
- Check orchestrator config for `test_command` and any per-subproject overrides (e.g., `fused-memory/orchestrator.yaml`)
- Default: `pytest` at the project root
- If scoped to a subproject: `uv run --project {subproject} pytest`

Run the full suite — not task-scoped. The point is to catch failures that per-task verification missed.

### Parse results

Classify each failure:

| Classification | How to determine | Action |
|---------------|------------------|--------|
| **New failure** | Not in known failures list (from memory or briefing) | Flag as finding, severity: high |
| **Known flake** | Search memory for "flaky test" + test name | Note as known, severity: info |
| **Pre-existing** | Failure exists on main before recent changes | Note as pre-existing, severity: warning |
| **Integration vs unit** | Check test file path (tests/integration/, tests/unit/, tests/e2e/) | Tag accordingly |

To distinguish new from pre-existing: if the test file hasn't been modified recently (`git log --since="2 weeks" -- {test_file}`), it's likely pre-existing.

### Capture output

For each failure, capture:
- Test name and file path
- Error message and traceback (truncated to relevant portion)
- Classification (new/known/pre-existing)
- Affected module(s)

## Step 2: Lint and type-check

Run both across the full project scope:

```bash
# Lint
ruff check {scope_path} --output-format json

# Type-check (per-subproject if scoped)
uv run --project {subproject} pyright --outputjson
```

### Classify results

For lint and type-check issues, distinguish new from pre-existing:
- Check `git diff main --name-only` to see which files were recently changed
- Issues in recently changed files → likely new → severity: warning
- Issues in untouched files → pre-existing → severity: info

Don't waste time on pre-existing issues unless they're in modules covered by the current review scope.

## Step 3: Smoke tests (requires briefing)

If no review briefing exists, skip this step entirely.

For each smoke test in the briefing:

1. **Setup** (if specified): run the setup command
2. **Execute**: run the test command
3. **Evaluate**: check against the `expect` condition:
   - `exit 0` → check exit code
   - `json_field: key = value` → parse JSON output and check field
   - `contains: text` → check stdout contains text
   - `regex: pattern` → match against stdout
4. **Teardown** (if specified): run cleanup regardless of pass/fail
5. **Record**: pass/fail, stdout, stderr, diagnosis for failures

### Failure diagnosis

For smoke test failures, attempt basic diagnosis:
- Exit code 1 with `ModuleNotFoundError` → missing dependency
- Exit code 1 with `ImportError` → broken import chain
- Connection refused → service not running
- `FileNotFoundError` → missing config or data file
- Timeout → service hanging on startup

Include the diagnosis in the report — it saves time in Phase 3 triage.

## Step 4: Compile Phase 1 report

Write a structured JSON report to `review/reports/phase1-{timestamp}.json`:

```json
{
  "phase": 1,
  "timestamp": "2026-03-24T14:30:00Z",
  "scope": "full | subproject-name | focused:mod1,mod2",
  "test_results": {
    "total": 142,
    "passed": 139,
    "failed": 3,
    "failures": [
      {
        "test": "test_round_trip_mem0",
        "file": "tests/integration/test_round_trip.py",
        "error": "AssertionError: search returned 0 results",
        "classification": "new",
        "severity": "high",
        "modules": ["fused_memory/mem0_client.py", "fused_memory/mcp_tools.py"]
      }
    ]
  },
  "lint_results": {
    "total_issues": 5,
    "new_issues": 2,
    "issues": [
      {
        "file": "fused_memory/classifier.py",
        "line": 47,
        "code": "E501",
        "message": "Line too long",
        "classification": "new",
        "severity": "warning"
      }
    ]
  },
  "typecheck_results": {
    "total_errors": 0,
    "new_errors": 0,
    "errors": []
  },
  "smoke_tests": {
    "total": 5,
    "passed": 4,
    "results": [
      {
        "name": "Health endpoint responds",
        "passed": false,
        "stdout": "",
        "stderr": "Connection refused",
        "diagnosis": "Server not running — smoke test requires fused-memory server on port 8002",
        "severity": "warning"
      }
    ]
  }
}
```

## Display summary

After compiling the report, show the user a concise summary:

```markdown
### Phase 1: Integration Verification
- Test suite: 139/142 passed (2 new failures, 1 known flake)
- Lint: 2 new issues (E501, E712 in classifier.py)
- Type-check: clean
- Smoke tests: 4/5 passed — FAILED: "Health endpoint" (server not running)
```

If there are blocking failures (widespread test failures, nothing compiles), flag this clearly — the user may want to fix before proceeding to Phase 2.
