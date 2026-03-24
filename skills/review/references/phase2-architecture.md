# Phase 2: Architectural Coherence — Detailed Guide

This phase answers: **is the codebase internally consistent and complete?**

This is where the review earns its keep. Per-task verification can't catch the class of issues that only surface when you look across module boundaries — broken wiring, plausible-looking stubs, type mismatches at integration points, tests that mock so thoroughly they never test real integration.

You (Opus) do the architectural reasoning. Spawn Sonnet sub-agents for the mechanical scanning, then synthesise their findings with your own analysis.

## Step 1: Stub and placeholder audit

### Mechanical scan (delegate to Sonnet agent)

Scan the codebase for indicators of incomplete implementation:

```
# Patterns to search for
TODO, FIXME, HACK, XXX
raise NotImplementedError
pass  (as sole function body — not in except blocks or abstract methods)
...   (Ellipsis as implementation — not in type stubs or overloads)
return None  (in functions with non-None return type annotations)
return {}  or return []  (hardcoded empty returns in non-trivial functions)
"not implemented", "placeholder", "stub"
```

For each hit, capture: file, line, function name, surrounding context (5 lines each side).

### Cross-reference (your job)

For each potential stub, check three sources:

1. **Task tree** — query `get_tasks` and check: was this function's implementation assigned to a task marked `done`? If so, this is an **unintended stub** — the task claims complete but the implementation is placeholder. High severity.

2. **Review briefing known gaps** — does the briefing's `known_gaps` section mention this? If so, it's a **known gap** — documented, intentionally deferred. Info severity.

3. **Project memory** — `search(query="decision to defer {function_name} implementation")`. Was there an explicit decision to leave this as a stub? If so, it's **accepted**. Info severity.

4. **Code context** — is this an abstract base class method? A protocol definition? A test fixture placeholder? These are **acceptable** by nature. Skip them.

### Classify

| Classification | Criteria | Severity |
|---------------|----------|----------|
| Unintended stub | Task claims done, implementation is placeholder | High |
| Known gap | Listed in briefing `known_gaps` | Info |
| Accepted decision | Found in memory as explicit deferral | Info |
| Structural | ABC, Protocol, type stub | Skip |
| Unknown | No context found — needs human judgment | Warning |

## Step 2: Critical path tracing (requires briefing)

If no review briefing exists, skip this step. Without defined critical paths, path tracing is speculative — better to focus on the other steps.

For each critical path in the briefing:

### Trace the actual code

Follow the execution flow described in the `trace` section. At each step:

1. **Does the function exist?** Check that the module and function referenced in the trace are present and haven't been renamed or moved.

2. **Does it call the next step?** Read the function body. Does it actually invoke the next function in the trace, or does it return early, call something else, or have a conditional branch that skips the call?

3. **Are types compatible at the boundary?** Check that the return type of step N matches the parameter types of step N+1. Look for subtle mismatches: `Optional[X]` passed where `X` is expected, `dict` where a dataclass is expected, `str` where an enum is expected.

4. **Are runtime dependencies satisfied?** Check for:
   - Imports that could fail (conditional imports, optional dependencies)
   - Config values read but not defined in config files
   - Environment variables used but not documented
   - Services assumed to be running (database, API, MCP server)

5. **Is error handling consistent?** If step N raises `ValueError`, does step N+1 catch it or let it propagate? Is there a coherent error handling strategy across the path, or does each module do its own thing?

### What to look for

The most common critical path failures:
- **Wiring gaps**: function A is supposed to call function B, but the call was never added (task implemented A and B independently but nobody wired them together)
- **Mock-masked failures**: the integration test mocks the boundary between A and B so thoroughly that the real call path is never exercised
- **Conditional short-circuits**: a feature flag, config check, or early return prevents the path from executing in practice
- **Stale imports**: module was refactored, import path changed, but callers still import from the old location (Python may not error if the old module file still exists)

### Record findings

For each critical path issue:
- Which path, which step
- What the code actually does vs what the trace says it should do
- Evidence (the specific code lines)
- Suggested fix

## Step 3: Deep read of high-risk modules

Steps 1–2 find structural issues (stubs, broken paths). This step finds behavioral bugs — the kind that only surface when you actually read the code and think about what it does at runtime. These are the highest-value findings because they're the hardest to catch with automated tools or checklist-driven scanning.

### Identify high-risk modules

High-risk modules are those where a bug has outsized impact or where complexity creates hiding places for mistakes. Typically:

- **Server/application startup** — initialization order, service wiring, port bindings, config loading. Bugs here mean nothing works.
- **Configuration and settings** — where values are defined, defaulted, and consumed. Mismatches between config files, code defaults, and documentation are common.
- **Pipeline stages and orchestration** — multi-step processes where data flows through stages. Each handoff is a potential type or semantic mismatch.
- **Infrastructure files** — Dockerfiles, docker-compose, CI configs. These often drift from the code they deploy (wrong ports, stale paths, missing dependencies).
- **Shared utilities** — code used by multiple subsystems. A bug here multiplies.

### What to look for

Read each high-risk module carefully — not scanning for patterns, but thinking about what the code actually does at runtime. The bugs you're hunting are semantic, not syntactic:

- **Wrong variable passed** — a function receives `project_id` (a logical name like `"dark_factory"`) where it needs `project_root` (an absolute path like `"/home/leo/src/dark-factory"`). Both are strings, so the type system won't catch it. The only way to find this is to read the call site and the callee and notice the mismatch.
- **Hardcoded assumptions** — code that creates `AsyncOpenAI()` regardless of the configured LLM provider. A function that hardcodes a port number that should come from config. A path that only works on the developer's machine.
- **Missing initialization** — a fallback object is created but never has `.initialize()` called on it. A service is constructed but `.start()` is never invoked. A connection pool is created but never warmed.
- **Missing cleanup** — server startup creates resources but shutdown never calls `.close()` or `.dispose()`. File handles or database connections that leak on exit.
- **Port/path/ID drift** — a Dockerfile `EXPOSE`s port 8000 but the server binds to 8002. A config file says one thing, the code defaults to another, and the documentation says a third.
- **Provider coupling** — code that's supposed to be provider-agnostic but imports or instantiates a specific provider unconditionally.

### Process

1. List the high-risk modules for the current scope (5–10 files max — focus on impact, not coverage)
2. Read each one thoroughly, not just the function signatures but the bodies
3. For each potential issue, verify it's real: trace the call chain, check the config, confirm the mismatch
4. Record findings with the specific file, line, what's wrong, and why it matters

This step is where unstructured deep reading pays off. Don't rush it — the bugs found here are typically higher severity than anything from the structural scans.

## Step 4: Cross-module consistency (renumbered from Step 3)

### API surface consistency

Compare public interfaces across modules in the review scope:

- **Naming conventions**: do similar operations use consistent naming? (e.g., `add_memory` vs `create_memory` vs `insert_memory` — pick one)
- **Error patterns**: do all modules use the same error types for the same kinds of failures?
- **Return types**: do similar operations return consistent types? (e.g., some return `dict`, others return dataclasses for the same kind of data)
- **Parameter ordering**: is there a consistent parameter convention? (e.g., `project_id` always first, or always last?)

### Data flow across boundaries

Trace how data structures transform as they cross module boundaries:
- Is data serialised/deserialised correctly at each boundary?
- Are optional fields handled consistently? (one module sets `None`, another expects the key to be absent)
- Are enums/constants defined in one place and imported, or duplicated across modules?

### Configuration coherence

- Are all config keys referenced in code actually defined in config files?
- Are default values consistent between code and config?
- Are there config keys defined but never read?
- **Port numbers** — do config files, code defaults, Dockerfiles, docker-compose, and documentation all agree on the same ports for each service?
- **Paths** — are paths consistent between config, code, and infrastructure files? Are any hardcoded to a specific developer's machine?
- **IDs and names** — where a logical name (like a project ID) and a filesystem path (like a project root) are both used, are they passed to the right parameters?

### Import health

- Circular dependencies (A imports B, B imports C, C imports A)
- Missing transitive dependencies in `pyproject.toml`
- Modules that import from internal paths of other packages (fragile coupling)

## Step 5: Dead code and orphan detection

### Delegate scanning to Sonnet agent

Find:
- **Orphan modules**: Python files that are never imported by anything else in the project
- **Unused exports**: functions/classes defined in `__init__.py` or `__all__` but never imported
- **Unused functions**: defined but never called (careful: exclude entry points, CLI handlers, MCP tool handlers, test fixtures)
- **Dead config entries**: keys in config files that nothing reads
- **Stale test files**: test files whose test subjects no longer exist

### Validate findings

Not everything that looks dead is actually dead:
- MCP tool handlers are called by the framework, not by direct import
- CLI entry points are invoked externally
- Fixtures are discovered by pytest
- Some modules are imported dynamically (`importlib`)

Check each candidate before flagging it. When in doubt, flag as "possibly dead — verify before removing" rather than definitively.

## Step 6: Test coverage analysis (qualitative)

This is not about line-coverage percentages. It's about whether the things that matter are tested in ways that would actually catch breakage.

### For each critical path (from briefing)

- Is there an integration test that exercises the full path with real implementations (not mocks)?
- If mocks are used, what exactly is mocked? Is the mock realistic?
- Would the test catch a wiring change (e.g., function A stops calling function B)?

### For integration boundaries

- Are there tests that cross module boundaries with real data?
- Or do tests only exercise each module in isolation?
- Where mocks are used at boundaries, could the mock diverge from the real implementation without the test noticing?

### Gap identification

Flag areas where:
- Only unit tests exist for functionality that involves multiple modules
- Mocks are so thorough that the tests would pass even if the real integration is completely broken
- Critical paths have no dedicated integration/e2e test

## Step 7: Compile Phase 2 report

Write to `review/reports/phase2-{timestamp}.json`:

```json
{
  "phase": 2,
  "timestamp": "2026-03-24T14:45:00Z",
  "scope": "full | subproject-name | focused:mod1,mod2",
  "stubs": {
    "total": 8,
    "unintended": 3,
    "known_gaps": 2,
    "accepted": 1,
    "structural": 2,
    "findings": [
      {
        "file": "fused_memory/graphiti_client.py",
        "line": 142,
        "function": "bulk_import",
        "pattern": "raise NotImplementedError",
        "classification": "known_gap",
        "reference": "briefing known_gap: task 112",
        "severity": "info"
      }
    ]
  },
  "critical_paths": {
    "traced": 4,
    "issues_found": 1,
    "findings": [
      {
        "path": "Memory write → search round-trip",
        "step": "Classifier routes to Mem0",
        "issue": "classify_and_route returns category but never calls mem0_client.add()",
        "evidence": "classifier.py:67 — returns after classification without invoking store",
        "severity": "high",
        "suggested_fix": "Add store dispatch after classification in classify_and_route()"
      }
    ]
  },
  "deep_read": {
    "modules_read": 7,
    "findings": [
      {
        "file": "fused_memory/routing/classifier.py",
        "line": 23,
        "issue": "WriteClassifier creates AsyncOpenAI() unconditionally — ignores configured LLM provider",
        "category": "hardcoded_assumption",
        "severity": "high",
        "suggested_fix": "Use the configured provider from settings instead of hardcoding OpenAI"
      }
    ]
  },
  "cross_module": {
    "findings": []
  },
  "dead_code": {
    "orphan_modules": [],
    "unused_exports": [],
    "dead_config": []
  },
  "test_coverage_gaps": {
    "findings": []
  }
}
```

## Display summary

```markdown
### Phase 2: Architectural Coherence
- Stubs: 3 unintended (tasks claimed done), 2 known gaps, 1 accepted
- Critical paths: 1 issue — classifier returns without calling Mem0 store
- Cross-module: no issues
- Dead code: 2 orphan modules
- Test gaps: no integration test for cross-store search
```
