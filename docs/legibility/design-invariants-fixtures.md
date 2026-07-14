# Design-invariants fixtures

Calibration fixtures and a rehearsal verdict table for the two consumers of
`docs/legibility/design-invariants.md`: `/prd` decompose's G7 gate
(`skills/prd/references/gates.md`) and `/review` phase 2's Step 5.5
design-invariants audit (`skills/review/references/phase2-architecture.md`).

**Normative source.** `docs/legibility/design-invariants.md` is the single
normative copy of the five invariant slugs, rules, and checkable design
questions — this doc does not restate them (per INV-5
`no-lockstep-duplication`). When in doubt about a rule or a checkable
question, Read the normative doc, not this one.

**Two fixture shapes.** Each invariant below carries exactly two seeded
violations — both expressions of the SAME underlying violation, so the two
gate consumers stay calibrated against one shared meaning per slug:

- **PRD-leaf-shaped** — a realistic 2-4-line decomposition-plan row: the
  shape `/prd` decompose's G7 walk (section "G7 — Design invariants pass",
  `skills/prd/references/gates.md`) sees when it walks a batch.
- **Code-snippet-shaped** — a short illustrative snippet or described
  module shape: the shape `/review` phase 2's Step 5.5 design-invariants
  audit (`skills/review/references/phase2-architecture.md`) sees when it
  audits modules in scope. File paths in these snippets are illustrative —
  chosen to NOT collide with any real file in this repo — not pointers
  into the actual codebase.

**Expected-verdict formats.**

- PRD-leaf-shaped fixtures are annotated with the expected G7 disposition —
  `flag: <slug>` — plus the redesign that clears it, so the fixture also
  demonstrates the fix, not just the failure.
- Code-snippet-shaped fixtures are annotated with the expected `/review`
  Step 5.5 finding: an `invariant_findings` entry
  `{"invariant": <slug>, "file": ..., "line": ..., "issue": ..., "severity": ...}`
  (schema per phase2-architecture.md Step 8), with `severity` drawn from
  `{high, warning, info}`.

**Rehearsal verdict-table legend.** The table at the end of this doc walks
the AS-LANDED G7 text against every PRD-leaf-shaped fixture, and the
AS-LANDED Step 5.5 text against every code-snippet-shaped fixture, then
records the verdict each yields against the expected slug. Columns:

| Column | Meaning |
|---|---|
| Fixture ID | `<INV-n>-<PRD\|CODE>` — identifies the fixture block and shape below |
| Shape | `PRD` (walked against G7) or `CODE` (walked against Step 5.5) |
| Invariant | The numeric alias + slug being targeted |
| Expected slug | The exact slug string the gate/review text should emit |
| Verdict | The disposition the as-landed gate/review text actually yields when walked against the fixture |
| Match | `Y` if the verdict's slug equals the expected slug, else `N` |

Acceptance: every fixture flags with the correct slug — all 10 rows `Y`.

## INV-1 `contracts-machine-checked`

### PRD-leaf-shaped (`INV-1-PRD`)

> Add a `priority_boost` fast path: the scheduler's dispatch loop
> special-cases `metadata.priority_boost == true` to jump a task to the
> front of the queue. No schema field, no `submit_task` validation — the
> convention is documented only in this decomposition-plan row and in the
> on-call runbook.

**Expected disposition**: `flag: contracts-machine-checked`

**Redesign that clears it**: Declare `priority_boost` as a first-class,
validated `submit_task` parameter (`ValidationError`+hint guard at the
submit boundary, per the INV-1 house pattern), persisted to `metadata` and
checked at dispatch — not a bare dict-get special-case buried in the
dispatch loop.

### Code-snippet-shaped (`INV-1-CODE`)

```python
def dispatch_next(tasks):
    for t in tasks:
        # priority_boost: agreed informally with the on-call team,
        # not declared anywhere in the task schema
        if t.metadata.get("priority_boost"):
            return t
    return tasks[0] if tasks else None
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "contracts-machine-checked", "file": "orchestrator/src/orchestrator/priority_router.py", "line": 5, "issue": "priority_boost routing contract lives only in a dispatcher-internal special-case and prose comment, not a declared/validated schema field", "severity": "high"}
```

## INV-2 `structured-facts-at-failure`

### PRD-leaf-shaped (`INV-2-PRD`)

> Add a `verify_runner` step: on failure it prints `FAIL: step {name}
> exited {code}` to stdout. The orchestrator's escalation handler regexes
> that line out of the captured log to recover the step name and exit
> code for the escalation report.

**Expected disposition**: `flag: structured-facts-at-failure`

**Redesign that clears it**: Emit a structured evidence field at the
failure point (e.g. `{"step": name, "exit_code": code, "measured_at": ts}`)
instead of a printed line, separating raw observation from any
hypothesis, per the structured `evidence` field house pattern (2558).

### Code-snippet-shaped (`INV-2-CODE`)

```python
def run_step(name, cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: step {name} exited {result.returncode}")
        return False
    return True

def handle_failure(log_text):
    # re-derive what failed by regexing the printed line instead of
    # receiving name/returncode as structured data from run_step
    m = re.search(r"FAIL: step (\S+) exited (\d+)", log_text)
    step, code = m.group(1), m.group(2)
    escalate(f"{step} failed with {code}")
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "structured-facts-at-failure", "file": "orchestrator/src/orchestrator/step_reporter.py", "line": 11, "issue": "handle_failure regexes a printed log line to recover the step name and exit code that run_step already held in local variables", "severity": "high"}
```

## INV-3 `corroborate-before-acting`

### PRD-leaf-shaped (`INV-3-PRD`)

> Add a nightly sweep that reads `task.status` from the last
> reconciliation snapshot and calls `remove_task` on every row still
> showing `status: cancelled` — no re-read of the live task store
> immediately before deleting.

**Expected disposition**: `flag: corroborate-before-acting`

**Redesign that clears it**: Re-fetch live task status (`get_task`)
immediately before the delete call and abort if it no longer reads
`cancelled`, per the Merge Tier-3.5 / `already_merged` genuine-check house
pattern.

### Code-snippet-shaped (`INV-3-CODE`)

```python
def nightly_cleanup(snapshot):
    for row in snapshot.cached_rows:
        if row.status == "cancelled":
            # acts on the cached snapshot value; no live re-read of
            # the task's current status immediately before deleting
            remove_task(row.task_id)
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "corroborate-before-acting", "file": "scripts/nightly_task_sweep.py", "line": 6, "issue": "remove_task is called from a cached snapshot row with no re-read of live task status immediately before the delete", "severity": "high"}
```

## INV-4 `storm-escape-required`

## INV-5 `no-lockstep-duplication`

## Rehearsal verdict table
