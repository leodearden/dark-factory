# Design-invariants fixtures

Calibration fixtures and a rehearsal verdict table for the two consumers of
`docs/legibility/design-invariants.md`: `/prd` decompose's G7 gate
(`skills/prd/references/gates.md`) and `/review` phase 2's Step 5.5
design-invariants audit (`skills/review/references/phase2-architecture.md`).

**Normative source.** `docs/legibility/design-invariants.md` is the single
normative copy of the invariant slugs, rules, and checkable design
questions — this doc does not restate them (per INV-5
`no-lockstep-duplication`). When in doubt about a rule or a checkable
question, Read the normative doc, not this one. INV-1..5 fixtures were
seeded 2026-07-14; INV-6..7 fixtures were added 2026-08-02 with the
task/escalation state-graph invariants; INV-8 fixtures were added
2026-08-06 with the loop-occupancy invariant; INV-9 fixtures were added
2026-08-24 with the one-fact-one-home invariant; INV-10 fixtures were
added 2026-08-30 with the guards-exercise-behaviour invariant (task 4666);
INV-11 fixtures were added with the promotion of `no-silent-fail-soft`.

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

Acceptance: every fixture flags with the correct slug. The base table
holds 10 rows (INV-1..5); the 2026-08-02 addendum adds 4 (INV-6..7), the
2026-08-06 addendum adds 2 (INV-8), the 2026-08-24 addendum adds 2
(INV-9), the 2026-08-30 addendum adds 2 (INV-10), and the INV-11 addendum
adds 2 — 22 rows cumulative, all `Y`.

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

### PRD-leaf-shaped (`INV-4-PRD`)

> Add a suppression counter with no escalation: the watcher increments an
> in-memory `suppressed_count` each time it swallows a duplicate
> escalation, but never surfaces the count anywhere — no rate/streak
> threshold, no escalation if it fires 100×/hr.

**Expected disposition**: `flag: storm-escape-required`

**Redesign that clears it**: Add a consecutive-streak or rate-threshold
escalation — name who hears about it (e.g. `escalate_info` to the
steward) once `suppressed_count` crosses N/hr — per the consecutive-streak
gate / storm-counter (1755) house pattern.

### Code-snippet-shaped (`INV-4-CODE`)

```python
_suppressed_count = 0

def maybe_suppress_escalation(esc):
    global _suppressed_count
    if is_duplicate(esc):
        _suppressed_count += 1  # counted but never compared to a
        return None             # rate/streak threshold or escalated
    return esc
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "storm-escape-required", "file": "escalation/src/escalation/dedup_filter.py", "line": 6, "issue": "_suppressed_count is incremented on every duplicate but never compared against a rate/streak threshold — no escalation path if suppression fires continuously", "severity": "high"}
```

## INV-5 `no-lockstep-duplication`

### PRD-leaf-shaped (`INV-5-PRD`)

> Add event-type validation to the new webhook-delivery task: hand-copy
> the list of allowed event types (`task.created, task.done,
> task.blocked, escalation.filed`) into a new validator module instead of
> importing the canonical list from the existing event-emitter module —
> justified in the row as "the webhook subsystem is self-contained."

**Expected disposition**: `flag: no-lockstep-duplication`

**Redesign that clears it**: Extract `ALLOWED_EVENT_TYPES` into a shared
module (or import it directly from the event-emitter) so both the
emitter and the new validator reference the one list, per the
extract-helper house pattern; add a drift/pinning test if the two ever
need to diverge intentionally.

### Code-snippet-shaped (`INV-5-CODE`)

This fixture is deliberately distinct from the classifier.py/router.py
category-table example in phase2-architecture.md's own Step 8 sample, so
it is a genuine calibration rather than an echo of the doc's sample. (It
is also deliberately NOT about a retry/backoff schedule — a duplicated
retry-absorb mechanism would read as an INV-4 `storm-escape-required`
concern too and defeat the single-invariant requirement below.)

File `orchestrator/src/orchestrator/events.py` (existing, upstream —
illustrative, not a real file in this repo):

```python
ALLOWED_EVENT_TYPES = ["task.created", "task.done", "task.blocked", "escalation.filed"]
```

File `webhooks/src/webhooks/validator.py` (new, the violation):

```python
# hand-copied from orchestrator/src/orchestrator/events.py, kept "in sync" manually
ALLOWED_EVENT_TYPES = ["task.created", "task.done", "task.blocked", "escalation.filed"]
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "no-lockstep-duplication", "file": "webhooks/src/webhooks/validator.py", "line": 2, "issue": "ALLOWED_EVENT_TYPES hand-copied from orchestrator/src/orchestrator/events.py (line 1) — no shared helper or import ties the two lists together, so they must be kept byte-for-byte identical by hand", "severity": "warning"}
```

## INV-6 `status-matches-liveness`

### PRD-leaf-shaped (`INV-6-PRD`)

> Add a `capacity-probe` bail: when the runner detects host memory
> pressure at VERIFY entry, it returns `ABORTED` immediately with no
> status write, releasing the slot — the row stays `in-progress`, and
> "the stranded sweep will re-pend it within one sweep interval."

**Expected disposition**: `flag: status-matches-liveness`

**Redesign that clears it**: Write `pending` (or the appropriate park
status) through the requeue choke point *before* the slot is released,
with a test pinning the exit; the stranded sweep remains the crash
backstop only, never a designed exit.

### Code-snippet-shaped (`INV-6-CODE`)

```python
async def _bail_on_pressure(self) -> Outcome:
    if host_memory_pct() > 95:
        # no status write here — caller's finally releases the slot;
        # the reconcile sweep will eventually revert the row
        return Outcome.ABORTED
    return Outcome.CONTINUE
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "status-matches-liveness", "file": "orchestrator/src/orchestrator/pressure_bail.py", "line": 5, "issue": "the ABORTED exit releases the slot without writing a successor status, leaving an in-progress row with no live claimant and delegating its recovery to the stranded sweep by design", "severity": "high"}
```

## INV-7 `holds-owned-and-bounded`

### PRD-leaf-shaped (`INV-7-PRD`)

> Add a `quota-hold`: when the model quota is exhausted mid-run, file an
> escalation and wait on its resolution event — `await
> self._quota_event.wait()` with no timeout. The watcher will resolve
> the escalation when quota returns, which wakes the wait.

**Expected disposition**: `flag: holds-owned-and-bounded`

**Redesign that clears it**: Bound the wait with a progress-refreshed
idle deadline (the steward-wait house pattern) whose expiry stops the
wait and re-escalates; name the owner that exits the hold; surface the
hold's age on the attention rail.

### Code-snippet-shaped (`INV-7-CODE`)

```python
def maybe_skip_recovery(task, open_escalations):
    if open_escalations:
        # someone is nominally on it — skip this task.
        # No log, no counter, no deadline, no record of which
        # escalation pinned it or for how long.
        return None
    return plan_recovery(task)
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "holds-owned-and-bounded", "file": "orchestrator/src/orchestrator/recovery_skip.py", "line": 5, "issue": "an open escalation vetoes recovery with no structured fact naming the pinning record, no streak counter, and no bound — the hold has no visible owner and can persist indefinitely in silence", "severity": "high"}
```

## INV-8 `loop-thread-occupancy-bounded`

### PRD-leaf-shaped (`INV-8-PRD`)

> Add a staleness badge to the dashboard's status payload: for every task
> on the live board, shell out to `git log -1` in that task's worktree to
> read its last commit timestamp, then format the badge from it. The
> renderer is a plain `def` called from the payload coroutine. Measured at
> ~30 ms per task in local testing, so the added render cost is negligible.

**Expected disposition**: `flag: loop-thread-occupancy-bounded`

**Redesign that clears it**: Offload the blocking git calls
(`asyncio.to_thread` at the boundary, per INV-8's house pattern), hoist
the per-render-invariant work out of the loop body, and bound the fan-out
with an explicit cap that LOGS what it dropped — so worst-case
loop-thread occupancy is a function of the cap, not of board size. Both
limbs are required: offloading alone still burns unbounded wall clock,
and capping alone still blocks the loop.

### Code-snippet-shaped (`INV-8-CODE`)

```python
async def build_status_payload(board) -> dict:
    badges = []
    for t in board.active_tasks:  # uncapped — the whole live board
        # blocking wait AND an inline fork/exec, both on the loop thread
        proc = subprocess.run(
            ["git", "-C", t.worktree, "log", "-1", "--format=%cI"],
            capture_output=True, text=True,
        )
        badges.append(format_badge(t.id, proc.stdout.strip()))
    return {"badges": badges}
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "loop-thread-occupancy-bounded", "file": "dashboard/src/dashboard/status_badges.py", "line": 5, "issue": "build_status_payload runs a blocking git subprocess per task on the event-loop thread over the uncapped active_tasks set — nothing offloads the call and nothing bounds the item count, so worst-case occupancy scales with board size", "severity": "high"}
```

## INV-9 `one-fact-one-home`

### PRD-leaf-shaped (`INV-9-PRD`)

> When the identity-mode question is settled, record the decision in the
> task's description, restate it in the PRD's §4 so future readers see it
> there too, and update the capability manifest's prose copy of the
> identity rule to match. The escalation that raised the question closes
> in a later cleanup pass.

**Expected disposition**: `flag: one-fact-one-home`

**Redesign that clears it**: Pick the home (the escalation record
carries the ruling via amendment/resolution, anchored to the ruling
commit); the task description, PRD, and manifest each gain a dated
POINTER (`esc-id + commit + date`) rather than a restated copy; and the
record-write happens at ruling time, not in a later pass — a deferred
write is what dies with the session.

### Code-snippet-shaped (`INV-9-CODE`)

```python
def record_ruling(task, ruling: str) -> None:
    # write the decision everywhere readers might look
    task.description = f"RULED: {ruling}\n\n" + task.description
    update_task(task)
    prd = Path(task.prd_path).read_text()
    Path(task.prd_path).write_text(
        prd.replace("## Open questions",
                    f"## Open questions\n\nRESOLVED: {ruling}")
    )
    # the escalation record that asked the question is left for cleanup;
    # nothing links the copies or says which one wins
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "one-fact-one-home", "file": "orchestrator/src/orchestrator/ruling_writer.py", "line": 3, "issue": "record_ruling writes the same ruling as independent prose into the task description and the PRD while the escalation record that owns the question is never updated — three copies, no designated home, no pointers, no reconciler, so the copies drift the moment any one is corrected", "severity": "warning"}
```

## INV-10 `guards-exercise-behaviour`

### PRD-leaf-shaped (`INV-10-PRD`)

> The /unblock runbook must keep telling agents to derive the landing SHA
> with a task-scoped search rather than from main's tip. Add a pytest
> guard that greps `skills/unblock/SKILL.md` for the
> `--grep="Merge task/<TASK_ID> into main"` form and fails if a
> `git log --format=%H -1 main` appears anywhere in the file.

**Expected disposition**: `flag: guards-exercise-behaviour`

**Redesign that clears it**: Run the documented derivation against a
fixture repo carrying two task merges and assert it returns THIS task's
merge SHA rather than main's tip. The guard then fails on the defect
itself instead of on the spelling, survives a rewrite of every sentence
around the command, and cannot be defeated by a placeholder the matcher
did not anticipate.

### Code-snippet-shaped (`INV-10-CODE`)

```python
_SCOPED_RE = re.compile(r'--grep="Merge task/<[A-Za-z0-9_]+> into main"')

def test_runbook_derivation_is_task_scoped():
    text = (REPO_ROOT / "skills/orchestrate/SKILL.md").read_text()
    for cmd in _extract_git_commands(text):
        assert _SCOPED_RE.search(cmd), f"unscoped SHA derivation: {cmd}"
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "guards-exercise-behaviour", "file": "tests/scripts/test_runbook_sha_derivation.py", "line": 1, "issue": "the guard asserts on the SPELLING of a documented command rather than on what the command resolves to: its character class excludes the hyphen, so orchestrate's `task/<task-id>` placeholder makes every CORRECT instruction in that runbook fail, while nothing here would notice a well-spelled command that returns the wrong commit — the derivation is runnable against a fixture repo and should be run", "severity": "warning"}
```

## INV-11 `no-silent-fail-soft`

### PRD-leaf-shaped (`INV-11-PRD`)

> Add a `--since` filter to the transcript census: parse each record's
> timestamp, keep the ones inside the window, and skip any record whose
> timestamp fails to parse so one malformed line cannot abort a census
> over 40k records. The command prints the matching-record count and
> exits 0 as it does today; skipped records are noted in the debug log.

**Expected disposition**: `flag: no-silent-fail-soft`

**Redesign that clears it**: Carry the shortfall in the RESULT the caller
sees — report `complete: false` with a named entry counting the skipped
records and why, and exit nonzero when any were skipped — so a partial
census cannot be read as a full one. The debug line is not the result: the
caller deciding whether the count is trustworthy never reads it. Skipping
the malformed record is fine; skipping it invisibly is the defect.

### Code-snippet-shaped (`INV-11-CODE`)

```python
def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("skipping malformed record")
            continue
    return records  # a short list looks exactly like a complete one

def census(path: Path) -> int:
    return len(load_records(path))
```

**Expected `invariant_findings` entry**:

```json
{"invariant": "no-silent-fail-soft", "file": "fused-memory/scripts/transcript_window_census.py", "line": 7, "issue": "load_records drops unparseable lines and returns a plain list, so census() returns a count its caller cannot distinguish from a complete one — the only trace of the shortfall is a debug log the caller never reads, and neither the return value nor the exit code carries it", "severity": "high"}
```

## Rehearsal verdict table

Walked 2026-07-14 against `skills/prd/references/gates.md` §"G7 — Design
invariants pass" and `skills/review/references/phase2-architecture.md`
"Step 5.5: Design-invariants audit", both as landed on `main`.

**Snapshot caveat.** The Verdict column below quotes phrasing (trigger-shape
list entries, checkable questions) directly from the G7 and Step 5.5 text as
it read on 2026-07-14. That phrasing is a point-in-time transcription for
this rehearsal record, not a live pin on the source docs — if
`skills/prd/references/gates.md` §G7 or
`skills/review/references/phase2-architecture.md` Step 5.5 wording changes,
re-walk the fixtures against the new text rather than trusting this table's
quoted rationale as still current.

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-1-PRD` | PRD | INV-1 contracts-machine-checked | `contracts-machine-checked` | G7's trigger-shape list fires on "a contract in prose" / "a tool without a declared filter/envelope convention": the row states the routing rule lives only in the plan row and a runbook → `flag: contracts-machine-checked` | Y |
| `INV-1-CODE` | CODE | INV-1 contracts-machine-checked | `contracts-machine-checked` | Step 5.5's audit applies INV-1's checkable question ("does a new tool/agent surface declare its envelope where callers see it, or is it discovered by failure?") to the undeclared `metadata.get("priority_boost")` special-case → `invariant_findings` entry with `invariant="contracts-machine-checked"` | Y |
| `INV-2-PRD` | PRD | INV-2 structured-facts-at-failure | `structured-facts-at-failure` | G7's trigger-shape list fires on "a log-scrape of emitter-known facts": the escalation handler regexes a printed line instead of receiving structured data → `flag: structured-facts-at-failure` | Y |
| `INV-2-CODE` | CODE | INV-2 structured-facts-at-failure | `structured-facts-at-failure` | Step 5.5's audit applies INV-2's checkable question ("must any consumer parse logs/prose to recover a fact the emitter knew?") to `handle_failure`'s regex over `run_step`'s known values → `invariant_findings` entry with `invariant="structured-facts-at-failure"` | Y |
| `INV-3-PRD` | PRD | INV-3 corroborate-before-acting | `corroborate-before-acting` | G7's trigger-shape list fires on "action on snapshot state without corroboration": the sweep deletes off a cached snapshot read with no live re-check → `flag: corroborate-before-acting` | Y |
| `INV-3-CODE` | CODE | INV-3 corroborate-before-acting | `corroborate-before-acting` | Step 5.5's audit applies INV-3's checkable question ("does this feature act on state that could have changed since read? where exactly is the re-check?") to `remove_task` firing straight off `snapshot.cached_rows` → `invariant_findings` entry with `invariant="corroborate-before-acting"` | Y |
| `INV-4-PRD` | PRD | INV-4 storm-escape-required | `storm-escape-required` | G7's trigger-shape list fires on "adds a detector/suppressor/fallback without a storm escape" — this fixture is the PRD brief's own exemplar → `flag: storm-escape-required` | Y |
| `INV-4-CODE` | CODE | INV-4 storm-escape-required | `storm-escape-required` | Step 5.5's audit applies INV-4's checkable question ("if this feature's fallback fires 100× in an hour, who hears about it, and via what counter?") to `_suppressed_count` incrementing with no threshold check → `invariant_findings` entry with `invariant="storm-escape-required"` | Y |
| `INV-5-PRD` | PRD | INV-5 no-lockstep-duplication | `no-lockstep-duplication` | G7's trigger-shape list fires on "duplicated lock-step logic": the row hand-copies `ALLOWED_EVENT_TYPES` instead of importing the canonical list → `flag: no-lockstep-duplication` | Y |
| `INV-5-CODE` | CODE | INV-5 no-lockstep-duplication | `no-lockstep-duplication` | Step 5.5's audit applies INV-5's checkable question ("does this feature copy logic, constants, or prompt text that must stay in agreement with another site?") to `webhooks/src/webhooks/validator.py`'s hand-copied list → `invariant_findings` entry with `invariant="no-lockstep-duplication"` | Y |

**Result: 10/10 match.** Every seeded violation flags with the correct
slug under the as-landed G7 and Step 5.5 text — acceptance met, no
rehearsal miss.

### Addendum — INV-6/INV-7 walk (2026-08-02)

Walked against the same as-landed G7 §"Design invariants pass" text
(which Reads the normative doc's checkable questions at run time — its
illustrative inline family-inventory row was updated to INV-1..7 in the
same change, since a hardcoded INV-1..5 row would be an INV-5 lock-step
copy contradicting the doc) and the Step 5.5 audit text:

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-6-PRD` | PRD | INV-6 status-matches-liveness | `status-matches-liveness` | G7's walk of the checkable question ("does any exit/bail path leave a task in a status whose implied owner is gone?") fires on the ABORTED-with-no-status-write row that names the sweep as its designed exit → `flag: status-matches-liveness` | Y |
| `INV-6-CODE` | CODE | INV-6 status-matches-liveness | `status-matches-liveness` | Step 5.5 applies the same question to the slot-releasing return with no successor-status write → `invariant_findings` entry with `invariant="status-matches-liveness"` | Y |
| `INV-7-PRD` | PRD | INV-7 holds-owned-and-bounded | `holds-owned-and-bounded` | G7's walk of the checkable question ("who exits it, what bounds it, where does an operator see it?") fires on the timeout-less `await` whose only exit is another component's action → `flag: holds-owned-and-bounded` | Y |
| `INV-7-CODE` | CODE | INV-7 holds-owned-and-bounded | `holds-owned-and-bounded` | Step 5.5 applies the question to the silent `return None` veto with no structured fact, counter, or deadline → `invariant_findings` entry with `invariant="holds-owned-and-bounded"` | Y |

**Addendum result: 4/4 match** (cumulative 14/14). No wording changes to
the gate/review text were needed — both consumers read the normative
doc's questions at run time.

### Addendum — INV-8 walk (2026-08-06)

Walked against the as-landed G7 §"Design invariants pass" text and the
Step 5.5 audit text, both re-read from the commit that landed them
(steps 1-2 of task 3779) rather than from the drafting context. Two G7
paths now exist for a PRD-shaped fixture and both were walked: the
normative path (G7's "walk every task in the batch against each
invariant's checkable question", which Reads the doc at run time and
auto-extended to INV-8 with no edit) and the no-invariants-file fallback
path (G7's trigger-shape list, which does NOT auto-extend and gained an
INV-8 entry in the same change — same reasoning as the 2026-08-02
family-row update above).

The same snapshot caveat applies: the Verdict column transcribes
phrasing as it read on 2026-08-06, not a live pin.

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-8-PRD` | PRD | INV-8 loop-thread-occupancy-bounded | `loop-thread-occupancy-bounded` | Normative path: G7's walk of the checkable question ("who bounds that collection's size?", "does the process spawn itself run on the loop thread?") fires — the row bounds `board.active_tasks` nowhere and its plain `def` called from the payload coroutine puts the fork/exec on the loop thread, while its own justification cites only the per-item ~30 ms, which is the per-item-cost-vs-occupancy confusion the rule names. Fallback path: the new trigger-shape entry ("a coroutine doing blocking or unbounded per-item work on the event-loop thread") fires on the same row → `flag: loop-thread-occupancy-bounded` | Y |
| `INV-8-CODE` | CODE | INV-8 loop-thread-occupancy-bounded | `loop-thread-occupancy-bounded` | Step 5.5 ("Read it and audit the modules in scope against each invariant's checkable question" — unchanged, Reads the doc generically) applies INV-8's question to `build_status_payload`: the `subprocess.run` is neither non-blocking nor offloaded, `board.active_tasks` is uncapped, and nothing is hoisted out of the body, so worst-case occupancy scales with board size → `invariant_findings` entry with `invariant="loop-thread-occupancy-bounded"`, `severity="high"` per Step 5.5's blast-radius classification | Y |

**Addendum result: 2/2 match** (cumulative 16/16). No wording change to
`docs/legibility/design-invariants.md`'s INV-8 checkable question, to
G7's walk instruction, or to Step 5.5 was needed *to make the walk
match* — it matched on its first pass. (A later review pass did narrow
the rule's fan-out limb, for over-firing rather than under-firing; see
the re-walk note below.) The one gate-text edit this change did require
(G7's trigger-shape entry) is not a rehearsal miss: that list is the
enumerated fallback for projects with no invariants file, so it never
auto-extends on any invariant addition.

Isolation re-checked while walking: no other trigger shape fires on
either INV-8 fixture — there is no fallback or suppressor (INV-4), no
copied constant (INV-5), no prose contract or undeclared tool envelope
(INV-1), no log-scrape of emitter-known facts (`--format=%cI` reads git
ground truth rather than re-deriving a fact a local emitter already held,
so not INV-2), and reading live git is corroboration rather than action
on a snapshot (not INV-3).

**Re-walked 2026-08-06 (review amendment).** After the walk above, review
narrowed INV-8's second limb and its checkable question: the fan-out
clause now reaches only per-item work that can block or is non-trivial,
over a collection not already bounded by an upstream contract
(pagination, a config cap, a fixed enum), and "already bounded upstream"
is now named as a complete answer — a scope fix against over-firing on
cheap non-blocking loops, not a rehearsal miss. Both rows were re-walked
against the narrowed text as landed and both still yield `Y`:
`INV-8-PRD` shells out per task at ~30 ms (blocking *and* non-trivial)
over "every task on the live board", which names no upstream bound, and
`INV-8-CODE` loops over a set its own comment marks `uncapped — the whole
live board` around a blocking `subprocess.run`. In both, limb 1 fires
independently of the narrowing — the call is neither non-blocking nor
offloaded. The narrowing is monotone (it can only shrink what INV-8
flags), so the other fixtures' single-invariant isolation is preserved a
fortiori. The verdicts and count above stand as written.

### Addendum — INV-9 walk (2026-08-24)

Walked against the as-landed G7 text (both paths: the normative walk,
which Reads the doc at run time and auto-extended to INV-9 with no edit;
and the trigger-shape fallback, which never auto-extends and gained an
INV-9 entry in the same change) and the Step 5.5 audit text. The same
snapshot caveat applies: the Verdict column transcribes phrasing as it
read on 2026-08-24, not a live pin.

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-9-PRD` | PRD | INV-9 one-fact-one-home | `one-fact-one-home` | Normative path: G7's walk of the checkable question ("which surface is the home, and do the others point at it rather than restate it?") fires — the row restates the ruling into three surfaces, names no home, and defers the question-owning record's update to "a later cleanup pass". Fallback path: the new trigger-shape entry ("a world-fact written into a second record/store as an independent copy, with no designated home, pointer discipline, or reconciler") fires on the same row → `flag: one-fact-one-home` | Y |
| `INV-9-CODE` | CODE | INV-9 one-fact-one-home | `one-fact-one-home` | Step 5.5 (unchanged — Reads the doc generically) applies INV-9's question to `record_ruling`: three independent prose copies, the question-owning record never written, nothing links the copies or names a winner → `invariant_findings` entry with `invariant="one-fact-one-home"`, `severity="warning"` | Y |

**Addendum result: 2/2 match** (cumulative 18/18). Isolation re-checked
while walking: no other trigger shape fires on either INV-9 fixture —
the copies are prose world-facts, not lock-step logic, constants, or
prompt text (`no-lockstep-duplication`'s scope; INV-9's own rule text
draws that seam explicitly), there is no fallback/suppressor (INV-4),
no log-scrape of emitter-known facts (INV-2), and the defect is writing
copies, not acting on a stale one (not INV-3).

### Addendum — INV-10 walk (2026-08-30)

Walked against the as-landed G7 text (both paths: the normative walk,
which Reads the doc at run time and auto-extended to INV-10 with no edit;
and the trigger-shape fallback, which never auto-extends and gained an
INV-10 entry in the same change) and the Step 5.5 audit text. The same
snapshot caveat applies: the Verdict column transcribes phrasing as it
read on 2026-08-30, not a live pin.

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-10-PRD` | PRD | INV-10 guards-exercise-behaviour | `guards-exercise-behaviour` | Normative path: G7's walk of the checkable question ("if what needs protecting is an instruction or a documented command, can it be EXECUTED against a fixture instead of matched?") fires — the row proposes a grep over a runbook for a command that is runnable against a fixture repo. Fallback path: the new trigger-shape entry ("a guard whose assertion target is prose/text rather than the behaviour it protects, where the thing could have been executed or mirrored") fires on the same row → `flag: guards-exercise-behaviour` | Y |
| `INV-10-CODE` | CODE | INV-10 guards-exercise-behaviour | `guards-exercise-behaviour` | Step 5.5 (unchanged — Reads the doc generically) applies INV-10's question to `test_runbook_derivation_is_task_scoped`: the assertion target is the command's spelling, the matcher is itself untested and excludes a placeholder the guarded doc actually uses, and a correctly-spelled command resolving the wrong commit passes → `invariant_findings` entry with `invariant="guards-exercise-behaviour"`, `severity="warning"` | Y |

**Addendum result: 2/2 match** (cumulative 20/20). Isolation re-checked
while walking: no other trigger shape fires on either INV-10 fixture —
the guard is not duplicated lock-step logic (`no-lockstep-duplication`
covers the copy, not the check over it), there is no fallback/suppressor
(INV-4), no log-scrape of emitter-known facts (INV-2), and nothing acts
on a stale snapshot (INV-3). INV-1 is adjacent by construction — INV-10
is INV-1 turned on the checks themselves — but INV-1's trigger shapes ask
whether a CONTRACT lives in prose, not whether a GUARD asserts on it.

### Addendum — INV-11 walk (2026-08-20)

Walked against the as-landed G7 §"Design invariants pass" text and the
Step 5.5 audit text, both re-read from the working tree at the commit that
promotes INV-11 (task 3803) rather than from the drafting context. Both G7
paths were walked, as in the 2026-08-06 addendum: the normative path
(which Reads the doc at run time and auto-extended to INV-11 with no edit)
and the no-invariants-file fallback path (the trigger-shape list, which
does NOT auto-extend and gained an INV-11 entry in the same commit).

The same snapshot caveat applies: the Verdict column transcribes phrasing
as it read on 2026-08-20, not a live pin.

| Fixture ID | Shape | Invariant | Expected slug | Verdict (as-landed text yields) | Match |
|---|---|---|---|---|---|
| `INV-11-PRD` | PRD | INV-11 no-silent-fail-soft | `no-silent-fail-soft` | Normative path: G7's walk of the checkable question ("what value does the caller receive, and can that value be told apart from the success value?") fires — the row's caller receives a count and a zero exit whether or not records were skipped, and the row's own justification offers the debug log as the answer, which the rule names as silent. Fallback path: the new trigger-shape entry ("a failure, fallback or partial enumeration whose result a caller cannot tell apart from success") fires on the same row → `flag: no-silent-fail-soft` | Y |
| `INV-11-CODE` | CODE | INV-11 no-silent-fail-soft | `no-silent-fail-soft` | Step 5.5 ("Read it and audit the modules in scope against each invariant's checkable question" — unchanged, Reads the doc generically) applies INV-11's question to `load_records`/`census`: the returned list is shorter and otherwise identical to a complete one, nothing in the return value or the exit code names the delta, and the `logger.debug` call is not a value any caller receives → `invariant_findings` entry with `invariant="no-silent-fail-soft"`, `severity="high"` per Step 5.5's blast-radius classification | Y |

**Addendum result: 2/2 match** (cumulative 22/22). No wording change to
G7's walk instruction or to Step 5.5 was needed — both read the normative
doc at run time, so INV-11's arrival is enough. The two gate-text edits
this change did require (G7's trigger-shape entry and the family-inventory
row) are not rehearsal misses: neither enumeration auto-extends, which is
why `scripts/tests/test_design_invariants_consistency.py` now fails the
moment a heading lands without them.

Isolation re-checked while walking: no other trigger shape fires on
either INV-11 fixture. There is no rate or streak dimension anywhere in
them — a single skipped record is already the defect — so INV-4
`storm-escape-required`, which asks who hears about the hundredth firing
of a path that legitimately degrades, does not apply; and no signal that
exists is being log-scraped or re-derived, so INV-2
`structured-facts-at-failure`, which constrains the SHAPE of a signal that
is emitted, does not apply either. The fixtures are deliberately seeded
against exactly that seam, which the normative doc's own "Family boundary"
paragraph draws.

## Reconciliation — 2026-07-14 base walk

Not required. The step-7 rehearsal above found no miss on its first
walk — all 10 fixtures already flagged with the correct slug against the
as-landed `skills/prd/references/gates.md` §G7 text and
`skills/review/references/phase2-architecture.md` Step 5.5 text. No
wording changes were made to `docs/legibility/design-invariants.md`,
`skills/prd/references/gates.md`, or
`skills/review/references/phase2-architecture.md`.

**Both later addenda did edit G7 — neither was a rehearsal miss.** The
2026-08-02 walk updated G7's illustrative family-inventory row (INV-1..5
-> INV-1..7) and its Calibration pointer; the 2026-08-06 walk added an
INV-8 entry to G7's trigger-shape list, extended the family row to
INV-1..8, and updated the Calibration pointer and "What it catches"
prose. That material is *enumerated* in gates.md rather than read from
the normative doc, so it never auto-extends on an invariant addition —
unlike G7's walk instruction and Step 5.5, which Read the doc at run time
and have needed no edit for any addition so far. Each addendum above
gives its own account.
