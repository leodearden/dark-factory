# WP-F plan: curator `combine` safety

## Goal
Prevent `TaskInterceptor._execute_combine` from silently overwriting the wrong
task when the curator's LLM mis-identifies `target_id`. Keep the combine
affordance (we want it — it saves a full architect→implement→review cycle when
the candidate truly subsumes a pending task).

## Fingerprint strategy — verbatim title echo

Chosen: **ask the LLM to echo the target's current title into a new
`target_fingerprint` field**. Normalize (lowercase + collapse whitespace) before
comparing against the live target's title.

Why not a hash:
- The curator runs with `disallowed_tools=['*']` and `max_turns=3`. Any hashing
  would be done by the LLM itself from memory — more error-prone than copying
  a short string the prompt already shows verbatim (`_PoolEntry.render` puts
  `title: {self.title}` into the user prompt).
- Titles appear in the audit log too — human-readable is an advantage when
  triaging a blocked combine.
- If a title was concurrently renamed between corpus assembly and the write,
  aborting is the safe failure — we degrade to `create`, which is the cheap
  side of the tradeoff. The target module-lock eventually serializes anyway.

Normalization: `" ".join(title.strip().lower().split())`. Case and internal
whitespace are cheap to tolerate; everything else should match.

## Schema change

Extend `CURATOR_OUTPUT_SCHEMA` in `task_curator.py`:

```python
'target_fingerprint': {'type': ['string', 'null']},
```

Extend `CuratorDecision`:

```python
target_fingerprint: str | None = None
```

Propagate through `_parse_decision` (read `raw.get('target_fingerprint')`).

## Prompt update

Append to `_SYSTEM_PROMPT` (after the Output format section):

> When you choose "combine", you MUST also populate `target_fingerprint` with
> the VERBATIM title of the pool task you are targeting — copy the string
> immediately following `title:` from the pool entry whose id matches
> `target_id`. This fingerprint is verified against the live task before the
> rewrite is applied, so the wrong target is detected and the candidate is
> created fresh instead of silently clobbering an unrelated task. For "drop"
> and "create" leave `target_fingerprint` null.

## `_execute_combine` guard

Before calling `tm.update_task`, in order:

1. Fetch the live target via `tm.get_task(decision.target_id, project_root)`.
2. If fetch fails or returns no dict → log WARN, return `None` (fall-through).
3. Normalize `target.status` — if in `{'done', 'cancelled'}` → log WARN, return
   `None`.
4. Normalize `target.title` and `decision.target_fingerprint`; compare. On
   mismatch (including `None`) → log WARN with truncated values, return
   `None`.
5. Write the audit record to `data/combine_audit.jsonl` (append-only).
6. Call `tm.update_task` with the rewrite prompt.

All guard failures go to WARN not raise (consistent with existing best-effort
semantics). Returning `None` makes callers fall through to `create` which is
the desired degrade.

## Audit log

Path: `Path(os.getenv("DARK_FACTORY_DATA_DIR", "data")) / "combine_audit.jsonl"`.

- Append-only, no rotation.
- Parent dir created with `mkdir(parents=True, exist_ok=True)`.
- One JSON per line; no pretty printing.
- Fields: `ts, project_id, target_id, old (title, description_truncated, status),
  new (title, description_truncated), justification_truncated,
  curator_decision_id` (a fresh UUID4 — no existing field to reuse).
- Description truncation: first 500 chars.
- Write attempted best-effort; I/O errors logged as WARN but do not block the
  update (consistent with "audit is for recovery, not gate").

Actually — user briefing says "write to tamper-evident log so recovery is
possible". Make the audit write synchronous and before the update, so the
record exists even if `update_task` crashes mid-flight. I/O errors still log
WARN but proceed with the update (otherwise disk pressure blocks task merging,
bad trade).

## Tests to add (in `test_task_interceptor.py`)

Follow the `_mock_curator` fixture pattern. Use `tmp_path` + monkeypatch of
`DARK_FACTORY_DATA_DIR` to isolate the audit log per test.

1. `test_curator_combine_fingerprint_match_proceeds` — mock `get_task` to
   return `{title: 'Target Title', status: 'pending'}`; decision carries
   `target_fingerprint='Target Title'`; assert `update_task` called once and
   audit log has one line matching the decision.
2. `test_curator_combine_fingerprint_mismatch_aborts` — target title is
   `'Other Title'`; decision fingerprint `'Target Title'`; assert
   `update_task` NOT called, add_task IS called (fall-through), audit log is
   empty or missing.
3. `test_curator_combine_target_done_aborts` — target `status='done'`,
   fingerprint correct; assert abort, warn.
4. `test_curator_combine_target_cancelled_aborts` — same shape, status
   `'cancelled'`.
5. `test_curator_combine_audit_log_written_on_success` — (covered by #1 via
   the audit assertion).
6. `test_curator_combine_bulk_dedupe_respects_guard` — set up
   `_dedupe_bulk_created` with a curator decision that targets a `done` task;
   assert the new task is KEPT (not removed), update_task NOT called, and
   `remove_task` on the new task NOT called. (This exercises the integration
   path.)
7. `test_curator_combine_missing_fingerprint_aborts` — decision has
   `target_fingerprint=None`; assert abort + fall-through. Covers the "old
   LLM output missing the field" case.

### Test updates to existing tests

`test_curator_combine_updates_target_and_returns_id` and
`test_curator_combine_failure_falls_through_to_create` currently build a
`CuratorDecision` with no fingerprint. They will regress. Update them to
populate `target_fingerprint` matching the mocked `get_task` title
(`'Test Task'`).

The `add_subtask` combine test (if present) needs the same fix — check.

## Files touched

- `fused-memory/src/fused_memory/middleware/task_curator.py`
  - `CURATOR_OUTPUT_SCHEMA` (add `target_fingerprint`)
  - `CuratorDecision` dataclass (add field)
  - `_SYSTEM_PROMPT` (append combine-fingerprint instructions)
  - `_parse_decision` (plumb the field through)
- `fused-memory/src/fused_memory/middleware/task_interceptor.py`
  - `_execute_combine` (guard + audit log)
  - Add module-level helpers `_normalize_title`, `_extract_task_dict`,
    `_append_combine_audit`
- `fused-memory/tests/test_task_interceptor.py` — update 2 existing tests, add
  6-7 new tests

No changes to `_dedupe_bulk_created` — its `_execute_combine` call is
automatically protected once the guard lands.

## Out of scope (explicit)
- Removing combine.
- Human-loop escalation on combine.
- Backfilling audit for historical combines.
- Log rotation.

## Risk review
- Extra `tm.get_task` call per combine: adds one round-trip to taskmaster (a
  subprocess CLI call). Combine is rare (curator decides it maybe 1-5% of the
  time); the cost is acceptable vs. the safety gain.
- Audit log grows unbounded. Size it matters, but at ~1 KB/line and a few
  combines per day it will be < 1 MB/year. Revisit if rate climbs.
- LLM ignores the instruction → fingerprint None → guard aborts → fall-through
  to `create`. Safe degrade; we log WARN so this shows up if it becomes
  systemic.
