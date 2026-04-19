# WP-F: Curator `combine` safety

## Mission
Prevent the curator's `combine` decision from silently overwriting the wrong task, while preserving the affordance (which reduces token-spend on micro-task architecting).

## Defect context
When the task curator decides `action=combine`, `TaskInterceptor._execute_combine` calls Taskmaster's `update_task(target_id, prompt="Replace this task with EXACTLY the following fields, verbatim…")`. If the LLM curator mis-identifies `target_id`, an unrelated legitimate task gets its content replaced — silent destructive rewrite.

A reify steward on 2026-04-17 observed (likely) this very scenario: attempted to add a prerequisite task, saw task 1959's content overwritten with the new payload. Curator logs in that window showed `decision=create` (not combine) — so combine wasn't the direct culprit for that incident — but the hazard is live and the exact mechanism fits the observation.

## Design decision (agreed: harden, do not remove)
Add a **fingerprint verification** gate before `_execute_combine` writes: the curator's LLM output must include a short identifying quote of the TARGET task's current state, and we verify it against the live tasks.json before rewriting. If mismatched, abort and degrade to `create`.

Plus status guards: never combine into a task that is `done` or `cancelled`, or (stretch) actively being worked on.

Plus audit log: before overwriting, write the old content to a tamper-evident log so recovery is possible.

## Scope

### 1. Schema change
In `fused-memory/src/fused_memory/middleware/task_curator.py`:
- Extend `CURATOR_OUTPUT_SCHEMA` to include a new field `target_fingerprint` (string, required when `action=="combine"`, else null/absent).
- Recommended shape: a hash (e.g. sha256 prefix, 16 hex chars) of `target.title + "\n" + target.description[:200]` — the LLM computes this from what it was shown, we re-compute from live data to verify.
- Alternatively (simpler for the LLM): ask the LLM to **copy** the target's current title verbatim into `target_fingerprint`. We then normalize and compare. This is easier for the LLM to produce reliably but brittle if titles get edited concurrently.
- **Choose the simpler "verbatim title echo" approach** unless you have a strong reason otherwise — document your choice in the plan.

### 2. Prompt update
The curator's system prompt (look for where the curator prompts are loaded — probably `fused-memory/src/fused_memory/reconciliation/prompts/` or inline in `task_curator.py`) must instruct the LLM to populate `target_fingerprint` whenever it chooses `combine`. Explain why (prevents clobbering unrelated tasks).

### 3. Pre-write verification in `_execute_combine`
In `fused-memory/src/fused_memory/middleware/task_interceptor.py` (method `_execute_combine`, currently at ~line 471):
- Before calling `tm.update_task(...)`, fetch the current state of `target_id` via `tm.get_task(target_id, project_root)`.
- Compute the expected fingerprint from the live target and compare with `decision.target_fingerprint`.
- If **mismatch**: log WARN with both fingerprints truncated, return `None` (which causes the caller to fall through to `create`). Don't raise.
- If **target.status in {'done', 'cancelled'}**: log WARN, return `None`.
- If all checks pass, proceed with the update.

### 4. Audit log
Before the combine update succeeds, append a line to `data/combine_audit.jsonl` (create if missing) with:
```json
{
  "ts": "...",
  "project_id": "...",
  "target_id": "...",
  "old": {"title": "...", "description_truncated": "...", "status": "..."},
  "new": {"title": "...", "description_truncated": "..."},
  "justification_truncated": "...",
  "curator_decision_id": "..."
}
```
- Path resolver: `Path(os.getenv("DARK_FACTORY_DATA_DIR", "data")) / "combine_audit.jsonl"`. Use whatever the codebase's existing data-dir convention is — search for other `.jsonl` writers to follow precedent.
- Append-only, no rotation in this WP. If this grows unbounded, that's a future cleanup.

### 5. The `_dedupe_bulk_created` path
`_dedupe_bulk_created` (task_interceptor.py ~line 521) also calls `_execute_combine`. The hardened `_execute_combine` automatically protects this path — verify no additional changes needed there. But add a test covering it.

## Files
- `fused-memory/src/fused_memory/middleware/task_curator.py` — schema + prompt text
- `fused-memory/src/fused_memory/middleware/task_interceptor.py` — `_execute_combine` (~line 471), possibly `_dedupe_bulk_created`
- Curator prompt source file — locate it; might be inline or under `reconciliation/prompts/`
- Tests under `fused-memory/tests/` — look for existing curator/combine tests and follow pattern

## Tests to add
1. **Fingerprint match → combine proceeds.** Mock `tm.get_task` to return a target whose title matches the curator's fingerprint; assert update_task is called.
2. **Fingerprint mismatch → abort, returns None.** Mock target with different title; assert update_task NOT called, log warning emitted, audit log NOT written.
3. **Target status = done → abort.** Assert no update, warn logged.
4. **Target status = cancelled → abort.** Same.
5. **Audit log written on success.** Assert `combine_audit.jsonl` contains the expected JSON record after a successful combine.
6. **Bulk-dedupe path respects the guard.** Feed `_dedupe_bulk_created` a scenario where the curator picks a done task as combine target; assert it falls through to keep the task rather than clobbering.

Don't invent new test infrastructure — follow existing `fused-memory/tests/` patterns.

## Out of scope
- Removing combine entirely (user explicitly wants it kept).
- Human-loop / escalation on combine decisions.
- Changing drop / create logic.
- Backfilling audit log for historical combines.

## Acceptance
- [ ] Curator emits `target_fingerprint` for every `combine` decision.
- [ ] `_execute_combine` verifies fingerprint + status guards before writing.
- [ ] Audit log written for every successful combine.
- [ ] All existing tests pass.
- [ ] New tests (mismatch, status guards, audit, bulk path) pass.
- [ ] No regression in drop/create paths.

## Dependencies
- **WP-A must be complete** (healthy fused-memory for testing).
- Can run in parallel with WP-E. If merging after WP-E, no expected conflicts (different files).

## Workflow for this session

1. Read this briefing.
2. Search fused-memory for prior curator context: `mcp__fused-memory__search` with "task curator combine", "curator decision fingerprint" (if any prior memory exists).
3. Read the relevant source:
   - `fused-memory/src/fused_memory/middleware/task_curator.py` (full)
   - `fused-memory/src/fused_memory/middleware/task_interceptor.py` — `_execute_combine` and `_dedupe_bulk_created`
   - Find and read the curator's prompt text (grep for recognizable phrases like "curator" or "drop.*combine.*create" in prompts/)
4. Write `plans/wp-f-plan.md`: which fingerprint strategy you chose and why, exact schema changes, prompt modifications (quote the new prompt text), test list.
5. Create branch `wp-f/curator-combine-safety` off main.
6. Implement. Small commits welcome.
7. Run `cd fused-memory && uv run pytest tests/ -x`. Fix until green.
8. Run the new tests specifically; confirm they cover each acceptance item.
9. Final: `/reflect`. Save notes on: LLM-output schema constraints, any latency impact from the extra `get_task` call, whether the fingerprint approach feels robust.
10. Emit JSON summary:
    ```json
    {
      "wp": "F",
      "branch": "wp-f/curator-combine-safety",
      "commits": ["<sha>", ...],
      "fingerprint_strategy": "title-echo|hash|...",
      "tests_added": [...],
      "tests_passed": N,
      "tests_failed": 0,
      "audit_log_path": "data/combine_audit.jsonl",
      "reflection_saved": true,
      "notes": "..."
    }
    ```

## Permission mode
`--dangerously-skip-permissions`. Do NOT push, modify git config, merge to main, or restart fused-memory systemd.
