# merge_request bare-error diagnostic

**Status:** active — author 2026-05-28
**Repos touched:** `dark-factory` (orchestrator merge-queue, escalation server response surface)
**Origin:** hand-off note from `plans/occt-throttle-layer-2-3.md` (WP-D).

## Goal

When the orchestrator's merge queue fails a merge with the bare git error `not something we can merge` (or any similar low-level fatal), the failure surface returned to the `merge_request` caller (escalation MCP, the `/merge-queue` skill, or any operator-facing path) names the underlying diagnostic: which base SHA was attempted, whether it was the speculative base or actual `main` HEAD, whether the task branch ref resolved in the temp worktree, and the raw git stderr. Optionally, when a speculative-base attempt fails this way, retry once against actual `main` HEAD before reporting failure.

End-user observable outcome: on the next `merge_request` failure of this class, the operator reads a self-explanatory diagnostic and decides next steps without `/deb` + grep `merge_queue.py` archaeology — the loss of which on 2026-05-28 cost ~30 min and forced a documented `--no-verify` bypass.

## Background

During the Layer 1 OCCT throttle ship on 2026-05-28, two `mcp__escalation__merge_request` calls on branch `task/occt-throttle-layer-1` against base `main` returned the bare string `not something we can merge` with no further detail. The failure was almost certainly a `SpeculativeMergeWorker` race: the temp merge worktree was created at a base SHA from which the task branch wasn't reachable (the speculation built on a parent commit that wasn't `main`'s actual HEAD at merge time, or the task branch ref didn't resolve in the temp worktree). The operator had no signal to distinguish that hypothesis from any other cause and resorted to a `--no-verify` direct merge per `decision_merge_no_verify_with_orchestrator_live.md`.

The diagnostic gap is the bug. The retry is the affordance that would have made the operator-side path unblock without bypass.

Relevant code substrate (verified at authoring time):
- `orchestrator/src/orchestrator/merge_queue.py:1217` — `class SpeculativeMergeWorker`.
- `orchestrator/src/orchestrator/merge_queue.py:1896` — `_remerge` returns `SpeculativeItem` from a git-merge invocation.
- `orchestrator/src/orchestrator/merge_queue.py:1935` — `_verify_and_advance` consumes `SpeculativeItem`.
- `escalation/src/escalation/server.py:488` — `merge_request` endpoint; routes via `enqueue_merge_request`.

## Sketch of approach

Two tasks. The diagnostic improvement is the hard requirement; the retry is a soft follow-on.

### Task μ — diagnostic enrichment (hard)

At the point where a git-merge subprocess fails inside `SpeculativeMergeWorker._remerge` (or wherever the bare `not something we can merge` string originates today), capture and propagate to the `SpeculativeItem` failure path:
1. The base SHA the temp worktree was created at (full 40-char).
2. Whether that SHA is the speculative base (the assumed parent of preceding queued items) or actual `main` HEAD at the time of the merge attempt — emit both labels and the SHA.
3. The result of resolving the task branch ref in the temp worktree: SHA if resolved, `<unresolved>` if not (this directly disambiguates the suspected race).
4. The raw git stderr from the failing invocation.

The enriched diagnostic flows through `SpeculativeItem` → the `MergeRequest` failure path → the `merge_request` response surface (escalation server `server.py:488`) → the caller. Existing successful-merge paths are unchanged; only the failure response shape gains fields.

### Task ν — speculative-fail retry against actual main (soft)

When `_remerge` fails with the "not something we can merge" class (detected by stderr-string match against the known git output — the same class μ enriched) AND the base SHA used was NOT actual `main` HEAD at merge time, retry the merge once with the temp worktree rebuilt against actual `main` HEAD. If the retry succeeds, the operator sees a successful merge and a logged note explaining a speculative-base race was self-corrected. If the retry also fails, both attempts' diagnostics (per μ) are returned.

The retry covers the 2026-05-28 incident's exact failure shape without overshooting. Other merge failures (true conflicts, true non-ancestor branches, missing refs) get the enriched diagnostic but no retry — they're real failures, not races.

## Resolved design decisions

1. **Diagnostic-first, retry-second.** The diagnostic is the hard requirement because it makes every future incident self-explanatory; the retry is a convenience that silently resolves one specific race. Diagnostic-only would have unblocked 2026-05-28 (operator could have re-attempted manually); retry-only without the diagnostic would have hidden the race entirely and made the underlying mechanism opaque. Both are landed in this PRD, μ before ν.

2. **No new error class hierarchy.** Existing `SpeculativeItem` carries the failure summary today. Extend its existing failure-summary field (or add one structured field — implementer's call) rather than introducing a typed exception hierarchy for one failure mode. YAGNI until a second failure mode needs structured diagnostic.

3. **Retry attempts capped at 1.** If actual `main` HEAD also fails with "not something we can merge", the situation is no longer a speculation race — it's a real ancestry problem (e.g. the task branch was branched from a now-rewritten history). Retrying further would mask the real bug. One retry is enough to cover the race.

4. **Speculation race detection is stderr-string-based, not structural.** Git's exit code on "not something we can merge" doesn't distinguish it from other fatals; the string `not something we can merge` (the git porcelain phrase) is the signal. Documented as load-bearing in code comments — narrow exact match, don't paraphrase.

5. **The retry runs the SAME verify pipeline (β/γ env vars apply).** The retry produces a normal `SpeculativeItem` that flows through `_verify_and_advance` like any other merge; no special path. This is the whole point of having `_verify_and_advance` be the choke-point.

## Pre-conditions for activating

None. Substrate verified at authoring time (see Background).

## Cross-PRD relationship

No cross-PRD seams. Co-existence note: this PRD modifies the same `merge_queue.py` file that the OCCT throttle PRD's β/γ tasks (1533/1534) modify (`_resolve_verify_env` extension at verify.py + spawn-site changes at merge_queue.py:1935). Narrow-file-lock serialization will sequence these naturally; the changes are in different functions (μ/ν in `_remerge` and `SpeculativeItem` carry-through; β in `_verify_and_advance`'s call to `_resolve_verify_env`). Document the co-existence in μ's task description so the implementer doesn't conflict-merge with a concurrent β branch.

## Decomposition plan

- **μ — enrich `_remerge` failure diagnostic with base SHA, base label, branch-ref resolution, and raw git stderr.**
  - Repo: `dark-factory`.
  - Files: `orchestrator/src/orchestrator/merge_queue.py` (around `_remerge` line 1896 and `SpeculativeItem` at line 630); possibly `escalation/src/escalation/server.py` (line 488) if the response surface drops fields.
  - Implementation: capture base SHA before merge, resolve task branch ref in the temp worktree, run the merge, on failure attach all four diagnostic items to `SpeculativeItem`'s failure summary; propagate through `merge_request` response surface so the caller sees them.
  - Leaf signal (user-observable): integration test in `orchestrator/tests/` that simulates a `SpeculativeMergeWorker` base-SHA mismatch (constructs a SpeculativeItem with a base that doesn't include the branch) and asserts the resulting `MergeRequest` failure response contains all four diagnostic items: literal `base_sha=<sha>` (40-char), literal `base_label=speculative` or `base_label=main_head`, literal `branch_ref_in_worktree=<sha>` or `branch_ref_in_worktree=<unresolved>`, and the raw git stderr in a labelled field. Bonus signal: a manual repro of the 2026-05-28 incident (branching from a stale main) returns the enriched diagnostic.
  - Prereqs: none.

- **ν — retry against actual main HEAD when speculative-base merge fails with "not something we can merge".**
  - Repo: `dark-factory`.
  - Files: `orchestrator/src/orchestrator/merge_queue.py` (around `_remerge`).
  - Implementation: on a `_remerge` failure where (a) the stderr matches `not something we can merge` AND (b) the base SHA used was not actual `main` HEAD at merge time, rebuild the temp worktree from actual `main` HEAD and retry the merge once. If the retry succeeds, log a structured note `merge_retry_after_speculation_race` and proceed with the resulting `SpeculativeItem` through `_verify_and_advance`. If the retry also fails, both attempts' diagnostics (per μ) flow through to the response surface.
  - Leaf signal (user-observable): integration test simulates a speculation-race failure on the first `_remerge`, asserts the retry against `main` HEAD is invoked exactly once, succeeds, and produces a successful `MergeRequest` outcome with a `merge_retry_after_speculation_race` note in the response/log; a second test where the retry also fails asserts both diagnostics are returned and no further retries happen.
  - Prereqs: μ (the retry's failure path produces enriched diagnostics).

### Out of scope for this PRD

- Restructuring `merge_queue.py`'s `SpeculativeMergeWorker` to avoid the race entirely (e.g. always merging against `main` HEAD with no speculation). The race is real but the speculation buys throughput; redesigning the speculation is a separate larger PRD.
- A typed exception hierarchy for merge failures.
- Retrying more than once.
- Detecting other classes of git merge failure (true conflict, missing ref, etc.) and giving them custom retries.

## Open questions (tactical — defer to implementation)

1. **Failure-summary shape — single concatenated string vs structured fields on `SpeculativeItem`?** A structured dict (`{base_sha, base_label, branch_ref_resolution, git_stderr}`) is cleaner and the integration test asserts against it; a concatenated string is simpler but harder to test against. Suggested: structured dict added to `SpeculativeItem`; the existing string `summary` field is preserved as a human-readable rendering of the dict. Decide during μ.
2. **Where does the stderr-string match for ν live?** Inline in `_remerge`'s except path, or in a small `_is_speculation_race(stderr)` helper. Suggested: helper, so the match string lives in one place and gets a docstring noting it's a load-bearing exact match on git porcelain output. Decide during ν.
3. **Test fixture for the simulated speculation race.** Need a git repo with two branches where main has advanced past the speculated base and the task branch was branched from the speculated base — the implementer constructs this in the test setup. Suggested: a small helper in `orchestrator/tests/conftest.py` since the same fixture serves μ and ν. Decide during μ.
