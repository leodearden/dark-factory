# Capability manifest — B3 low-risk auto-unblock hardening PRD

Per-leaf capability→evidence bindings (mechanized G3+G6). Built at decompose time 2026-06-04.
PRD: `plans/b3-low-risk-auto-unblock-hardening-prd.md`. Verdict: **no FAIL bindings — batch clear to queue.**

## T1 — sha stamping + proposal trim + config fields (orchestrator)

| Capability asserted by signal | Evidence | Verdict |
|---|---|---|
| Entries built parent-side (agent can't forge shas) | grep:dry_run_unblock.py:232-278 `_build_entry`; schema `additionalProperties: False` at :43 | wired |
| Capture point exists before agent invocation | grep:dry_run_unblock.py:166-180 `invoke_agent` call site in `run_dry_run_unblock` | wired |
| `git rev-parse HEAD`/`main` resolves in worktrees | worktrees share refs with primary checkout (standard git); `git rev-parse` already in `_ALLOWED_TOOLS` (dry_run_unblock.py:86) | wired |
| Append path for proposals | grep:dry_run_unblock.py:198-203 `update_task(..., append=True)` | wired |
| Trim can preserve sibling metadata | grep:scheduler.py:1571-1593 — `append=False` replaces whole blob ⇒ trim = read-modify-write of full metadata inside the single-writer parent coroutine; sibling-survival regression test is T1's own signal | wired (hazard noted) |
| Config home for new fields | grep:config.py:232 `UnblockAutoConfig` | wired |
| Test infra for dry_run_unblock | orchestrator/tests already patch it (e.g. test_workflow_e2e.py `_patch_dry_run_unblock`) | wired |

## T2 — b3_gate module (orchestrator)

| Capability | Evidence | Verdict |
|---|---|---|
| `_build_entry` output (with shas) feedable to boundary test | producer: **T1 upstream** (dep wired T2→T1); base shape exists today at dry_run_unblock.py:232 | producer-upstream, wired |
| `fcntl.flock` + `os.replace` atomicity | stdlib, linux host; tmp+rename is the established queue-writer convention | wired |
| Config yaml loading | existing pydantic config loader (orchestrator/config.py) | wired |
| `git diff <sha>..main -- <paths>` overlap probe | plain git; worktree shares refs | wired |
| State file home `data/escalations/` survives sweeps | current sweep glob is `esc-*.json` only (sweep.py:116, queue.py:70); PRD-1 D6 pins it permanently + regression assertion (gaps-prd1 entry 4) | wired (cross-PRD, acked) |

## T3 — unblock-low-risk SKILL.md rewrite (skills)

| Capability | Evidence | Verdict |
|---|---|---|
| `b3_gate check`/`charge` CLI exists & runnable via `.venv/bin/python -m orchestrator.b3_gate` | producer: **T2 upstream** (dep wired T3→T2); root .venv is editable to main src | producer-upstream, wired |
| `merge_request` / `release_workflow` / `resolve_issue` MCP tools | existing escalation MCP (current SKILL.md steps 1, 7, 8d) | wired |

## T4 — escalation-watcher SKILL.md B3 subsection + AFK shift 2 (skills)

| Capability | Evidence | Verdict |
|---|---|---|
| `b3_gate check`/`record-launch` CLI | producer: **T2 upstream** (dep wired T4→T2) | producer-upstream, wired |
| `attended_b3_enabled` config field | producer: **T1**, transitively wired via T4→T2→T1 | producer-upstream, wired |
| unblock-auto skill for drift re-investigation | exists: skills/unblock-auto/SKILL.md (verified 2026-06-04) | wired |
| Background sub-agent launches from watcher sessions | existing prose pattern (escalation-watcher/SKILL.md:211-224) | wired |
| `update_task(append=true)` reachable from watcher session | fused-memory MCP tool, listed | wired |

## Excluded signal (G6 guard)

"A fresh proposal merges via the live queue" is **not** a leaf signal: no unit-scope task can
deterministically produce a live merge. The merge path is mocked at the `merge_request` boundary
in T2's tests; the queue's post-merge re-rebase+verify is the register-declared existing
invariant, owned by nobody, re-proved by no one.
