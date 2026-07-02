---
name: warm
description: "Claim a warm, CoW-seeded git worktree for ad-hoc interactive work — 'put me in a warm worktree now' for work that doesn't go through /do. ONLY runs when the user explicitly types /warm — never auto-invoke it. Claims an _iact-* worktree on a task/<slug> branch via the escalation MCP's claim_warm_worktree, whose target/ build cache is seeded from the orchestrator's warm base so the first build recompiles near-zero; cold-falls-back to a plain EnterWorktree worktree (never errors) when the orchestrator/escalation MCP is unreachable or the claim fails."
argument-hint: "[slug — short task/<slug> branch name; omit to auto-derive]"
---

# /warm — claim a warm build worktree right now

`/warm` puts you into a git worktree whose build cache is already warm — for ad-hoc interactive work (exploring, a quick fix, a spike) that doesn't go through `/do`'s plan-and-hand-off flow. It claims an isolated `_iact-*` worktree on a `task/<slug>` branch whose `target/` is CoW-reflinked from the orchestrator's rolling warm base, so the first build in it recompiles close to nothing instead of paying a full cold build.

**What to claim:** $ARGUMENTS

> If the line above is empty, auto-derive a short slug (see step 1) rather than asking the user to supply one.

## Steps

### 1. Determine the slug

- If `$ARGUMENTS` was given, use it as the slug verbatim — it must be short and safe-charset: `[A-Za-z0-9][A-Za-z0-9._-]*`, no `..`.
- If empty, auto-derive a short slug of your own (e.g. a short topic word or timestamp token) that satisfies the same safe-charset rule.
- The resulting branch will be `task/<slug>` — the same convention `/merge-queue` expects, so this work can be merged the normal way later.

### 2. Determine `project_root`

This is the absolute path to the target project's **main checkout** — the same value that project's orchestrator config declares as `project_root` (see `/orchestrate`'s "identify the target project" step; `pwd` / `git rev-parse --show-toplevel` from the main checkout is normally enough — not a worktree path). Pass the canonical root: `claim_warm_worktree` validates it against the escalation server's wired `harness.git_ops.project_root` by resolved-path equality, so getting this wrong degrades to a clean `{error}` (handled below) rather than a crash.

### 3. Claim the warm worktree

```
mcp__escalation__claim_warm_worktree(
  slug="<slug>",
  project_root="<project_root>",
)
```

Pass `start_ref="<ref>"` only if the user asked to branch from something other than the current tip of `main`.

### 4. On success — enter it and report

Success returns `{path, branch, warm, base_ref}`:

- `cd` into `path` (an absolute path — a fresh `_iact-<slug>` worktree, not tracked by the Claude Code harness's own worktree bookkeeping).
- Surface to the user, plainly:
  - **branch** — `task/<slug>` (this is what `/merge-queue` will submit later).
  - **path** — where you now are.
  - **warm** — if `True`, the build cache was CoW-seeded from the orchestrator's warm base (first build recompiles near-zero); if `False`, the seed was skipped or failed and the worktree is a plain cold one — still fully usable, just without the head start.
  - **base_ref** — the commit SHA this worktree was branched from.

No test for this step — `SKILL.md` is a Claude-facing instruction file with no runtime harness to exercise (see plan analysis).

### 5. Failure handling — cold fallback is never silent

A claim can fail two ways: a **transport error** (the `mcp__escalation__claim_warm_worktree` call itself errors or times out — the orchestrator isn't running, or this project has no escalation MCP wired) or a **returned error dict** — `{error}` or `{error, reason}`. Both route to the same place: the cold fallback in step 6 below — never surface a raw exception, and never silently do nothing.

Read `reason` when present to decide whether a quick retry is worth it before falling back:

- **`interactive_worktree_limit`** — the project's `_iact-*` cap is reached. Either release an existing warm worktree you're done with (`mcp__escalation__release_warm_worktree`, step 7) and retry once, or go straight to the cold fallback.
- **`invalid_slug`** — the slug failed the safe-charset check. Retry once with a shorter/safe slug (`[A-Za-z0-9][A-Za-z0-9._-]*`, no `..`).
- **`git_failure`** — `start_ref`/`main` failed to resolve, or `git worktree add` itself failed. Surface the error and fall back — this isn't something a bare retry fixes.
- **No `reason` key** (bare `{error}`) — a standalone escalation server (no harness wired) or a `project_root` mismatch (wrong escalation endpoint for this project). Fall back.

### 6. Cold fallback — `EnterWorktree`

When warm isn't available, still get the user a usable worktree — just say so first, plainly:

> "Warm claim unavailable (**\<reason, e.g. orchestrator not running / interactive_worktree_limit\>**) — falling back to a cold worktree."

Then call `EnterWorktree`, naming it so the resulting branch follows the `task/<slug>` convention from step 1 (rename the branch afterward if `EnterWorktree`'s default doesn't already match it — mirrors `/do`'s own cold-worktree step). The session still ends up in a usable worktree on the right branch; it just pays a full cold build instead of a warm one. This path never errors out to the user — the orchestrator being down is a normal, handled condition, not a failure.

### 7. Release when done

An `_iact-*` worktree is a **raw** git worktree — the Claude Code harness does not track it (there's no `ExitWorktree` for it), so nothing removes it automatically when this session ends. When you're done with the work — merged via `/merge-queue`, or abandoning it — release it explicitly:

```
mcp__escalation__release_warm_worktree(
  path_or_branch="<path from step 4>",
  project_root="<project_root from step 2>",
)
```

Returns `{removed, path, branch, branch_pruned}` (+ `detail` when `removed` is False and something's worth surfacing). `removed=False` just means the worktree was already gone — the call is **idempotent**, so it's always safe to make even if the δ reaper already swept it up; treat it as a routine cleanup step, not one to skip on doubt. This is independent of `/merge-queue`, which is branch-based (`task/<slug>`) and lands the work regardless of whether the worktree that produced it still exists — the reaper is the safety net if this step is ever missed.

No test for this step — documentation deliverable (see plan analysis / design decisions).

## Quick reference

| Situation | Action |
|-----------|--------|
| Escalation MCP reachable, claim succeeds | `cd` into the returned `path`; report `branch`/`warm`/`base_ref` |
| Claim succeeds with `warm: false` | Still a success — usable cold worktree, just no cache head start |
| Escalation MCP unreachable (transport error) | Cold fallback: announce it, then `EnterWorktree` on `task/<slug>` |
| `reason: "interactive_worktree_limit"` | Release an idle warm worktree and retry once, or cold fallback |
| `reason: "invalid_slug"` | Retry once with a safe-charset slug |
| `reason: "git_failure"` | Surface the error, cold fallback |
| Bare `{error}` (no `reason`) | Standalone server or `project_root` mismatch — cold fallback |
| Done with the worktree | `release_warm_worktree(path_or_branch, project_root)` — idempotent, safe even if already reaped |
