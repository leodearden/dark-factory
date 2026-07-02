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
