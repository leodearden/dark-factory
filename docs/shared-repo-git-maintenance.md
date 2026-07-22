# Shared-Repo Git Maintenance

Operator note on why git's background auto-gc and auto-maintenance are
**disabled** on orchestrator-managed shared repos (e.g. the Dark Factory
main repo at `project_root`), and why gc ownership is now the operator's /
orchestrator's out-of-band responsibility. Applies under the OS-sandbox
worktree-containment model — see
`plans/os-sandbox-worktree-containment-prd.md` task **α5** (the **D2**
narrow-shared-`.git`-write-set corollary).

## What is disabled

On every orchestrator-managed shared repo the orchestrator sets, repo-locally
(in `.git/config`):

| Key                | Value   | Effect                                           |
|--------------------|---------|--------------------------------------------------|
| `gc.auto`          | `0`     | Disables automatic `git gc --auto` runs          |
| `maintenance.auto` | `false` | Disables the automatic `git maintenance` trigger |

Both are set **idempotently** (`git config` overwrites in place), at two
sites, so the setting is present before the first dispatch and reasserted on
any config drift or re-clone:

1. **Orchestrator startup** — `Harness.run()` calls
   `GitOps.disable_shared_repo_auto_maintenance()` as an early best-effort
   step (right after the singleton lock), so the config is in place before any
   task is dispatched.
2. **Worktree-create path** — `GitOps.create_worktree()` reasserts it (right
   after the idempotent `core.hooksPath` block), covering every dispatch.

The write is **best-effort/loud**: a non-zero `git config` return code is
logged at `WARNING` but never raised, and the startup call is additionally
wrapped so a git fault can never block orchestrator startup or a dispatch. A
failure merely leaves auto-gc enabled — itself only a benign-but-noisy
failure — rather than crashing the orchestrator (loud-over-silent-degradation).

## Why

Under the OS-sandbox model the shared `.git` write-set is deliberately
**narrow**: task lanes may write `objects/`, `worktrees/<name>/`,
`refs/heads/task/`, and `logs/refs/heads/task/`, but the `.git` root,
`packed-refs`, `refs/heads/main`, and all non-task refs stay **read-only**
(PRD D2). Background auto-gc/maintenance wants to rewrite `packed-refs` and
other read-only paths, so — were it left enabled — it would fire during an
ordinary `git commit` and **fail benignly but noisily** (EROFS / gc warnings
cluttering task output), even though the commit itself succeeds.

Disabling it removes that noise (enforcement-matrix row 12: `git commit` with
auto-maintenance disabled → succeeds with zero gc/EROFS noise). The trade-off
is that repository maintenance no longer happens on its own.

## gc ownership is now out-of-band

Because auto-gc no longer runs, packing/gc of an orchestrator-managed shared
repo is the **operator's / orchestrator's** responsibility, scheduled
out-of-band (e.g. a cron/systemd timer running `git gc` — or `git maintenance
run` — while no task lane holds the narrow read-only write-set, i.e. against a
repo with full write access). This is a deliberate ownership move, not an
oversight.

## Verification

After orchestrator startup, on the shared repo:

```bash
# git ≥ 2.46 subcommand form (the PRD signal):
git config get gc.auto            # → 0

# Portable form (all git versions):
git config --get gc.auto          # → 0
git config --get maintenance.auto # → false
```

`gc.auto` reading `0` on the DF main repo after startup is the user-observable
α5 signal.

## See also

- `plans/os-sandbox-worktree-containment-prd.md` — task α5 (Shared-repo git
  maintenance discipline) and decision D2 (narrow shared-`.git` write-set).
- `orchestrator/src/orchestrator/git_ops.py` —
  `GitOps.disable_shared_repo_auto_maintenance()` and its `create_worktree()`
  call site.
- `orchestrator/src/orchestrator/harness.py` — the `Harness.run()` startup
  call.
