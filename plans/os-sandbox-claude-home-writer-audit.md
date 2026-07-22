# `~/.claude` writer audit for sandboxed task agents (α1)

**Task**: 2903 (PRD `plans/os-sandbox-worktree-containment-prd.md`, task α1, decision
D5 / Open Q4). **Consumer**: α2 (`compute_write_set()` carve-out constants).

## Method

Static audit only — no dispatch was run. Three questions:

1. What is the complete, literal hook inventory wired in `~/.claude/settings.json`
   plus the spawn hooks it points at (`skills/spawn/hooks/`)?
2. Does a sandboxed task agent (implementer / debugger / SIMPLE_TASK, dispatched
   via `orchestrator/src/orchestrator/agents/invoke.py`, `CLAUDE_CONFIG_DIR`
   redirected to a per-task dir) actually **load** `~/.claude/settings.json` —
   i.e. are these hooks wired for it at all?
3. For each hook that is wired, does it **write** anywhere under `~/.claude`,
   and can a sandboxed task agent actually trigger it?

## Q2: does a task agent load `~/.claude/settings.json`? — **Yes**

`shared/src/shared/config_dir.py` (`TaskConfigDir._setup_symlinks`, lines 21-24,
40-49) **symlinks** `settings.json` and `settings.local.json` from `~/.claude/`
into the per-task `CLAUDE_CONFIG_DIR` at construction time:

```python
_SYMLINK_FILES = ['settings.json', 'settings.local.json']
...
def _setup_symlinks(self) -> None:
    for name in _SYMLINK_FILES:
        src = _HOME_CLAUDE / name
        dst = self._dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
```

Empirically confirmed on this very task's own redirected config dir
(`.task/claude-config-2903/`, `CLAUDE_CONFIG_DIR` for this session):

```
$ ls .task/claude-config-2903/
backups plugins policy-limits.json projects remote-settings.json
session-env sessions settings.json settings.local.json shell-snapshots
$ find .task/claude-config-2903 -iname "settings*.json"
.task/claude-config-2903/settings.local.json
.task/claude-config-2903/settings.json
```

`invoke.py:275-276` sets `CLAUDE_CONFIG_DIR` to this per-task dir for the
sandboxed dispatch sub-path; the symlink means the CLI resolves the **real**
`~/.claude/settings.json` content, so **every hook wired there is active for
task agents**, including the ones that reference other files by absolute path
(hook `command` values in `settings.json` are absolute paths, not
`CLAUDE_CONFIG_DIR`-relative — they resolve identically regardless of which
config dir loaded them).

`~/.claude/hooks/` itself (containing the hook scripts) is **not** in
`_SYMLINK_FILES` and has no per-task symlink — but every hook is invoked via
absolute path (`/home/leo/.claude/hooks/...`,
`/home/leo/src/dark-factory/skills/spawn/hooks/...`), so that's immaterial to
whether they fire.

## Q1/Q3: hook inventory + write analysis

`~/.claude/settings.json`'s `hooks` block, verbatim structure:

| Event | Matcher | Script | Writes under `~/.claude`? |
|---|---|---|---|
| `PreToolUse` | `Bash` | `~/.claude/hooks/skim-rewrite.sh` | No — `exec`s `skim rewrite --hook`, a stateless stdin→stdout command-rewrite filter (binary `strings` audit: `rewrite --hook` mode has no write path under `~/.claude`; the writing subcommands are `skim init`/`skim learn`, project-local `.claude/rules/*.mdc`, never invoked by this hook) |
| `PreToolUse` | `EnterWorktree` | `~/.claude/hooks/worktree-hookspath-capture.sh` | **Yes** — `$HOME/.claude/hooks/state/hookspath_<key>` (line 27-30) |
| `PostToolUse` | `ExitWorktree` | `~/.claude/hooks/worktree-hookspath-restore.sh` | **Yes** (transiently) — reads then `rm -f`s the same `$HOME/.claude/hooks/state/hookspath_<key>` file (line 18-41); net effect on that dir is delete, not create, but it does touch the path |
| `SessionStart` | `*` | `skills/spawn/hooks/session-start.sh` → `orchestrator/session_hooks.py session-start` | **Yes** — `session_registry.write_record()` → `~/.claude/fleet/sessions/<slug>/record.json` (`session_registry.py:470,480,559`) |
| `Notification` | `*` | `skills/spawn/hooks/notification.sh` → `orchestrator/session_hooks.py notification` | **Yes** — same `~/.claude/fleet/sessions/<slug>/record.json` via `refresh_record`/`write_record` |
| `Stop` | `*` | `skills/spawn/hooks/stop.sh` → `orchestrator/session_hooks.py stop` | **Yes** — same `~/.claude/fleet/sessions/<slug>/record.json` |

No other hook events are wired (`PreCompact`, `SubagentStop`, etc. are absent
from `~/.claude/settings.json`). `settings.local.json` carries no `hooks` key.

## Can a sandboxed task agent actually trigger the `EnterWorktree`/`ExitWorktree` hooks?

**No.** Hook firing for a `PreToolUse`/`PostToolUse` matcher requires the
matched tool call to actually be issued, which requires the tool to be in the
invocation's `--allowed-tools` set (`shared/src/shared/cli_invoke.py:1522-1523`
passes `allowed_tools` straight through as `--allowed-tools`, the CLI's
allow-list gate). Checked every sandboxed-eligible role's `allowed_tools` in
`orchestrator/src/orchestrator/agents/roles.py`:

- `IMPLEMENTER` (line 474, `sandboxed=True` at line 480): `['Read', 'Edit',
  'Write', 'Bash', 'Glob', 'Grep', ...MCP families...]` — no `EnterWorktree`/
  `ExitWorktree`.
- `DEBUGGER` (line 538, `sandboxed=True` at line 544): same shape, same
  absence.
- `SIMPLE_TASK` (line 1489; not yet `sandboxed=True` today — PRD task α3 adds
  it): `['Read', 'Glob', 'Grep', 'Edit', 'Write', 'Bash', ...MCP
  families...]` — same absence.

`EnterWorktree`/`ExitWorktree` are Claude Code's own native worktree-switching
tools, meant for an interactive session moving between worktrees. Task agents
are dispatched with `cwd` already pinned to their assigned worktree for the
invocation's lifetime and never need to switch — consistent with the tool
being absent from all three roles' allow-lists rather than merely unused.
Since the tool call can never be issued, the `PreToolUse(EnterWorktree)` /
`PostToolUse(ExitWorktree)` hook matchers can never match, so
`worktree-hookspath-capture.sh`/`-restore.sh` never run under sandboxed
task-agent dispatch, and `~/.claude/hooks/state/` is never touched by them.

By contrast, `SessionStart`/`Notification`/`Stop` are Claude Code **lifecycle**
hooks (matcher `*`) — not gated by `allowed_tools` at all — and fire
unconditionally for every session, including orchestrator-dispatched task
agents (`session_hooks.py:376-378`'s own docstring: "true for every
hand-launched session" describes the *first-sight* case, but the handler is
invoked identically for every `SessionStart`, spawned or not; see the dual-
record convergence note for `CLAUDE_SPAWN_SESSION_ID`-bearing spawned sessions,
same file lines 388-405). So `~/.claude/fleet/` is a **real, unconditional**
write target for sandboxed task agents; `~/.claude/hooks/state/` is not.

## Resolution: PRD Open Q4

> Drop `~/.claude/hooks/state/` from the writable set if task agents never
> load `~/.claude/settings.json` under the redirected config dir.

Task agents **do** load `~/.claude/settings.json` (Q2 above) — so the
literal premise as phrased doesn't hold — but the two hooks that write to
`~/.claude/hooks/state/` are gated on tools (`EnterWorktree`/`ExitWorktree`)
that are absent from every sandboxed role's `allowed_tools`, so those hooks
are provably unreachable from a sandboxed task-agent invocation regardless.
Net effect is the same as the question intended: **drop
`~/.claude/hooks/state/` from the writable set.** `~/.claude/fleet/` stays —
it's written unconditionally by lifecycle hooks that do fire for every task
agent.

## FINAL-WRITABLE-LIST:

- `~/.claude/fleet/`

`~/.claude/hooks/state/` is **excluded** — no sandboxed task-agent invocation
can trigger a write to it (the only two hooks that touch it require
`EnterWorktree`/`ExitWorktree`, neither of which is in `IMPLEMENTER`'s,
`DEBUGGER`'s, or `SIMPLE_TASK`'s `allowed_tools`). If a future change adds
either tool to a sandboxed role's `allowed_tools`, this list must be
re-audited before that role is dispatched under `sandbox.enabled: true`.
