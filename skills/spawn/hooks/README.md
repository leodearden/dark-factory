# Spawn hooks — SessionStart / Notification / Stop trio

Attention Rail T6 (`plans/session-attention-rail-prd.md` §4.6, §7 T6). These
three Claude Code hooks fire on **every** session — Leo's hand-launched
interactive sessions as well as ones started by `spawn-claude.sh` — and do
two things, fail-soft and fast:

1. **Register/refresh a session-registry record** (`orchestrator/src/orchestrator/session_registry.py`,
   Attention Rail T3) so hand-launched sessions show up in the registry too,
   not just programmatic spawns.
2. **Retitle the terminal tab** via an emulator-agnostic OSC escape sequence
   (`\033]0;<glyph> <title>\007`) so Leo-on-return can tell at a glance which
   windows are running, awaiting input, or idle.

## Layout

- `session-start.sh` / `notification.sh` / `stop.sh` — thin bash entrypoints
  wired into `~/.claude/settings.json`'s `SessionStart`/`Notification`/`Stop`
  hook events. Each reads the hook's JSON payload on stdin, invokes
  `orchestrator/src/orchestrator/session_hooks.py` by absolute path with
  `PYTHONPATH=<repo>/orchestrator/src` set (no venv/install required — the
  same trick `tests/scripts/test_spawn_claude.py` uses), and writes the
  emitted OSC retitle sequence to `/dev/tty`.
- `install-hooks.sh` — merges the trio into the real `~/.claude/settings.json`.
- `session_hooks.py` (in `orchestrator/src/orchestrator/`, not here) holds all
  the testable logic: identity/slug resolution, the registry read/refresh
  calls, the OSC/title helpers, and the settings-merge function. The bash
  files here stay thin by design (PRD §4 decision 9) — assert behavior
  against the Python module in `orchestrator/tests/test_session_hooks.py`,
  not against these scripts directly (except for the one bash-level
  integration test that exercises the PYTHONPATH + stdin-passthrough wiring).

## Absolute paths, not `~/.claude/hooks/`

Unlike the pre-existing `~/.claude/hooks/*.sh` scripts (`skim-rewrite.sh`,
`worktree-hookspath-*.sh`), these hooks live **in this repo**, version
controlled, and are referenced from `~/.claude/settings.json` by **absolute
path** into this checkout (e.g. `/home/leo/src/dark-factory/skills/spawn/hooks/session-start.sh`).
That is a deliberate PRD decision (§4 decision 5): registry/hook logic is
part of the dark-factory codebase, reviewed and tested like any other change,
not an ad-hoc file dropped in `~/.claude/`.

## MERGE, never clobber

`~/.claude/settings.json` already carries a populated `hooks` object
(`PreToolUse`: Bash → skim-rewrite.sh, EnterWorktree → worktree-hookspath-capture.sh;
`PostToolUse`: ExitWorktree → worktree-hookspath-restore.sh) plus other
top-level keys (`env`, `permissions`, `statusLine`, `enabledPlugins`, …).
Installing this trio must **add** the three new event keys
(`SessionStart`/`Notification`/`Stop`) to the existing `hooks` object and
leave every existing event, matcher, and top-level key byte-identical. This
is enforced by `session_hooks.merge_hook_settings()` (a pure, idempotent
dict-merge function, unit-tested with a JSON-diff assertion) — never by
hand-editing the real file.

## Fail-soft + fast, always

These hooks run on **every** SessionStart/Notification/Stop event across
every project. A bug here must never block a session start or a turn:

- `session-start.sh`/`notification.sh`/`stop.sh` run under `set +e` and
  always `exit 0`, mirroring `~/.claude/hooks/worktree-hookspath-capture.sh` /
  `worktree-hookspath-restore.sh`.
- For the `session-start`/`notification`/`stop` verbs, the Python side's
  `main()` wraps its dispatch in `try/except Exception`, logs loudly
  (`logger.error(..., exc_info=True)`), and always returns `0` — mirroring
  `session_registry.main()`.
- A missing `python3`, a corrupt registry record, or a registry-write failure
  degrades to "no retitle / no record this event" — never a hang or a
  non-zero exit propagated back to Claude Code.

`install-hooks.sh`/the `install` verb are the deliberate exception: that is a
human-invoked one-shot command, not a per-event hook, so `main()` instead
propagates exit code `1` on failure (see "Installing" below) — a failed
install must never be silently reported as a success.

## Installing

```bash
skills/spawn/hooks/install-hooks.sh
```

Idempotent: safe to re-run. Takes a timestamped backup of the current
`~/.claude/settings.json` (`settings.json.<UTC-timestamp>.bak`) next to it
before writing, and writes atomically (tmp file + `os.replace`) so a crash
mid-write can never leave `~/.claude/settings.json` truncated or corrupt.

Exits `0` on a successful merge, `1` if the install itself failed (e.g. a
permission error) — check `$?` in a non-interactive caller.
