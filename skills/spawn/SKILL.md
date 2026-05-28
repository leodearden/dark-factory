---
name: spawn
description: "Spawn a new Claude Code CLI session in a new terminal window with a given prompt. Use when another skill (such as escalation-watcher) needs to hand a task off to a fresh interactive Claude session running in its own terminal — for example, spawning `/unblock <task>` so a human can drive it. Triggers on phrases like 'spawn a claude session', 'open a new terminal with claude', 'launch /unblock in a new window'. NOT for: running arbitrary shell commands in a new terminal, or invoking Claude inside the current session (use a sub-agent for that)."
---

# Spawn

A one-shot helper that opens a new terminal window and launches an independent `claude` CLI session inside it with a given prompt. The spawn happens via a background Bash task that **blocks until the spawned session exits**, so the caller can treat the background task's completion as a reliable "session finished" signal — even on emulators (gnome-terminal, konsole, macOS Terminal) whose launcher process normally detaches immediately. The spawned session is otherwise fully independent — separate terminal, separate Claude conversation, separate tool state.

Use this whenever a skill needs to hand a task off to a fresh interactive Claude — most commonly, escalation-watcher spawning `/unblock <task_id>` so a human can drive resolution in a dedicated window.

## Arguments

- **`prompt`** (required) — the literal string passed as the first positional argument to `claude`. May be a slash command (e.g. `/unblock 123`) or natural language. The caller is responsible for ensuring the prompt contains no unescaped single quotes; if it must, escape each one as `'\''` (close-single, escaped-single, open-single — the standard Bourne idiom). Example: `it'\''s fine` becomes a valid payload.
- **`cwd`** (required) — directory to `cd` into before invoking `claude`. Usually the project root. Must be an absolute path that exists on the host.
- **`skip_permissions`** (default `true`) — when `true`, passes `--dangerously-skip-permissions` so the spawned session runs without permission prompts. Set `false` when the spawned session should prompt for permissions normally (e.g. interactive exploration where the human wants oversight).
- **`terminal_title`** (optional) — passed via `--title` to emulators that support it (`gnome-terminal`, `konsole`, `xterm`; `kitty` ignores it harmlessly). Useful for distinguishing multiple spawned sessions in the window manager.

## Invocation

The skill ships with a helper script — `$DARK_FACTORY_ROOT/skills/spawn/spawn-claude.sh` — that handles emulator discovery, per-emulator wait-flag quirks (`gnome-terminal --wait`, naturally-foreground `xterm`/`kitty`), and a sentinel-based wait for detaching emulators (`konsole`, macOS `Terminal`). The background task this skill kicks off completes **when the spawned `claude` session exits**, and its exit code equals the spawned session's exit code — so the caller gets a reliable "session finished" signal without re-implementing any of that plumbing.

```
Bash(
  command="$DARK_FACTORY_ROOT/skills/spawn/spawn-claude.sh <cwd> <skip_perms> '<title>' '<prompt>'",
  run_in_background=true
)
```

Where:
- `<cwd>` is an absolute path that exists.
- `<skip_perms>` is the literal string `true` or `false`.
- `<title>` is the terminal-window title (pass `''` for none).
- `<prompt>` is the literal argument passed to `claude`. Wrap in single quotes; escape any inner single quote as `'\''`.
- `run_in_background=true` is essential — the script blocks until the session exits, which may be hours. Foreground would tie up the caller.

If `$CLAUDE_TERMINAL_CMD` (preferred) or `$ESCALATION_TERMINAL_CMD` (legacy) is set, the script honours it; otherwise it discovers an emulator in `$PATH`. **Do not** re-implement discovery in the skill — the script is the single source of truth.

## Verification

The background task's completion is a reliable signal that the spawned session has exited; its exit code is claude's own exit code. Script-level exit codes the caller should distinguish:

| Exit code | Meaning |
|-----------|---------|
| `0..125`  | The spawned `claude` session's own exit code |
| `126`     | No terminal emulator found — prompt the user for `$CLAUDE_TERMINAL_CMD` and retry. Suggest they export it in their shell profile for future sessions. |
| `127`     | Launcher failed (e.g. emulator binary errored before opening a window). Surface the error to the caller. |
| `2`       | Bad usage — caller bug, not user-recoverable. |

If you want to confirm the spawned session is alive mid-run, `ps -ef | grep claude` works. The background task itself is the canonical liveness signal — don't poll it via `TaskGet` in tight loops; just wait for its completion notification.

## When NOT to use

- **In-session work** — if the work belongs in your current conversation, do it directly or delegate to a sub-agent via the `Agent` tool. Don't spawn a new terminal just to get a sub-task done.
- **Non-Claude commands** — for arbitrary shell commands in a new window, call Bash directly with `run_in_background=true`. This skill is specifically for launching `claude` sessions.
- **Inside skills that forbid spawning** — `escalation-watcher-auto` explicitly prohibits any form of terminal spawning, including this skill. Respect the hosting skill's hard constraints.
