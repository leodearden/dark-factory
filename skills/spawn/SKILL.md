---
name: spawn
description: "Spawn a new Claude Code CLI session in a new terminal window with a given prompt. Use when another skill (such as escalation-watcher) needs to hand a task off to a fresh interactive Claude session running in its own terminal — for example, spawning `/unblock <task>` so a human can drive it. Triggers on phrases like 'spawn a claude session', 'open a new terminal with claude', 'launch /unblock in a new window'. NOT for: running arbitrary shell commands in a new terminal, or invoking Claude inside the current session (use a sub-agent for that)."
---

# Spawn

A one-shot helper that opens a new terminal window and launches an independent `claude` CLI session inside it with a given prompt. The spawn happens via a background Bash task that **blocks until the spawned session exits**, so the caller can treat the background task's completion as a reliable "session finished" signal — even on emulators (gnome-terminal, konsole, macOS Terminal) whose launcher process normally detaches immediately. The spawned session is otherwise fully independent — separate terminal, separate Claude conversation, separate tool state.

Use this whenever a skill needs to hand a task off to a fresh interactive Claude — most commonly, escalation-watcher spawning `/unblock <task_id>` so a human can drive resolution in a dedicated window.

## Resolving the prompt for a fresh context

The spawned session begins with an **empty conversation**. Whatever prompt you pass becomes its first and only context. So the prompt must **stand on its own there** — anything that only means something *here* (in this conversation) is meaningless once it lands in the fresh session.

Before you build the script invocation, run this decision procedure on the prompt. **Default to the least transformation** — resolve only when there is a genuine unresolved reference:

1. **Already self-contained → pass verbatim.** A slash command whose argument fully specifies the work: `/unblock 8888`, `/review`, an absolute path, a quoted literal. Change *nothing*. This is the common case, and the only case programmatic callers (e.g. escalation-watcher) ever hit. **Never "enrich" a self-contained command** — doing so risks corrupting a prompt that was already correct.

2. **Contains a contextual reference → resolve, then rewrite, keeping the command.** Deixis or anaphora (`that`, `this`, `it`, `above`, `the fix`, `the approach we discussed`) or a bare instruction that only parses given this conversation. Replace each reference with the concrete thing it denotes — drawn from *this* conversation, rendered unambiguously — and keep any leading slash command so the target skill still runs in the fresh session. `/do that` → resolve `that`; `/prd that feature` → resolve `that feature`.

For two commands the resolution is richer than swapping a pronoun, because the fresh session is **autonomous** and cannot ask you the questions it would normally ask:

### `/do` — distill here, execute autonomously there

`/do`'s real work is compressing this session into a self-contained plan. The spawned session can't do that (the session is gone), so **do it here**, then hand the spawned session pure execution:

- Produce the plan exactly as the `/do` skill body prescribes: **Objective**, **Decisions & rationale** (including the alternatives you rejected and *why*, so the executor doesn't relitigate them), **Implementation** (name real files/functions), **Verification**, and the **Execution protocol** (worktree → `/merge-queue` → `/reflect`, work autonomously).
- **Do not spawn `/do <plan>`.** A fresh `/do` would enter plan mode and stall forever waiting for a human to choose *"Clear Context and Follow Plan."* Instead spawn a plain execution prompt: hand over the plan and tell the session to carry it out end-to-end without pausing. This is the autonomous-executor path.
- If the plan is more than a few lines, **write it to a file** (Write tool, to an absolute path *outside* the repo working tree — e.g. under `~/.claude/spawn-briefs/` or a `mktemp` path — so the worktree the spawned session creates can't shadow it and it can't be accidentally git-staged) and spawn a short prompt pointing at it. Passing a multi-paragraph plan as one shell-quoted CLI arg is fragile. e.g. prompt → `Read the plan at <abs-path> and execute it end-to-end, autonomously. It is the contract — don't pause for confirmation unless you hit something genuinely blocking the plan doesn't cover.` with `skip_permissions=true`.

### `/prd` — resolve the subject and pre-answer the gates

- Resolve the contextual subject (`that feature`, `the thing we discussed`) into a concrete PRD subject.
- `/prd`'s value is its G1–G6 gates, and a fresh autonomous session can't ask you the questions those gates raise. So **fold in everything from this conversation that bears on them** — the named consumer (G1), the user-observable leaf signal (G2), substrate assumptions (G3), cross-PRD seams (G4), the premise (G6) — so the gates can be satisfied without interrogating an absent human.
- **Keep the `/prd` wrapper** (unlike `/do`): spawn `/prd <concrete subject + folded-in context>` so the gates run in the fresh session (file-spill as above if long). `/prd` is high-stakes and may still legitimately pause at a gate it genuinely can't resolve from the brief — that's acceptable.

When in doubt about whether a prompt is self-contained, prefer leaving it untouched and noting the ambiguity, rather than rewriting and risking corruption.

## Arguments

- **`prompt`** (required) — the string passed as the first positional argument to `claude`. May be a slash command (e.g. `/unblock 123`) or natural language. **First resolve it per [Resolving the prompt for a fresh context](#resolving-the-prompt-for-a-fresh-context) above** — a contextual prompt typed in an interactive session must be rewritten to stand alone *before* it becomes this argument. The caller is responsible for ensuring the final string contains no unescaped single quotes; if it must, escape each one as `'\''` (close-single, escaped-single, open-single — the standard Bourne idiom). Example: `it'\''s fine` becomes a valid payload.
- **`cwd`** (required) — directory to `cd` into before invoking `claude`. Usually the project root. Must be an absolute path that exists on the host.
- **`skip_permissions`** (default `true`) — when `true`, passes `--dangerously-skip-permissions` so the spawned session runs without permission prompts. Set `false` when the spawned session should prompt for permissions normally (e.g. interactive exploration where the human wants oversight).
- **`terminal_title`** (the target convention for programmatic callers; optional for ad-hoc interactive use) — passed via `--title` to emulators that support it (`gnome-terminal`, `konsole`, `xterm`; `kitty` ignores it harmlessly). Distinguishes multiple spawned sessions in the window manager, and is load-bearing for Leo-on-return and the (future) attention-manifest to identify sessions at a glance without opening each window. Follow the convention `<role>:<project>#<task-id> <short-slug>` — `<role>` is the spawned skill/command (`unblock`, `review`, `prd`, ...), `<project>` is the project_id, `#<task-id>` is included when the spawn is task-scoped (omit it for project-level spawns), and `<short-slug>` is a few hyphenated words summarizing the work. Examples: `unblock:df#2085 routing-mechanism`, `prd:df attention-rail`. Programmatic callers (e.g. escalation-watcher) should pass a convention-shaped title going forward — this is the convention new and updated call sites follow, not yet a property every existing caller satisfies. Only a genuinely ad-hoc interactive spawn may omit a title entirely.

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
- `<title>` is the terminal-window title — programmatic callers should follow the `<role>:<project>#<task-id> <short-slug>` convention documented in [Arguments](#arguments) above (existing call sites are migrating to it; not all have yet). Pass `''` only for a genuinely ad-hoc interactive spawn with no meaningful title.
- `<prompt>` is the literal argument passed to `claude`. Wrap in single quotes; escape any inner single quote as `'\''`.
- `run_in_background=true` is essential — the script blocks until the session exits, which may be hours. Foreground would tie up the caller.

If `$CLAUDE_TERMINAL_CMD` (preferred) or `$ESCALATION_TERMINAL_CMD` (legacy) is set, the script honours it; otherwise it discovers an emulator in `$PATH`. **Do not** re-implement discovery in the skill — the script is the single source of truth.

## Verification

The background task's completion is a reliable **liveness** signal: it tells you the spawned session's process is gone, present vs. died-silent. As of Attention Rail T5, exit codes are documented as **liveness-only** — they are not, and should not be treated as, the semantic outcome channel. That channel is an explicit file the session writes: see [Result-handback (result.md)](#result-handback-resultmd) below. All codes below are still returned unchanged (nothing here is a breaking change); only their meaning narrows to "was the process alive, and how did it stop," not "did the work succeed." Script-level exit codes the caller should distinguish:

| Exit code | Meaning |
|-----------|---------|
| `0..125`  | The spawned `claude` session's own exit code. Liveness only — a `0` means the process exited cleanly, not that the work succeeded or is complete; check `result.md` for the outcome. |
| `126`     | No terminal emulator found — prompt the user for `$CLAUDE_TERMINAL_CMD` and retry. Suggest they export it in their shell profile for future sessions. |
| `127`     | Launcher failed to start the session (the emulator binary errored before running the payload — no sentinel was ever written). Surface the error to the caller. |
| `129`     | Terminal window closed while the session was still alive (SIGHUP reached the running session) — **or** a session that had *already finished cleanly* raced its own window teardown and still surfaced as 129 (see "The 129-on-clean-exit race" below). Inconclusive by construction: never read `129` as "the session failed" or "the work was lost" — read `result.md`. |
| `144`     | Claude never started — no new transcript appeared under `~/.claude/projects/<encoded-cwd>/` and no `claude` process was detected within the started-grace window; the background started-watchdog marked the session-registry record `failed-to-start` and emitted a loud caller-visible line on the spawn's stderr. This is the silent-no-transcript hang (2026-07-06 incident), now surfaced within grace instead of hanging. Additive; all existing codes retain their meaning. |
| `2`       | Bad usage — caller bug, not user-recoverable. |

The transcript path used by the `144` check encodes the session's `cwd` by replacing every `/` and `.` with `-`, matching `session_registry.transcript_path_for_cwd` byte-for-byte — e.g. `/home/leo/src/dark-factory` → `~/.claude/projects/-home-leo-src-dark-factory/`. The started-grace window defaults to ~90s and is tunable via `$SPAWN_STARTED_GRACE_SECS`.

**The 129-on-clean-exit race.** `129` is not a reliable "killed mid-session, work possibly lost" signal, even though that's what it originally meant. `spawn-claude.sh`'s payload arms an `EXIT` trap (captures `claude`'s real exit code via `${ec:-$?}`) and a `HUP` trap (converts a window-close SIGHUP into `exit 129`) before invoking `claude`. A **clean** exit — e.g. the user hits Ctrl-C (or the session simply finishes) at almost the same moment the terminal window is torn down (the emulator closing the window right as the payload shell is unwinding) — can still race the HUP trap: it wins the race and overwrites the EXIT trap's capture of the real, already-successful exit code, so a session that completed its work cleanly surfaces as `129` at the parent anyway. This has been observed in practice, not just theorized. The fix is not to chase a tighter race window; it's to stop reading exit codes for outcome at all — that's exactly what `result.md` exists for.

**Known limitations of the `144` check:**
- **Concurrent same-cwd spawns.** The transcript directory is keyed on `cwd`, not on the individual session — Claude Code writes every session for a given `cwd` into the same directory. If a second, genuinely healthy spawn for the same `cwd` writes its own new transcript while a sibling spawn is truly failing to start, the healthy sibling's file is indistinguishable from the failing spawn's own evidence, so the failing spawn is never flagged (a false negative). The detector is exact for the common single-spawn-per-cwd case, including the motivating 2026-07-06 incident, but treat coverage as reduced for a concurrent same-cwd fleet.
- **Detached launchers can false-flag a live-but-slow session.** For a detached launcher (`konsole`, or a custom `$CLAUDE_TERMINAL_CMD`), the transcript probe is the *only* positive evidence — a live `claude` process can't be observed once its launcher detaches. A real session that is merely slow to write its first transcript (heavy load, cold cache) can still be flagged `144` even though it is alive and keeps running detached. Treat a `144` on a detached launcher as "no evidence seen within grace," not as confirmation the session is dead, and raise `$SPAWN_STARTED_GRACE_SECS` for callers on slow or loaded hosts.

If you want to confirm the spawned session is alive mid-run, `ps -ef | grep claude` works. The background task itself is the canonical liveness signal — don't poll it via `TaskGet` in tight loops; just wait for its completion notification.

## Result-handback (result.md)

Exit codes (above) only tell you the process is gone and roughly how it stopped — never what happened. The semantic outcome channel is an explicit file the spawned session writes before it ends:

- `spawn-claude.sh` allocates `<record-dir>/result.md` — the same session-registry record directory captured as `SESSION_RECORD_DIR` — and exports its path into the spawned session's own environment as `$CLAUDE_SPAWN_RESULT_FILE`. The identical path is stored in the session-registry record's `result_file` field, so a parent reading the record never has to recompute or guess the path.
- The prompt handed to the spawned session gets a standard trailer appended, asking it to write — before ending, whether it finishes, hands off, or gets blocked — an `outcome` (`done|blocked|abandoned|handed-off`), `changed` (commits/branches/task ids touched), and `action_needed` (what a human or parent should do next) as a small structured markdown header, followed by a few sentences of prose context.
- This is **best-effort**: nothing in `spawn-claude.sh`, and nothing the trailer asks for, blocks the session's own exit. A parent joining a completed spawn (e.g. escalation-watcher on a `/spawn` background task's completion) should read `record.result_file` (equivalently `<record-dir>/result.md`) for the authoritative outcome, and fall back to exploring the worktree/task only when the file is absent, empty, or unparsable.
- **Fail-soft, matching the registry's own fail-soft contract:** if the session-registry `launching` write itself faults (missing `python3`, an unwritable fleet root, etc.), `SESSION_RECORD_DIR` is empty and both the env export and the prompt trailer are cleanly skipped — no bogus `/result.md` path is ever exported, referenced, or created. The exit-code contract above is unaffected either way.
- A future `Stop`/`SessionEnd` hook (Attention Rail T6) may add a fallback stub-write for a session that skips the trailer's instruction, but this protocol does not depend on T6 landing — the prompt trailer plus the session's own best-effort write is already a complete signal, if occasionally missed.

## When NOT to use

- **In-session work** — if the work belongs in your current conversation, do it directly or delegate to a sub-agent via the `Agent` tool. Don't spawn a new terminal just to get a sub-task done.
- **Non-Claude commands** — for arbitrary shell commands in a new window, call Bash directly with `run_in_background=true`. This skill is specifically for launching `claude` sessions.
- **Inside skills that forbid spawning** — `escalation-watcher-auto` explicitly prohibits any form of terminal spawning, including this skill. Respect the hosting skill's hard constraints.
