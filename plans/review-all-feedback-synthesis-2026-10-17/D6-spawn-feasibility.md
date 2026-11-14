# D6 - Spawned-session result.md checkpoints and self-termination: feasibility and design

Read-only investigation. No session or terminal spawned, `spawn-claude.sh` not run, nothing changed in any repo.
Claude Code under test: `2.1.303` (`claude --version`). Date 2026-10-19.
Paths are relative to `/home/leo/src/dark-factory` unless absolute.

## Verdicts

| Question | Verdict |
|---|---|
| Q1 checkpoint vs final | FEASIBLE, small. Add one non-terminal outcome value (`in-progress`) and make "final" mean terminal outcome AND the session is gone or settled. Four files carry the vocabulary today. |
| Q2 self-termination | FEASIBLE and worth doing. Recommended: session requests exit by marker file, and the existing Stop hook sends `SIGINT` to its own owning `claude` pid. Static evidence only. It has not been exercised end to end, because the brief forbids spawning a session. |
| Wrong ideas to avoid | A hook JSON "exit decision" does not exist. `claude -p` loses attendance. `CLAUDE_CODE_EXIT_AFTER_STOP_DELAY` is idle-based and headless-only. Details in Q2. |

## Measured: the Bash tool's parent process

Run from this session's own Bash tool, which runs under the same harness as a spawned session:

```
shell $$=1827675                 comm=bash
  ppid=2406625  comm=claude   args="claude --allow-dangerously-skip-permissions --dangerously-skip-permissions --model fable"
  2406625 -> ppid=10858  comm=bash (the payload shell)
  10858   -> ppid=9786   comm=konsole
$PPID = 2406625
$CLAUDE_PID = 2406625   (exported into the Bash tool env)
$CLAUDE_CODE_SESSION_ATTENDED = 1
ps -o pid,ppid,stat,tty,args -p $CLAUDE_PID  ->  2406625 10858 Sl+ pts/7 claude ...
```

- The Bash tool's `$$` is a bash whose direct parent IS the `claude` process. `$PPID == $CLAUDE_PID` here.
- `$CLAUDE_PID` is the stable handle. `orchestrator/src/orchestrator/session_registry.py:3366` records the same measurement ("`$$`=2487079 (transient wrapper) vs `$PPID`==`$CLAUDE_PID`=2345789") and warns that `$PPID` "is not stable across tool calls either". Use `$CLAUDE_PID`, never `$PPID`.
- Hook subprocesses are a different shape. `session_hooks.py:881` (`_owning_claude_pid`) documents a measured tree of `claude -> sh -c -> <hook>.sh -> python3`. A hook finds claude by walking up to the first ancestor whose comm is `claude`. I did not measure whether hook env carries `CLAUDE_PID`, so the design does not rely on it.
- The owning pid is already persisted on the record as `claude_owner_pid` (`session_registry.py:280-290`). It is bound only by a proven-owner hook event (`session_hooks.py:684`, `_bind_claude_session_id`).

## Q1. Checkpoint vs final

### What the trailer asks today

`skills/spawn/spawn-claude.sh:327-344`:

```
Before you end this session (whether you finish, hand off, or get
blocked), write your outcome to: $CLAUDE_SPAWN_RESULT_FILE
...
outcome: done|blocked|abandoned|handed-off
changed: (commits/branches/task ids touched, or none)
action_needed: (what a human or parent should do next, or none)
---
A few sentences of prose context.

This is best-effort: writing this file must never block your normal exit.
```

- The trailer asks for exactly one write, "before you end this session". It has no checkpoint notion, so a session that writes `outcome: done` early and amends later looks final to any reader.
- The vocabulary lives in four places (SPOT risk, heuristic 11):
  - `skills/spawn/spawn-claude.sh:338`
  - `skills/spawn/SKILL.md:141`
  - `skills/escalation-watcher/SKILL.md:2268`
  - `scripts/sitting/ownership.py:45` (`RESULT_OUTCOMES`, with `OWNING_OUTCOMES = {'done','handed-off'}` at line 46)
- A consumer already exists: `scripts/sitting/ownership.py:330-345`. It treats a record as "in flight" whenever `record.status not in TERMINAL_STATUSES`, and only after that reads `outcome`. In other words it already keys on liveness AND outcome, which is the right shape.
- `ownership.py::parse_result_outcome` (line 139-150) returns None for any unknown outcome. A new `in-progress` value therefore degrades safely to "absent or unparsable" in that consumer even before it is updated.

### Minimal change

1. Add `in-progress` to the outcome vocabulary. Terminal outcomes stay `done|blocked|abandoned|handed-off`.
2. Trailer text, in one sentence: "If you write result.md before you are finished, write `outcome: in-progress`; overwrite it with a terminal outcome as your last act."
3. Reading rule, valid today with no hook change:
   - **Final** means: the outcome is terminal, AND the registry record's `status` is in `TERMINAL_STATUSES` (`exited` or `failed-to-start`; set by `finish()` in `spawn-claude.sh:588` via `session_registry.py exit`).
   - Anything else is a checkpoint, even if the outcome says `done`.
   - This is exactly the `ownership.py` rule, generalised: liveness decides, the outcome field only says what happened.
4. For a watcher that cannot wait for process exit (the pile-up case Leo describes, before self-termination exists), add a settled stamp in the Stop handler. Optional; it is a heuristic, so it is not the primary rule:
   - `run_stop` (`session_hooks.py:1754`, which flows through `_run_status_refresh_and_retitle`, the single write at about line 1700) reads `result.md` once per Stop.
   - It stamps an additive record field, for example `result_settled = {outcome, result_mtime_ns, at}`. This is a SCHEMA_MINOR bump (additive, per `session_registry.py` lines 62-74).
   - A Stop fires only at turn end, so the stamp means: "at the end of the most recent turn, result.md read as X and the session was idle".
   - Watcher rule for a still-live session: final iff outcome is terminal, `status == idle`, `result_settled.result_mtime_ns` equals the current mtime of result.md, and no newer write has appeared for a quiet period (suggest 10 minutes). Any later write plus Stop re-stamps and resets the clock.
5. Do not add a `final: true` field the model writes itself. It carries the same trust as `outcome` and adds a second thing to keep in sync. The Stop-hook stamp is derived from observed state instead.

## Q2. Self-termination: candidate mechanisms

Reference facts, quoted:

- Spawned sessions are interactive. `spawn-claude.sh:518`: `cd $q_cwd && claude $flags $q_prompt; ec=$?; exit $ec`. The flags are only `--dangerously-skip-permissions`, `--model`, and `CLAUDE_SPAWN_CLAUDE_ARGS`. An interactive `claude <prompt>` handles the prompt and then waits for input.
- `skills/spawn/SKILL.md:8,50` says the background task blocks "until the spawned session exits", so a session that never exits never raises the finished signal.
- Payload traps, `spawn-claude.sh:515-517`: EXIT trap writes `${ec:-$?}` to the sentinel, `trap 'exit 129' HUP`, `trap 'exit 143' TERM`. These traps run in the payload bash, not in `claude`.
- Only three hooks are installed in `~/.claude/settings.json`: SessionStart, Notification, Stop. There is no SessionEnd hook.
- The Stop hook fires at EVERY turn end ("Fires when Claude Code finishes a turn and goes idle", `skills/spawn/hooks/stop.sh:4`), not only at session end.

### (a) Stop hook sends a signal to its own `claude` process. FEASIBLE (recommended)

**Finding the pid.** No new discovery code is needed:
- `session_hooks.py::_owning_claude_pid` (line 881) walks `/proc` up from the hook to the first ancestor named `claude`, memoised per event as `probes.owning_claude_pid()`.
- `record.claude_owner_pid` is the bound owner. Compare the two before signalling. That guards against a nested `claude -p` that inherited `CLAUDE_SPAWN_SESSION_ID` (the exact failure `_env_slug_is_owned` exists to prevent).
- `CLAUDE_SPAWN_OWNER_PPID` is the payload shell, i.e. claude's PARENT. Do not signal that one (see Q4 on exit codes).

**Which signal.** Static evidence from the 2.1.303 binary (`strings`-level; unminified readings of the signal installer):
- `SIGINT` handler: `Gne(), this.shutdown(0)` (guarded by `printModeSignalHandlersRegistered`, false in the interactive path). Clean shutdown, exit code 0. `changelog.md` line 5687 (`~/.claude/cache/changelog.md`): "Fixed external SIGINT (e.g. IDE stop button, `kill -INT`) not running graceful shutdown - terminal modes are now restored and the `--resume` hint is printed instead of an abrupt exit".
- `SIGTERM` handler: `this.shutdown(143)`. Also graceful, but exits 143.
- `SIGHUP` handler: `this.shutdown(129)`.
- Prefer `SIGINT`: exit code 0 and terminal restored. The skill states exit codes are liveness-only (`SKILL.md:115`), so the code is cosmetic, but 0 avoids ever looking like the 129/143 failure paths.

**Mid-write risk.** Low in the Stop hook: it fires only when the turn has ended, so no tool call (and so no result.md write) is in flight. The only writes that can be lost are ones a background task is still making (Q4).

**Gating, required.** A Stop hook that kills on "terminal outcome in result.md" alone is WRONG for the review-all attended case: the session writes `outcome: done`, then runs `/reflect` over more turns, and Leo may still be talking to it. The hook must act only on an explicit request. Use a marker file next to result.md:
- Session's last act: `touch "$(dirname "$CLAUDE_SPAWN_RESULT_FILE")/exit-requested"` (a plain Bash command; `CLAUDE_SPAWN_RESULT_FILE` is one of the five vars a spawned session may read, `SKILL.md:91-99`).
- Hook preconditions, all required, all fail-soft (any doubt means do nothing): owner proven; marker exists; result.md outcome is terminal; `claude_owner_pid == owning pid`; Stop input `background_tasks` empty (field exists per changelog.md line 5335: "Stop and SubagentStop hook input now includes `background_tasks` and `session_crons` fields"; I did not verify its element shape, so treat a non-empty or unparsable value as "do not kill").
- Order inside the handler: the single `write_record` (status `idle`) first, then the signal. The registry write is therefore complete before claude begins shutting down.
- The marker file is deleted by the hook before signalling, so a `--resume` of the same record does not die immediately.

### (b) Session runs `kill -INT $CLAUDE_PID` from its own Bash tool. FEASIBLE but weaker

- Measured above: `$CLAUDE_PID` is exported and equals the Bash tool's parent. The signal reaches the right process.
- Weaknesses: the kill happens mid-turn, inside a tool call, so the final assistant message and the Stop hook (idle write, retitle) may never run; the transcript ends on a `tool_use` with no result; and nothing checks result.md first. It also trusts the model to get a shell one-liner right, which is the exact pattern `session_registry.py:3366` documents getting wrong for a year (`--pid $$`).
- Acceptable as a manual fallback only. If used, deferred form: `( sleep 3; kill -INT "$CLAUDE_PID" ) >/dev/null 2>&1 &` so the turn can end first. Still prefer (a).

### (c) Claude Code flag or hook response that ends the session. NOT AVAILABLE for interactive sessions

- `claude --help` (2.1.303): no flag exits an interactive session after its last turn. `--max-turns` is not in the help output at all. `--max-budget-usd` and `--no-session-persistence` are "only works with --print". `--bg` starts a background session and `claude stop <id>` stops one, which is a different lifecycle (daemon-hosted, no terminal for Leo).
- Hook output: the local hook docs (`~/.claude/plugins/marketplaces/claude-plugins-official/plugins/plugin-dev/skills/hook-development/SKILL.md`, "Standard Output") describe `"continue": false` as "halt processing". The binary shows what Stop does with it: `if(he.preventContinuation)Le=!0,At=he.stopReason||"Stop hook prevented continuation",yield en({type:"hook_stopped_continuation",...hookName:"Stop"...})`. It ends the turn and prints a message. It does not exit the process. `decision: block` does the opposite (keeps the turn going; capped at 8 consecutive blocks, changelog.md line 5413).
- `SessionEnd` has `reason` in `clear|resume|logout|prompt_input_exit|other` (binary schema). It is an observation of exit, not a trigger.
- `CLAUDE_CODE_EXIT_AFTER_STOP_DELAY`: exists in the binary (`AR(e,n)` logs "Exiting after ${r}ms of idle time"). Its call sites are the streaming/headless session loop (`inputClosed`, `sessionState`, the bridge child-env builder sets it from `idleExitAfterMs`). It is undocumented in `--help`, I found no evidence it applies to the TUI, and being idle-based it cannot tell "finished" from "waiting for Leo", which is wrong for an attended pass. Do not use.
- Typing `/exit` into the pane from outside works only on the tmux lane (`tmux send-keys -t <record.display.tmux_target> '/exit' Enter`; the target is stamped by `spawn-claude.sh:1170`) and is racy against a half-typed prompt. Not recommended.

### (d) `claude -p` for non-interactive streams. FEASIBLE, different product

- `-p/--print` exits after the turn by construction (SIGTERM in print mode also exits 143, changelog.md lines 3582, 4153).
- Loses: Leo cannot attend or steer, trust dialog and permission UX differ, the `/spawn` terminal-title and registry hooks (SessionStart/Notification/Stop) behave differently, and `--max-budget-usd` becomes the only brake. Suitable only for agent-shaped streams (the review-all `agent` streams, which already use an agent team), not for the attended pass 1.

### Recommendation for review-all (Leo attends pass 1, session should end after it)

Mechanism (a): the session writes final result.md, runs `/reflect`, then as its very last act touches `exit-requested`. The next Stop hook, which fires when that turn ends, verifies preconditions and sends `SIGINT` to the owning claude. Fallback if anything is off: the session simply stays open, which is today's behaviour, so nothing regresses.

## Q3. Reconciling "writing this file must never block your normal exit"

- The sentence (`spawn-claude.sh:344`) is about fault tolerance: a failed or skipped result.md write must never trap the session open. Self-termination does not contradict it as long as the new step is additive and fail-soft.
- Keep the sentence. Add a second one: "When everything you were asked to do is complete, including result.md and any /reflect, request the session to end: `touch "$(dirname "$CLAUDE_SPAWN_RESULT_FILE")/exit-requested"` (best-effort; if it fails, just stop)".
- Ordering, which the marker makes enforceable rather than hoped-for: (1) write final result.md, terminal outcome; (2) `/reflect` and any other close-out turns; (3) touch `exit-requested` as the last act.
- The trailer is appended only when `CLAUDE_SPAWN_RESULT_FILE` is set (`spawn-claude.sh:327`). On a registry fault there is no record dir, so no marker path and no exit request; the session falls back to waiting. That is the right fail-soft direction.
- Do not make the hook infer "done" from outcome alone (Q2a). The marker carries intent; the outcome check is a second guard.

## Q4. Risks

1. **Background Bash task of the session's own.** The signal triggers a graceful shutdown that kills the session's child shells (changelog.md line 1291: `/exit` "stops the servers and shells it started"; line 5079: SIGTERM to shell subprocesses before SIGKILL; line 139: SessionEnd hooks now run even with a background command still running). Anything still running is lost, including a background agent or workflow. Mitigation: the hook refuses when Stop input `background_tasks` is non-empty or unparsable (verify the element shape once; I did not). The trailer should also tell the session to request exit only after background work has completed. The hook then simply tries again at the next Stop, so a refusal is not a deadlock.
2. **The 129-on-clean-exit race** (`SKILL.md:128`). The race is a window teardown arriving while the payload unwinds. A `SIGINT` self-exit gives `claude` exit 0, the payload runs `exit $ec`, the EXIT trap (`spawn-claude.sh:515`) publishes the sentinel. If the emulator also closes the window at that instant, HUP can still win. Self-termination does not widen the race (no human is closing a window), and the skill's own conclusion stands: read result.md, never the code. The one new exposure is Leo closing the window during the last second; same as today. Note the signal goes to `claude`, not the payload shell, so the payload's `trap 'exit 143' TERM` never fires; do not signal `CLAUDE_SPAWN_OWNER_PPID`.
3. **Registry status after a self-kill.** Order is deterministic: the Stop handler's single `write_record` (status `idle`) lands first (`session_hooks.py::_run_status_refresh_and_retitle`, one write, then `_stamp_session_pointer`), then the signal, then the payload exits, the sentinel appears, and `finish()` (`spawn-claude.sh:588`) writes `exited` with the exit code. `exited` is the last writer. No SessionEnd hook is installed, so nothing else writes after it. `write_record` is last-writer-wins by documented design (the NON-GOAL paragraph in `_run_status_refresh_and_retitle`), which is harmless here because only one writer is alive at each step.
4. **Sibling mode has no `exited` writer.** `spawn-claude.sh:879-896` states that in sibling mode "no EXITED write ever lands on it" and the record is reclaimed by the stale-pid reaper. A self-terminated sibling-mode session therefore stays `idle` until the reaper takes it (`NON_TERMINAL_HEARTBEAT_TTL = 1h` with a dead launcher_pid, `session_registry.py:2546`). For such a session the watcher must use the Q1 settled stamp or the result.md outcome, not `status == exited`. A clean fix is for the Stop handler to stamp `exit_requested_at` on the record when it signals; keep that inside the same single write.
5. **Wrong-process kill.** Signalling anything other than the owner is the dangerous failure. Guard: signal only `probes.owning_claude_pid()` and only when it equals `record.claude_owner_pid`, so an unbound or unproven record (the launch window, `_in_unbound_launch_window`) never self-kills. A stale pid cannot be reused within the milliseconds between the walk and the signal in a meaningful way, but the equality check and doing it in the same handler keep that window minimal.
6. **Unmeasured behaviour.** The SIGINT-exits-0 and graceful-shutdown claims rest on binary strings plus the changelog, not on a live run. Before the PRD leaf ships, run one probe in a scratch tmux-lane session (hook logs the pid, sends SIGINT, check the sentinel and `record.exit_code` and that the resume hint prints). Treat that probe as the leaf's own acceptance test, not as a precondition for filing it.
7. **Attended pass.** If Leo wants to keep talking after the session announces completion, the marker has not been touched yet, so nothing happens. Leo's interaction is the guard; no timer is involved.

## Recommended design (14 lines)

1. Vocabulary: add `in-progress` to the outcome set. Edit the four sites together: `spawn-claude.sh:338`, `skills/spawn/SKILL.md:141`, `skills/escalation-watcher/SKILL.md:2268`, `scripts/sitting/ownership.py:45` (non-owning).
2. Trailer text, after "must never block your normal exit": "Write `outcome: in-progress` if you write result.md before you are done. When all work, result.md and /reflect are complete, run: `touch \"$(dirname \"$CLAUDE_SPAWN_RESULT_FILE\")/exit-requested\"` as your final action (best-effort; skip it if it fails)."
3. Watcher rule: result.md is FINAL iff outcome is one of `done|blocked|abandoned|handed-off` AND record.status is `exited`/`failed-to-start`. Otherwise it is a checkpoint and must not be archived.
4. For live-but-finished sessions (and sibling mode), `run_stop` stamps `result_settled = {outcome, result_mtime_ns, at}` on the record. Final-while-live iff terminal outcome, status idle, mtime unchanged since the stamp, and 10 quiet minutes.
5. `run_stop` self-termination block, placed after the single `write_record` and `_stamp_session_pointer`, behind a fail-soft `try`:
6. Preconditions: ownership proven (not withheld, may_bind true); `<record_dir>/exit-requested` exists; result.md outcome terminal; `probes.owning_claude_pid() == record.claude_owner_pid`; Stop input `background_tasks` empty.
7. Action: delete the marker, then `os.kill(owner_pid, signal.SIGINT)`. Never signal the payload shell or `launcher_pid`.
8. `stop.sh` and `main()` keep their contract: always exit 0, log loudly on fault. No change to `stop.sh` itself; the handler already has the hook stdin.
9. `run_stop` return value stays the OSC retitle. Optionally add an exit glyph for the final retitle (cosmetic).
10. `finish()` is unchanged: it writes `exited` after the sentinel appears. In sibling mode nothing does; rely on rule 4.
11. Tests: `test_session_hooks.py` gets cases using the injectable `ProcessTree` and `_process_tree_fakes` (no real signal; inject the kill callable): kill only with marker plus terminal outcome plus pid equality; never with a non-empty `background_tasks`; never on a nested inheritor; marker deleted on act.
12. Acceptance probe (one-off, human run): scratch tmux-lane spawn of a trivial prompt that touches the marker; confirm exit 0, `exited` last, resume hint printed.
13. review-all: pass 1 trailer variant ends with the marker touch after the program doc and `/reflect`; `agent` streams keep using the agent team and need none of this.
14. Out of scope: `claude -p` conversion, `CLAUDE_CODE_EXIT_AFTER_STOP_DELAY`, tmux `send-keys /exit`, SessionEnd hook.

## Evidence index

- `skills/spawn/SKILL.md:8,50,91-99,115-128,136-144`
- `skills/spawn/spawn-claude.sh:327-344` (trailer), `:515-518` (payload and traps), `:588` (`finish`), `:879-896` (sibling mode)
- `skills/spawn/hooks/README.md` (fail-soft, always exit 0), `stop.sh:4`, `session-start.sh`
- `orchestrator/src/orchestrator/session_hooks.py:684` (bind), `:881` (`_owning_claude_pid`), `:996` (`_owner_ppid_verdict`), `:1590` (`_run_status_refresh_and_retitle`), `:1754` (`run_stop`), `:1921` (`main`)
- `orchestrator/src/orchestrator/session_registry.py:76` (`RESULT_FILENAME`), `:280-290` (`claude_owner_pid`), `:2546` (reaper TTL), `:3237,3366` (`CLAUDE_PID`)
- `scripts/sitting/ownership.py:45-46,139-150,330-345`
- `~/.claude/cache/changelog.md` lines 139, 1291, 3582, 4153, 5079, 5335, 5413, 5687
- Binary: `/home/leo/.local/share/claude/versions/2.1.303` (signal installer, Stop `preventContinuation`, `EXIT_AFTER_STOP_DELAY`)
- Hook docs: `~/.claude/plugins/marketplaces/claude-plugins-official/plugins/plugin-dev/skills/hook-development/SKILL.md`
