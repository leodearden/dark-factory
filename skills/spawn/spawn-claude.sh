#!/usr/bin/env bash
# Launch `claude` in a new terminal window and block until that session exits.
#
# Designed to be invoked as a background task from the /spawn skill so the
# background task's completion is a reliable signal that the spawned session
# has finished. Per-emulator wait quirks (gnome-terminal --wait, konsole and
# macOS Terminal daemonizing, etc.) are handled internally — callers just pass
# cwd + prompt and rely on the exit-code contract below.
#
# Usage:
#   spawn-claude.sh <cwd> <skip_permissions:true|false> <title|""> <prompt>
#
# Emulator discovery order:
#   1. $CLAUDE_TERMINAL_CMD             (preferred env override)
#   2. $ESCALATION_TERMINAL_CMD         (legacy fallback)
#   3. gnome-terminal / kitty / konsole / xterm in $PATH
#   4. `open -a Terminal` on macOS
#   5. exit 126 — caller must prompt user and set $CLAUDE_TERMINAL_CMD
#
# Exit codes:
#   0..125 — claude's own exit code (recovered from sentinel)
#   126    — no terminal emulator found
#   127    — launcher itself failed (emulator exited before writing the sentinel)
#   129    — terminal window closed while the session was alive (SIGHUP)
#   144    — Claude never started within the started-grace window (registry marked failed-to-start)
#   2      — bad usage

set -u

# Distinct exit code for the started-verification watchdog's failed-to-start
# signal (Attention Rail T4) -- see the exit-codes comment above. 144-199 is
# free (0-125 claude, 126 no-emulator, 127 launcher-fail, 129 window-closed,
# 143 SIGTERM, 2 usage), so this is additive and backward-compatible.
EXIT_FAILED_TO_START=144

# Absolute path to the repo root, computed from this script's own location
# (skills/spawn/spawn-claude.sh -> skills/spawn -> skills -> repo root) so the
# session-registry helper (task 2285) can be invoked by absolute path with no
# venv, PYTHONPATH, or install required.
REPO_ROOT="$(cd "$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")" && pwd)"
SESSION_REGISTRY_PY="$REPO_ROOT/orchestrator/src/orchestrator/session_registry.py"

if [ $# -ne 4 ]; then
  echo "usage: spawn-claude.sh <cwd> <skip_perms:true|false> <title> <prompt>" >&2
  exit 2
fi

cwd="$1"
skip_perms="$2"
title="$3"
prompt="$4"

# Started-verification watchdog (Attention Rail T4) state: a spawn-time
# reference marker (mtime = spawn start, used by the step-4 transcript-
# appearance probe to distinguish a fresh transcript from a stale one), a
# failed-to-start marker path (NOT created here -- mktemp -u only reserves a
# unique name; the watchdog creates the file itself when it flags), the
# started-grace window, and the transcript-scan root. All independent of
# SPAWN_LAUNCH_GRACE_SECS so genuine-launcher-failure (127) paths are
# unaffected and existing tests (which only shrink the launch grace) never
# trip the new code.
spawn_ref="$(mktemp -t spawn-claude-ref-XXXXXX)"
fts_marker="$(mktemp -u -t spawn-claude-XXXXXX.fts)"
SPAWN_STARTED_GRACE_SECS="${SPAWN_STARTED_GRACE_SECS:-90}"
CLAUDE_PROJECTS_DIR="${CLAUDE_PROJECTS_DIR:-$HOME/.claude/projects}"

# Session-registry: record this spawn as LAUNCHING (task 2285, Attention Rail
# T3). Best-effort and fully additive -- a missing python3 or any registry
# fault must never change this script's exit-code contract. Only stdout (the
# record dir, printed on success) is captured; stderr flows through to this
# script's own stderr for loud, caller-visible diagnostics. CLAUDE_SPAWN_ROLE/
# PROJECT/TASK_ID/ESCALATION_ID pass through from this process's own
# environment when a caller already set them.
SESSION_RECORD_DIR=""
if command -v python3 >/dev/null 2>&1; then
  SESSION_RECORD_DIR=$(
    CLAUDE_SPAWN_TITLE="$title" \
    CLAUDE_SPAWN_PROMPT="$prompt" \
    CLAUDE_SPAWN_CWD="$cwd" \
    CLAUDE_SPAWN_LAUNCHER_PID="$$" \
    python3 "$SESSION_REGISTRY_PY" launching
  ) || true
fi

flags=""
[ "$skip_perms" = "true" ] && flags="--dangerously-skip-permissions"

sentinel="$(mktemp -u -t spawn-claude-XXXXXX.done)"

q_cwd=$(printf %q "$cwd")
q_prompt=$(printf %q "$prompt")
q_sentinel=$(printf %q "$sentinel")

# Payload that runs inside the new terminal.  Traps ensure the sentinel is
# written even when the terminal window is closed (SIGHUP/TERM) while the
# session is alive:
#   - EXIT trap: always writes ${ec:-$?} — claude's real code on normal exit,
#     or a 128+signo default when pre-empted.
#   - HUP trap: converts SIGHUP into exit 129 so the EXIT trap records 129
#     (distinguishable "window closed while alive" code).
#   - TERM trap: converts SIGTERM into exit 143 (128+15).
# ec is set only after claude returns so ${ec:-$?} captures the signal-path
# default if claude is pre-empted before ec is assigned.
#
# CONTRACT: These HUP/TERM traps require the launching terminal to deliver the
# payload shell with DEFAULT (non-ignored) signal dispositions for HUP and TERM.
# Real terminal emulators (konsole, gnome-terminal, xterm, kitty, macOS Terminal)
# guarantee this: they reset child signal dispositions to SIG_DFL before exec'ing
# the child shell.  A launch context that inherits SIGHUP=SIG_IGN (e.g. a
# nohup'd or fully-detached background process) would, absent the terminal's
# reset, make `trap 'exit 129' HUP` a silent POSIX no-op: a non-interactive bash
# cannot trap a signal that was SIG_IGN on entry.  If the window-close path ever
# appears broken, verify the terminal's disposition reset — do NOT simply re-bump
# the await_sentinel timeout (that symptom is a hang, not a latency issue).
inner="trap 'echo \"\${ec:-\$?}\" > $q_sentinel' EXIT; \
trap 'exit 129' HUP; \
trap 'exit 143' TERM; \
cd $q_cwd && claude $flags $q_prompt; ec=\$?; exit \$ec"

# How long to wait for the sentinel to appear after the launcher returns
# (covers a hair-late write or a very fast emulator).  Tests can shrink this.
SPAWN_LAUNCH_GRACE_SECS="${SPAWN_LAUNCH_GRACE_SECS:-5}"

await_sentinel() {
  while [ ! -f "$sentinel" ] && [ ! -f "$fts_marker" ]; do sleep 1; done
}

# Wait up to SPAWN_LAUNCH_GRACE_SECS for the sentinel to appear.
# Returns 0 if the sentinel appeared in time, 1 if it did not.
_wait_sentinel_grace() {
  local end=$(( SECONDS + SPAWN_LAUNCH_GRACE_SECS ))
  while [ ! -f "$sentinel" ] && [ "$SECONDS" -lt "$end" ]; do
    sleep 0.1
  done
  [ -f "$sentinel" ]
}

finish() {
  local rc=127
  if [ -f "$sentinel" ]; then
    rc=$(cat "$sentinel" 2>/dev/null || echo 127)
  fi
  rm -f "$sentinel"
  # Session-registry: record the final exit code (task 2285). rc is already
  # fully determined from the sentinel above, independent of this call, so a
  # registry fault here can never change the spawn's exit-code contract.
  if [ -n "$SESSION_RECORD_DIR" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$SESSION_REGISTRY_PY" exit --record "$SESSION_RECORD_DIR" --code "$rc" || true
  fi
  exit "$rc"
}

# True when the started-watchdog has flagged failed-to-start AND no exit
# sentinel has appeared since. A real exit sentinel (the session actually
# ran) always wins over a late marker -- this is what lets await_sentinel/
# resolve_* unblock on the marker while never overriding a genuine exit.
_failed_to_start_pending() {
  [ -f "$fts_marker" ] && [ ! -f "$sentinel" ]
}

# Exit via the distinct failed-to-start code. Deliberately does NOT call the
# session-registry `exit` subcommand (unlike finish()) -- that would
# overwrite the failed-to-start status the watchdog already wrote with
# EXITED, destroying the signal this whole mechanism exists to produce.
finish_failed_to_start() {
  local code
  code=$(cat "$fts_marker" 2>/dev/null || echo "$EXIT_FAILED_TO_START")
  rm -f "$fts_marker" "$sentinel"
  exit "$code"
}

# Mirror session_registry.transcript_path_for_cwd's encoding byte-for-byte:
# every '/' then every '.' maps to '-' (empirically re-verified 2026-07-07,
# e.g. /home/leo/src/dark-factory -> -home-leo-src-dark-factory). Used to
# locate this spawn's transcript directory under $CLAUDE_PROJECTS_DIR.
_encode_cwd() {
  local e="${1//\//-}"
  printf '%s' "${e//./-}"
}

# Best-effort: is any process named `claude` a descendant of this script's
# own PID? Guarded end-to-end -- ps unavailable, or any parsing hiccup,
# yields "no descendant found" (fail-soft) rather than a fault. Scoped to
# $$'s ancestor chain so it only matches a foreground-launched claude
# (detached emulators reparent claude away from this script, so it is
# correctly empty there) and never a coincidental, unrelated claude process
# elsewhere on a busy host.
_claude_descendant_alive() {
  command -v ps >/dev/null 2>&1 || return 1

  local -A ppid_of=()
  local -a claude_pids=()
  local pid ppid comm

  while read -r pid ppid comm; do
    [ -n "$pid" ] || continue
    ppid_of["$pid"]="$ppid"
    [ "$comm" = "claude" ] && claude_pids+=("$pid")
  done < <(ps -eo pid=,ppid=,comm= 2>/dev/null)

  local cand anc hops
  for cand in "${claude_pids[@]}"; do
    anc="${ppid_of[$cand]:-}"
    hops=0
    while [ -n "$anc" ] && [ "$hops" -lt 100 ]; do
      [ "$anc" = "$$" ] && return 0
      anc="${ppid_of[$anc]:-}"
      hops=$(( hops + 1 ))
    done
  done
  return 1
}

# Positive start-evidence: either a transcript file appeared under this
# spawn's $CLAUDE_PROJECTS_DIR/<encoded-cwd>/ newer than the spawn-time
# reference marker, or a claude process is running as this script's
# descendant. The transcript probe is the primary, deterministically-tested
# signal (the motivating incident is "no transcript ever appeared"); the
# process probe is a guarded best-effort secondary.
_started_evidence() {
  local d="$CLAUDE_PROJECTS_DIR/$(_encode_cwd "$cwd")"
  if [ -d "$d" ] && [ -n "$(find "$d" -type f -newer "$spawn_ref" -print -quit 2>/dev/null)" ]; then
    return 0
  fi
  _claude_descendant_alive && return 0
  return 1
}

# Backgrounded started-verification watchdog (Attention Rail T4). Forked once
# right before the emulator case dispatch so it is already running whether
# the launcher turns out to be foreground (blocks until window-close) or
# detached (returns immediately). Polls up to SPAWN_STARTED_GRACE_SECS for
# start-evidence: the exit sentinel, a fresh transcript, or a live claude
# descendant.
#
# On grace-expiry with no evidence, flags failed-to-start: best-effort marks
# the session-registry record, emits a loud caller-visible stderr line, and
# writes the fts_marker so await_sentinel/resolve_* can break out of what
# would otherwise be an unbounded wait.
_started_watchdog() {
  # Clear inherited traps in this backgrounded copy -- without this, the
  # child would run the parent's _cleanup on its own return, which would be
  # wrong (double cleanup / killing its own pid) and is never what we want
  # for a background job.
  trap - EXIT HUP TERM

  local end
  end=$(( $(date +%s) + SPAWN_STARTED_GRACE_SECS ))
  while [ "$(date +%s)" -lt "$end" ]; do
    [ -f "$sentinel" ] && return 0
    _started_evidence && return 0
    sleep 0.25
  done
  # Final check -- avoid flagging on a sentinel that landed right at the
  # deadline boundary.
  [ -f "$sentinel" ] && return 0

  if [ -n "$SESSION_RECORD_DIR" ] && command -v python3 >/dev/null 2>&1; then
    python3 "$SESSION_REGISTRY_PY" refresh --record "$SESSION_RECORD_DIR" --status failed-to-start || true
  fi
  echo "spawn-claude.sh: FAILED-TO-START — claude never started within ${SPAWN_STARTED_GRACE_SECS}s (no transcript, no claude process); registry marked failed-to-start; exiting ${EXIT_FAILED_TO_START}." >&2
  echo "$EXIT_FAILED_TO_START" > "$fts_marker"
  return 0
}

# resolve_foreground: called after a foreground emulator returns.
# Sentinel present (or appears within a short grace) → session ran; its code
# wins.  Absent → launcher never started the payload → 127.
resolve_foreground() {
  if [ -f "$sentinel" ] || _wait_sentinel_grace; then
    finish
  fi
  if _failed_to_start_pending; then
    finish_failed_to_start
  fi
  exit 127
}

# resolve_detached: called after a detaching emulator's launcher process exits.
# $1 = launcher exit code.
#   Sentinel present                      → session ran; its code wins.
#   launch_rc != 0, no sentinel in grace  → genuine launcher failure → 127.
#   launch_rc == 0, no sentinel           → session still running → await unbounded.
resolve_detached() {
  local launch_rc="$1"
  if [ -f "$sentinel" ]; then
    finish
  elif [ "$launch_rc" -ne 0 ]; then
    if _wait_sentinel_grace; then
      finish
    fi
    exit 127
  else
    await_sentinel
    if _failed_to_start_pending; then
      finish_failed_to_start
    fi
    finish
  fi
}

# --- emulator selection ----------------------------------------------------

emulator=""
if [ -n "${CLAUDE_TERMINAL_CMD:-}" ]; then
  emulator="$CLAUDE_TERMINAL_CMD"
elif [ -n "${ESCALATION_TERMINAL_CMD:-}" ]; then
  emulator="$ESCALATION_TERMINAL_CMD"
  echo "spawn-claude.sh: \$ESCALATION_TERMINAL_CMD is a legacy fallback — please migrate to \$CLAUDE_TERMINAL_CMD" >&2
elif command -v gnome-terminal >/dev/null 2>&1; then
  emulator="gnome-terminal"
elif command -v kitty >/dev/null 2>&1; then
  emulator="kitty"
elif command -v konsole >/dev/null 2>&1; then
  emulator="konsole"
elif command -v xterm >/dev/null 2>&1; then
  emulator="xterm"
elif [ "$(uname)" = "Darwin" ]; then
  emulator="mac-terminal"
else
  echo "spawn-claude.sh: no terminal emulator found — set \$CLAUDE_TERMINAL_CMD" >&2
  exit 126
fi

# Dispatch by the first word of $emulator so $CLAUDE_TERMINAL_CMD="gnome-terminal --foo"
# still hits the gnome-terminal branch.
first_word="${emulator%% *}"

# Kill the started-watchdog on EVERY exit path -- MANDATORY. A backgrounded
# child left running past this script's exit would hold the caller-captured
# stdout/stderr pipe open (blocking /spawn's background task, and any test
# harness's subprocess.wait()) until the full started-grace elapsed. Set
# before the fork so no exit between here and the fork can leak the child.
_cleanup() {
  [ -n "${watchdog_pid:-}" ] && kill "$watchdog_pid" 2>/dev/null
  rm -f "$spawn_ref" "$fts_marker"
}
trap _cleanup EXIT

watchdog_pid=""
_started_watchdog &
watchdog_pid=$!

case "$first_word" in
  gnome-terminal)
    # --wait keeps the launcher attached until the window closes.
    args=(--wait)
    [ -n "$title" ] && args+=(--title="$title")
    args+=(-- bash -c "$inner")
    gnome-terminal "${args[@]}"
    resolve_foreground
    ;;
  xterm)
    # xterm is naturally foreground.
    args=()
    [ -n "$title" ] && args+=(-T "$title")
    args+=(-e bash -c "$inner")
    xterm "${args[@]}"
    resolve_foreground
    ;;
  kitty)
    # Default kitty (no --single-instance) is naturally foreground.
    args=()
    [ -n "$title" ] && args+=(--title "$title")
    args+=(bash -c "$inner")
    kitty "${args[@]}"
    resolve_foreground
    ;;
  konsole)
    # konsole daemonizes — launch, then wait on the sentinel.
    args=()
    [ -n "$title" ] && args+=(-p "tabtitle=$title")
    args+=(-e bash -c "$inner")
    konsole "${args[@]}" &
    wait $!
    resolve_detached $?
    ;;
  mac-terminal)
    # `open -a Terminal` needs a script file — it doesn't forward command args.
    tmpscript=$(mktemp -t spawn-claude-XXXXXX)
    printf '#!/usr/bin/env bash\n%s\n' "$inner" > "$tmpscript"
    chmod +x "$tmpscript"
    open -a Terminal "$tmpscript" || { rm -f "$tmpscript" "$sentinel"; exit 127; }
    await_sentinel
    rm -f "$tmpscript"
    if _failed_to_start_pending; then
      finish_failed_to_start
    fi
    finish
    ;;
  *)
    # User-supplied launcher via $CLAUDE_TERMINAL_CMD. Assume `<cmd> -- bash -c '<payload>'`
    # and detaching semantics — wait on sentinel.
    # Word-split $emulator into an array so multi-word commands like
    # "some-term --opt" work.  Do NOT use eval: $inner contains literal double
    # quotes which break the quoting when eval re-parses "bash -c \"$inner\"".
    # shellcheck disable=SC2206
    _emcmd=($emulator)
    "${_emcmd[@]}" -- bash -c "$inner" &
    wait $!
    resolve_detached $?
    ;;
esac
