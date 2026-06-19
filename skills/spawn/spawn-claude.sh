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
#   2      — bad usage

set -u

if [ $# -ne 4 ]; then
  echo "usage: spawn-claude.sh <cwd> <skip_perms:true|false> <title> <prompt>" >&2
  exit 2
fi

cwd="$1"
skip_perms="$2"
title="$3"
prompt="$4"

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
  while [ ! -f "$sentinel" ]; do sleep 2; done
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
  exit "$rc"
}

# resolve_foreground: called after a foreground emulator returns.
# Sentinel present (or appears within a short grace) → session ran; its code
# wins.  Absent → launcher never started the payload → 127.
resolve_foreground() {
  if [ -f "$sentinel" ] || _wait_sentinel_grace; then
    finish
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
    finish
    ;;
  *)
    # User-supplied launcher via $CLAUDE_TERMINAL_CMD. Assume `<cmd> -- bash -c '<payload>'`
    # and detaching semantics — wait on sentinel.
    eval "$emulator -- bash -c \"\$inner\"" &
    wait $!
    resolve_detached $?
    ;;
esac
