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
#   127    — launcher itself failed (couldn't open a window)
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

# Payload that runs inside the new terminal. We capture claude's exit code in
# the sentinel so it can be propagated back through this script.
inner="cd $q_cwd && claude $flags $q_prompt; ec=\$?; echo \$ec > $q_sentinel; exit \$ec"

await_sentinel() {
  while [ ! -f "$sentinel" ]; do sleep 2; done
}

finish() {
  local rc=127
  if [ -f "$sentinel" ]; then
    rc=$(cat "$sentinel" 2>/dev/null || echo 127)
  fi
  rm -f "$sentinel"
  exit "$rc"
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
    gnome-terminal "${args[@]}" || exit 127
    finish
    ;;
  xterm)
    # xterm is naturally foreground.
    args=()
    [ -n "$title" ] && args+=(-T "$title")
    args+=(-e bash -c "$inner")
    xterm "${args[@]}" || exit 127
    finish
    ;;
  kitty)
    # Default kitty (no --single-instance) is naturally foreground.
    args=()
    [ -n "$title" ] && args+=(--title "$title")
    args+=(bash -c "$inner")
    kitty "${args[@]}" || exit 127
    finish
    ;;
  konsole)
    # konsole daemonizes — launch, check that the launcher itself didn't fail,
    # then wait on the sentinel.
    args=()
    [ -n "$title" ] && args+=(-p "tabtitle=$title")
    args+=(-e bash -c "$inner")
    konsole "${args[@]}" &
    wait $! || { rm -f "$sentinel"; exit 127; }
    await_sentinel
    finish
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
    wait $! || { rm -f "$sentinel"; exit 127; }
    await_sentinel
    finish
    ;;
esac
