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
# Backend selection:
#   $CLAUDE_SPAWN_BACKEND=tmux — bypass terminal-emulator discovery entirely
#     and launch in a crash-survivable, reattachable tmux window instead
#     (Fleet Cockpit C6). Per-project session `fleet-<project>` (project from
#     $CLAUDE_SPAWN_PROJECT, else basename(cwd), else "fleet"), one window
#     per runner; override the session name via $CLAUDE_SPAWN_TMUX_SESSION.
#     Reattach with `tmux attach -t <session>`. Requires `tmux` on $PATH —
#     see exit code 126 below.
#
# Emulator discovery order (when $CLAUDE_SPAWN_BACKEND != tmux):
#   1. $CLAUDE_TERMINAL_CMD             (preferred env override)
#   2. $ESCALATION_TERMINAL_CMD         (legacy fallback)
#   3. gnome-terminal / kitty / konsole / xterm in $PATH
#   4. `open -a Terminal` on macOS
#   5. exit 126 — caller must prompt user and set $CLAUDE_TERMINAL_CMD
#
# Exit codes:
#   0..125 — claude's own exit code (recovered from sentinel)
#   126    — no usable launcher (no terminal emulator found / tmux missing in tmux mode)
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

# Fleet Cockpit C7 (plans/fleet-cockpit-prd.md §6.1): spawn_mode selects
# which identity the child inherits as its own CLAUDE_SPAWN_PARENT_ID (see
# the identity-export block below) and, later, whether this launch is
# fire-and-forget (see resolve_sibling). Read once, up front, alongside the
# positional-arg parse -- 'child' (the default) preserves every existing
# caller's behavior byte-for-byte.
spawn_mode="${CLAUDE_SPAWN_MODE:-child}"

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

# Kill the started-watchdog on EVERY exit path -- MANDATORY. A backgrounded
# child left running past this script's exit would hold the caller-captured
# stdout/stderr pipe open (blocking /spawn's background task, and any test
# harness's subprocess.wait()) until the full started-grace elapsed.
# Installed here, immediately after the mktemp calls above and before ANY
# possible exit path (including `exit 126` below if no terminal emulator is
# found) -- otherwise spawn_ref would leak on that path. watchdog_pid stays
# unset until right before the fork (further below); the `${watchdog_pid:-}`
# guard makes an early exit's kill attempt a safe no-op. Empirically verified
# that neither command substitutions (the registry-write subshell just below)
# nor backgrounded function calls (the watchdog fork below) independently
# re-fire an inherited EXIT trap in bash -- it fires exactly once, when this
# top-level script itself exits -- so installing it this early cannot cause
# spawn_ref to be removed prematurely.
_cleanup() {
  [ -n "${watchdog_pid:-}" ] && kill "$watchdog_pid" 2>/dev/null
  rm -f "$spawn_ref" "$fts_marker"
}
trap _cleanup EXIT

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

# Result-handback (Attention Rail T5): the record dir just captured above
# deterministically owns a result.md alongside record.json -- this literal
# 'result.md' MUST stay in sync with session_registry.RESULT_FILENAME (the
# two are never cross-checked at runtime, only by convention; see that
# module's docstring). Empty when SESSION_RECORD_DIR is empty (a registry
# fault already handled above via fail-soft), so both the env-export below
# and the prompt trailer (further down) cleanly no-op instead of pointing at
# a bogus path.
CLAUDE_SPAWN_RESULT_FILE=""
[ -n "$SESSION_RECORD_DIR" ] && CLAUDE_SPAWN_RESULT_FILE="$SESSION_RECORD_DIR/result.md"

# Exported into $inner (below) so the spawned session can write its outcome
# there. Built unconditionally (empty when CLAUDE_SPAWN_RESULT_FILE is
# empty) so `${result_export}` is always safe to splice into `inner` as a
# no-op empty-string prefix.
result_export=""
if [ -n "$CLAUDE_SPAWN_RESULT_FILE" ]; then
  q_result_file=$(printf %q "$CLAUDE_SPAWN_RESULT_FILE")
  result_export="export CLAUDE_SPAWN_RESULT_FILE=$q_result_file; "
fi

# Fleet Cockpit C1 (plans/fleet-cockpit-prd.md §6.1) identity exports: give
# the spawned child its OWN new session-registry slug, plus its parent
# linkage. Default 'child' spawn_mode -- the child's parent-of-record is the
# DIRECT spawner, i.e. THIS script's own inherited CLAUDE_SPAWN_SESSION_ID
# (empty when this is a human-launched root with nothing to inherit).
# Fleet Cockpit C7: spawn_mode=sibling instead parents the child at THIS
# script's own inherited CLAUDE_SPAWN_PARENT_ID -- the shared ancestor, not
# the direct spawner -- so a sibling handoff (e.g. /prd author->decompose)
# renders under the same ancestor as the spawner itself, rather than nesting
# one level deeper. Both branches are nested under the same SESSION_RECORD_DIR
# check as CLAUDE_SPAWN_RESULT_FILE above, so a registry fault (empty
# SESSION_RECORD_DIR) cleanly no-ops BOTH exports together -- it would be
# incoherent to hand the child a parent link with no session identity of its
# own. child_session_id is derived from SESSION_RECORD_DIR's basename,
# byte-identical to record.session_slug (_run_launching prints
# <root>/sessions/<slug>), exactly how CLAUDE_SPAWN_RESULT_FILE is derived
# above.
spawn_id_export=""
parent_id_export=""
if [ -n "$SESSION_RECORD_DIR" ]; then
  child_session_id="${SESSION_RECORD_DIR##*/}"
  q_child=$(printf %q "$child_session_id")
  spawn_id_export="export CLAUDE_SPAWN_SESSION_ID=$q_child; "

  if [ "$spawn_mode" = "sibling" ]; then
    parent_of_child="${CLAUDE_SPAWN_PARENT_ID:-}"
  else
    parent_of_child="${CLAUDE_SPAWN_SESSION_ID:-}"
  fi
  if [ -n "$parent_of_child" ]; then
    q_parent=$(printf %q "$parent_of_child")
    parent_id_export="export CLAUDE_SPAWN_PARENT_ID=$q_parent; "
  fi
fi

# Result-handback trailer (Attention Rail T5): appended to the prompt itself
# so the spawned session is told, in-band, to write its outcome before
# ending. Gated on the same non-empty check as result_export above -- a
# fail-soft empty record dir means no trailer is appended either, so a
# session is never pointed at a bogus path. This is purely additive text;
# `q_prompt=$(printf %q "$prompt")` below safely quotes the now-multi-line
# prompt through `bash -c "$inner"` unchanged. Best-effort by design: the
# trailer only asks the session to write the file, nothing here blocks exit
# on it, and a parent that finds no result.md simply falls back to its own
# exploration.
#
# DELIBERATE ORDERING: the session-registry `launching` write further above
# (CLAUDE_SPAWN_PROMPT="$prompt") already ran and persisted `prompt` BEFORE
# this trailer is appended, so record.prompt intentionally holds the
# caller's original prompt, not this machine-appended boilerplate --
# record.prompt is "what the caller asked for", not "the literal argv
# claude received". Do NOT reorder the registry call to run after this
# block to make them match; a reader who needs the exact text claude saw
# should reconstruct it from record.prompt plus this trailer template.
if [ -n "$CLAUDE_SPAWN_RESULT_FILE" ]; then
  prompt="${prompt}

---
Before you end this session (whether you finish, hand off, or get
blocked), write your outcome to: $CLAUDE_SPAWN_RESULT_FILE

Format (markdown with a small structured header -- parent readers are
LLMs, so keep it concise and scannable):

---
outcome: done|blocked|abandoned|handed-off
changed: (commits/branches/task ids touched, or none)
action_needed: (what a human or parent should do next, or none)
---
A few sentences of prose context.

This is best-effort: writing this file must never block your normal exit."
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
${spawn_id_export}${parent_id_export}${result_export}cd $q_cwd && claude $flags $q_prompt; ec=\$?; exit \$ec"

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
  # Associative arrays (bash 4+) are required below. Stock macOS ships
  # /bin/bash 3.2 (this script explicitly supports macOS via the
  # mac-terminal branch), where `local -A` is a hard parse/option error, not
  # a soft failure -- and this function is polled every ~0.25s for up to
  # SPAWN_STARTED_GRACE_SECS (default 90s) whenever no transcript is yet
  # present, so an unguarded error here would spam the caller-captured
  # stderr with hundreds of lines. Fail-soft: skip this best-effort probe on
  # bash <4 and rely on the transcript probe (_started_evidence's other
  # branch) instead.
  [ "${BASH_VERSINFO[0]:-0}" -ge 4 ] || return 1

  local -A ppid_of=()
  local -a claude_pids=()
  local pid ppid rest tok1 tok2 discard

  while read -r pid ppid rest; do
    [ -n "$pid" ] || continue
    ppid_of["$pid"]="$ppid"
    # A real `claude` CLI is commonly a shebang wrapper (e.g. node/bash), in
    # which case `comm` reports the INTERPRETER, not "claude" -- the kernel
    # rewrites the exec to run the interpreter with the script's own path as
    # an argument (empirically verified: a `#!/usr/bin/env bash` script named
    # `claude` reports comm=bash, args="bash /path/to/claude"). Match on the
    # first two whitespace-separated tokens of the full command line instead,
    # covering both a raw `claude` binary (tok1) and an interpreter+script
    # pair (tok2), by comparing each token's basename.
    read -r tok1 tok2 discard <<<"$rest"
    if [ "${tok1##*/}" = "claude" ] || [ "${tok2##*/}" = "claude" ]; then
      claude_pids+=("$pid")
    fi
  done < <(ps -eo pid=,ppid=,args= 2>/dev/null)

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
#
# KNOWN LIMITATION (concurrent same-cwd spawns): the transcript directory is
# keyed on cwd, not on this session -- Claude Code writes every session for a
# given cwd into the same $CLAUDE_PROJECTS_DIR/<encoded-cwd>/ directory. If a
# second, genuinely healthy spawn for the SAME cwd writes its own new
# transcript file while THIS spawn is failing to start, that sibling's file
# is indistinguishable here from this spawn's own evidence -- _started_evidence
# returns true and this spawn is never flagged (a false negative). This
# detector is therefore best-effort under a concurrent same-cwd fleet; it
# remains exact for the common single-spawn-per-cwd case, including the
# motivating 2026-07-06 incident. See SKILL.md's Verification section for the
# caller-facing note.
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
#
# KNOWN LIMITATION (detached slow-start false-failure): for a DETACHED
# launcher (konsole, or a custom $CLAUDE_TERMINAL_CMD), _claude_descendant_alive
# is always empty by construction (setsid reparents claude away from this
# script), so the transcript probe is the ONLY positive evidence available. A
# real claude that is merely slow to write its first transcript (heavy load,
# cold cache) past SPAWN_STARTED_GRACE_SECS is flagged failed-to-start even
# though it is alive and will keep running detached -- a wrong terminal
# status (144 + registry failed-to-start), not merely a late one. There is no
# way to retract the flag once the caller has observed it. Callers on slow or
# loaded hosts should raise SPAWN_STARTED_GRACE_SECS accordingly; see
# SKILL.md's Verification section for the caller-facing note.
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
  # Deliberately do NOT assert a specific final exit code here: resolve_foreground
  # and resolve_detached's launch_rc!=0 branch both prefer the more specific,
  # pre-existing 127 launcher-failure verdict over this flag when both fire for
  # the same underlying cause (see _reconcile_started_flag_with_launcher_failure
  # below), so this process may ultimately exit 127, not EXIT_FAILED_TO_START.
  echo "spawn-claude.sh: FAILED-TO-START — claude never started within ${SPAWN_STARTED_GRACE_SECS}s (no transcript, no claude process); registry marked failed-to-start (a more specific launcher-failure verdict may still override this)." >&2
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
  # Deliberately do NOT consult _failed_to_start_pending here. By this point
  # the foreground launcher has already returned AND _wait_sentinel_grace
  # (bounded by SPAWN_LAUNCH_GRACE_SECS, checked every 0.1s) has exhausted
  # its own full wait with the sentinel still absent -- $inner's EXIT trap
  # installs before claude ever runs, so this unambiguously means the
  # payload never started: a genuine launcher failure (127) under the
  # pre-existing exit-code contract (task 2285/T3), independent of the
  # started-watchdog's own verdict. Unlike resolve_detached's await_sentinel()
  # (unbounded, so it genuinely needs the fts_marker as its own break
  # condition), nothing here is waiting on the marker, so a pending flag at
  # this exact instant only ever reflects a caller-misconfigured
  # SPAWN_STARTED_GRACE_SECS <= SPAWN_LAUNCH_GRACE_SECS racing the
  # started-watchdog ahead of this check -- not a real distinction from the
  # 127 case. Preferring 127 unconditionally keeps the foreground exit code
  # stable regardless of how the two independently-tunable grace windows
  # happen to be set.
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

if [ "${CLAUDE_SPAWN_BACKEND:-}" = "tmux" ]; then
  # Fleet Cockpit C6: the tmux lane is crash-survivable/reattachable and
  # needs no terminal emulator at all -- bypass discovery entirely.
  # first_word drives the case dispatch further down exactly like a
  # discovered $emulator would. Mirrors the no-emulator exit 126 below:
  # fail fast, before the watchdog fork, if the one thing this lane
  # actually needs is missing.
  if ! command -v tmux >/dev/null 2>&1; then
    echo "spawn-claude.sh: CLAUDE_SPAWN_BACKEND=tmux but tmux not found — install tmux or unset CLAUDE_SPAWN_BACKEND" >&2
    exit 126
  fi
  first_word="tmux-lane"
else
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
fi

# _cleanup + trap installed earlier (right after the mktemp calls near the
# top of the script), so spawn_ref can never leak -- including on the
# `exit 126` (no emulator found) path just above. Only the fork itself
# happens here, immediately before the emulator case dispatch.
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
  tmux-lane)
    # Fleet Cockpit C6: crash-survivable, reattachable tmux lane. A window
    # created via `tmux new-window`/`new-session` is owned by the tmux
    # server, not this script -- it outlives spawn-claude.sh (and the
    # launching claude session), can be reattached with
    # `tmux attach -t $tmux_session`, and the on-disk session-registry
    # record persists independently. `-P -F` prints the exact
    # "<session>:<window>" target on stdout, captured below.
    # `new-window`/`new-session` return immediately while the tmux server
    # owns the payload -- DETACHED semantics, so resolve_detached already
    # delivers the right sentinel/exit/failed-to-start lifecycle, identical
    # to the konsole/custom branches above.
    tmux_project="${CLAUDE_SPAWN_PROJECT:-}"
    if [ -z "$tmux_project" ]; then
      tmux_project="$(basename "${cwd%/}")"
    fi
    [ -z "$tmux_project" ] && tmux_project="fleet"
    tmux_session="${CLAUDE_SPAWN_TMUX_SESSION:-fleet-$tmux_project}"
    win_name="$title"
    [ -z "$win_name" ] && win_name="${CLAUDE_SPAWN_ROLE:-session}"
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
      tmux_target=$(tmux new-window -t "$tmux_session" -n "$win_name" -P -F '#{session_name}:#{window_index}' bash -c "$inner")
    else
      tmux_target=$(tmux new-session -d -s "$tmux_session" -n "$win_name" -P -F '#{session_name}:#{window_index}' bash -c "$inner")
    fi
    launch_rc=$?
    # Stamp the display best-effort once the window actually exists.
    # Purely additive/fail-soft (mirrors the launching/exit registry calls
    # above): must never alter launch_rc or this script's exit-code
    # contract, so faults are swallowed with `|| true` exactly like those
    # calls. Depends on step-2's `set-display` CLI verb.
    if [ "$launch_rc" -eq 0 ] && [ -n "$SESSION_RECORD_DIR" ] && command -v python3 >/dev/null 2>&1; then
      if [ -n "$tmux_target" ]; then
        python3 "$SESSION_REGISTRY_PY" set-display --record "$SESSION_RECORD_DIR" --kind tmux --tmux-target "$tmux_target" --wm-title "$win_name" || true
      else
        # `-P -F` returned no output despite launch_rc==0 (unexpected on
        # supported tmux versions). Skip the stamp rather than persist an
        # empty tmux_target, which would leave the record unreattachable.
        echo "spawn-claude.sh: warning: tmux reported no window target — skipping display stamp" >&2
      fi
    fi
    resolve_detached "$launch_rc"
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
