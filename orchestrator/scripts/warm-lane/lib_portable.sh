#!/usr/bin/env bash
# Portable shell helpers for reify build scripts and infrastructure tests.
# Designed to be sourced, not executed directly.
#
# Usage:  source "$(dirname "${BASH_SOURCE[0]}")/lib_portable.sh"
#   or:   source "$REPO_ROOT/scripts/lib_portable.sh"

# Source guard — prevent double-sourcing.
if [ "${_REIFY_LIB_PORTABLE_SH_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || true
fi
_REIFY_LIB_PORTABLE_SH_SOURCED=1

# Portable SHA-256: prefer sha256sum (GNU coreutils), fall back to shasum (macOS).
portable_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1"
    else
        echo "ERROR: neither sha256sum nor shasum found on PATH." >&2
        return 1
    fi
}

# Allocate a free ephemeral TCP port by binding to port 0 and letting the OS
# choose.  Prints the port number (an integer in 1..65535) to stdout.
#
# Uses python3 socket bind-to-0, the same OS mechanism node uses in
# allocateFreePort() (gui/test/visual/endpoint.ts) and the exact one-liner
# already used in tests/infra/test_run_gui_scripts.sh Test 25.
#
# Returns non-zero with an error on stderr if python3 is unavailable.
#
# TOCTOU note: the socket is bound to port 0, the OS assigns a free port, then
# the socket is closed before returning.  A concurrent process can claim that
# port in the gap between close and the consumer's bind.  This mirrors the
# existing node allocateFreePort() behavior and is acceptable for the v1
# use-case; if stronger guarantees are needed, hold the socket open until the
# consumer binds, or retry on bind failure in run-gui-dev.sh.
allocate_free_port() {
    if ! command -v python3 >/dev/null 2>&1; then
        echo "ERROR: python3 not found on PATH; cannot allocate a free port." >&2
        return 1
    fi
    python3 -c '
import socket
s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()
print(port)
'
}

# Portable file mtime: print the file's modification time as a Unix epoch integer.
#
# Usage: portable_mtime <file>
#
# Prints the mtime as a decimal integer (seconds since the Unix epoch).
# Returns non-zero when the file is missing or unreadable (both stat forms fail).
#
# Idiom already open-coded in scripts/tree-sitter-generate.sh:95 and
# scripts/verify.sh:122 — intended to replace those inline copies; callers
# should source lib_portable.sh and call portable_mtime rather than
# re-inlining the stat idiom.
portable_mtime() {
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null
}

# Portable timeout: run a command with a wall-clock time limit.
#
# Usage: portable_timeout <seconds> <cmd> [args...]
#
# 3-tier strategy:
#   1. GNU timeout (Linux coreutils)
#   2. gtimeout (Homebrew coreutils on macOS)
#   3. background + sleep + kill fallback (POSIX-portable)
#
# Returns the command's exit code, or 124 if the command was killed
# due to exceeding the time limit (matches GNU timeout convention).
#
# Sets the global _PORTABLE_TIMEOUT_TIMED_OUT variable:
#   true  — the command was killed by the timeout mechanism
#   false — the command exited on its own (any exit code, including 124)
#
# POSIX fallback termination strategy:
#   The POSIX fallback sends SIGTERM first, then waits a 2-second grace period.
#   If the command has not exited, it escalates to SIGKILL via process-group
#   kill (kill -9 -- -$cmd_pid).  Process-group kill is PID-reuse safe: the
#   command has not been wait(2)ed on yet (the main shell is blocked), so the
#   PID cannot have been recycled.  The process-group kill also terminates any
#   orphaned child processes (e.g. a nested sleep inside a bash -c wrapper).
#
# Ambiguity note (GNU timeout / gtimeout paths):
#   When using GNU timeout or gtimeout, _PORTABLE_TIMEOUT_TIMED_OUT is set
#   to true whenever the exit code is 124.  However, exit 124 can also be
#   returned by the wrapped command itself (e.g. `bash -c 'exit 124'`).
#   There is no way to distinguish these two cases on the GNU path.
#   _PORTABLE_TIMEOUT_TIMED_OUT is only fully reliable on the POSIX
#   flag-file fallback path, where the flag is created by the timer process
#   and the command's exit code is additionally required to be 143 (SIGTERM).
portable_timeout() {
    local seconds="$1"
    shift

    _PORTABLE_TIMEOUT_TIMED_OUT=false
    local cmd_exit=0
    if command -v timeout >/dev/null 2>&1; then
        timeout "$seconds" "$@" || cmd_exit=$?
        # Ambiguity: exit 124 may mean the timeout fired OR the command
        # naturally exited 124.  See doc comment above for details.
        if [ "$cmd_exit" -eq 124 ]; then
            _PORTABLE_TIMEOUT_TIMED_OUT=true
        fi
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$seconds" "$@" || cmd_exit=$?
        # Same exit-124 ambiguity as the GNU timeout branch above.
        if [ "$cmd_exit" -eq 124 ]; then
            _PORTABLE_TIMEOUT_TIMED_OUT=true
        fi
    else
        # POSIX fallback: background the command, sleep, kill if still running.
        # Use a temp file flag so we can distinguish the timer's kill from
        # a coincidental exit code 143 or 124.
        local timeout_flag
        timeout_flag=$(mktemp "${TMPDIR:-/tmp}/portable_timeout.XXXXXX" 2>/dev/null) || timeout_flag=""
        local _pt_kill_grace=2  # SIGKILL grace period after SIGTERM in POSIX-fallback timer

        # Save caller's monitor-mode state BEFORE any set -m/set +m manipulation.
        # Placing this save/restore BEFORE the first set -m ensures that $- still
        # reflects the caller's original state; the subsequent set +m would clear
        # 'm' from $- if we saved after it.  Without this, portable_timeout
        # silently disables the caller's job control when they had set -m active.
        local _pt_had_monitor=0
        case $- in *m*) _pt_had_monitor=1 ;; esac

        # Start the command in its own process group (PGID=cmd_pid) so that
        # a later 'kill -- -$cmd_pid' can clean up any orphaned children (e.g.
        # if the timer escalates to SIGKILL, killing the command wrapper but
        # leaving its children running).  set +m immediately after to keep job
        # control disabled for the rest of the main shell's logic.
        set -m 2>/dev/null || true
        "$@" &
        local cmd_pid=$!
        set +m 2>/dev/null || true

        if [ -n "$timeout_flag" ]; then
            # Normal path: use flag file for precise timeout detection.
            rm -f "$timeout_flag"  # Remove so its presence signals timeout fired.
            # Trap INT/TERM so that if the caller's shell is interrupted during the
            # wait window the temp flag path is cleaned up.  Save and restore any
            # existing handler so we don't clobber the caller's traps.
            local _pt_old_int _pt_old_term
            _pt_old_int=$(trap -p INT 2>/dev/null || true)
            _pt_old_term=$(trap -p TERM 2>/dev/null || true)
            trap 'rm -f "$timeout_flag" 2>/dev/null' INT TERM
            # Enable job control temporarily so the timer subshell is placed in its
            # own process group, making 'kill -- -$timer_pid' safe and effective.
            set -m 2>/dev/null || true
            ( sleep "$seconds" && {
                touch "$timeout_flag" 2>/dev/null
                kill "$cmd_pid" 2>/dev/null
                # Grace period: if SIGTERM is ignored, escalate to SIGKILL via
                # process-group kill.  Safe: cmd_pid hasn't been wait(2)ed yet
                # (main shell is blocked), so no PID-reuse risk.  Process-group
                # kill also cleans up child processes (e.g. nested sleep).
                sleep "$_pt_kill_grace"
                kill -9 -- "-$cmd_pid" 2>/dev/null
              } ) &
            if [ "$_pt_had_monitor" -eq 0 ]; then set +m 2>/dev/null || true; fi
        else
            # Degraded path: mktemp failed, fall back to old 143-detection.
            echo "WARNING: mktemp failed, timeout detection degraded" >&2
            set -m 2>/dev/null || true
            ( sleep "$seconds" && {
                kill "$cmd_pid" 2>/dev/null
                sleep "$_pt_kill_grace"
                kill -9 -- "-$cmd_pid" 2>/dev/null
              } ) &
            if [ "$_pt_had_monitor" -eq 0 ]; then set +m 2>/dev/null || true; fi
        fi
        local timer_pid=$!
        wait "$cmd_pid" 2>/dev/null || cmd_exit=$?
        # Clean up timer — if command finished before timeout, kill the timer subshell
        # and its inner sleep child.  Process-group kill ('kill -- -$pid') terminates
        # all children of the subshell atomically; fall back to plain kill if unsupported.
        kill -- -$timer_pid 2>/dev/null || kill "$timer_pid" 2>/dev/null || true
        wait "$timer_pid" 2>/dev/null || true
        # Kill any orphaned children in the command's process group.  When the timer
        # escalates to SIGKILL it kills the command wrapper (e.g. a bash -c) but not
        # its children (e.g. a nested sleep).  'kill -9 -- -$cmd_pid' sends SIGKILL to
        # every process in the command's process group, cleaning up those orphans.
        # SIGKILL matches the timer's escalation: if we reached this point the command
        # already ignored SIGTERM, so re-sending SIGTERM here would be ineffective.
        # Safe after wait: a stale PGID returns ESRCH harmlessly (|| true handles it).
        kill -9 -- -$cmd_pid 2>/dev/null || true

        if [ -n "$timeout_flag" ] && [ -f "$timeout_flag" ] && { [ "$cmd_exit" -eq 143 ] || [ "$cmd_exit" -eq 137 ]; }; then
            # Timer fired and killed the process — genuine timeout.  Two cases:
            #   exit 143 (128+15): timer sent SIGTERM → command terminated normally.
            #   exit 137 (128+9):  timer escalated to SIGKILL (command ignored SIGTERM).
            # The flag file proves the timer touched it before killing; requiring the
            # exit code to be 143 or 137 closes the false-positive race where the
            # command exits naturally while the timer is between touch and kill.
            _PORTABLE_TIMEOUT_TIMED_OUT=true
            cmd_exit=124
            rm -f "$timeout_flag"
        elif [ -n "$timeout_flag" ]; then
            # Flag path: timer may have touched the flag but the command exited
            # naturally (not SIGTERM).  Clean up the stale flag file.
            rm -f "$timeout_flag"
        fi
        # Restore caller's INT/TERM traps (only set in the flag-file path above).
        if [ -n "$timeout_flag" ]; then
            if [ -n "$_pt_old_int" ]; then eval "$_pt_old_int"; else trap - INT; fi
            if [ -n "$_pt_old_term" ]; then eval "$_pt_old_term"; else trap - TERM; fi
        fi
        if [ -z "$timeout_flag" ] && { [ "$cmd_exit" -eq 143 ] || [ "$cmd_exit" -eq 137 ]; }; then
            # Degraded mode: 143 (SIGTERM) or 137 (SIGKILL escalation) likely means
            # our timer killed it.  Same logic as the flag-file path above.
            _PORTABLE_TIMEOUT_TIMED_OUT=true
            cmd_exit=124
        fi
    fi

    return "$cmd_exit"
}
