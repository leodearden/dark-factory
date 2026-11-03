#!/usr/bin/env bash
# shellcheck shell=bash
# scripts/lib/resolve_uv.sh -- the ONE copy of the ladder that resolves `uv` to
# an absolute path for every committed wrapper that runs it. Sourced, never
# executed: it defines functions only. It sets no shell options, has no
# top-level side effects and never calls `exit`, because whatever it did would
# happen inside the caller's own shell.
#
# WHY uv IS NEVER TRUSTED TO BE ON PATH. A Persistent=true timer's boot
# catch-up run fires before the login session pushes the user PATH into the
# systemd user manager, so a bare `uv` dies `exec: uv: not found` /
# status=127. That was observed on 2026-08-18 (task 2917). The journal excerpt
# lives in scripts/fused-memory-flag-marker-sweep.service.
#
# THE LADDER, first hit wins:
#   1. $UV_BIN -- an explicit operator or test pin;
#   2. `command -v uv` -- whatever PATH the caller actually has;
#   3. ${HOME}/.local/bin/uv -- the measured real install location, absent
#      from the minimal boot PATH;
#   4. /usr/local/bin/uv.
# A UV_BIN that is SET but not executable (typo, stale path, lost exec bit)
# stops the ladder: it never falls through, because quietly running some
# other uv than the one pinned is the silent degradation this ladder exists
# to prevent.
#
# Everything here is a bash builtin, so it works under an empty or minimal
# PATH. That is also why the diagnostics name the caller by "${0##*/}" (inside
# `$(...)`, $0 is still the caller's own script path) rather than by forking
# basename.
#
# USAGE, after the caller's own `set -a; source .env; set +a` so a UV_BIN or
# PATH set there is honored:
#
#   case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
#   _uv_lib="$_self_dir/lib/resolve_uv.sh"
#   source "$_uv_lib" || { echo "${0##*/}: ERROR: cannot load ... $_uv_lib ..." >&2; exit 127; }
#   ...
#   UV_RESOLVED="$(require_uv_bin)" || exit $?
#
# Locate this file relative to the CALLER, never via `$(dirname ...)` (it is
# unfindable under an empty PATH and silently yields "") or via a repo path.

# The silent probe. rc 0: the resolved path on stdout. rc 1: nothing found
# anywhere. rc 2: UV_BIN is set but not executable. rc 2 is kept apart from
# the generic miss so a caller can name the bad pin.
resolve_uv_bin() {
  if [ -n "${UV_BIN:-}" ]; then
    if [ -x "${UV_BIN}" ]; then
      printf '%s' "${UV_BIN}"
      return 0
    fi
    return 2
  fi
  local from_path
  if from_path="$(command -v uv 2>/dev/null)" && [ -x "$from_path" ]; then
    printf '%s' "$from_path"
    return 0
  fi
  local candidate
  for candidate in "${HOME:-}/.local/bin/uv" /usr/local/bin/uv; do
    if [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# The loud form every caller uses. On success, the resolved path on stdout.
# On failure, one `<caller>: ERROR:` line on stderr and rc 127, the code a
# bare missing `uv` would have produced, so the journal status is unchanged.
require_uv_bin() {
  local resolved rc=0
  resolved="$(resolve_uv_bin)" || rc=$?
  case "$rc" in
    0)
      printf '%s' "$resolved"
      return 0
      ;;
    2)
      echo "${0##*/}: ERROR: \$UV_BIN is set to '${UV_BIN}' but that path is not executable (bad path, or the exec bit is missing). Refusing to fall back to PATH / \$HOME/.local/bin/uv / /usr/local/bin/uv, because that would run a DIFFERENT \`uv\` than the one you pinned. Fix the path or unset UV_BIN." >&2
      ;;
    *)
      echo "${0##*/}: ERROR: cannot resolve \`uv\` -- UV_BIN unset, not on PATH (${PATH:-}), and not at \$HOME/.local/bin/uv or /usr/local/bin/uv. A systemd timer's boot catch-up run fires before the login session pushes the user PATH into the user manager, which is how a bare \`uv\` dies with status=127. Install uv, or set UV_BIN to its absolute path." >&2
      ;;
  esac
  return 127
}
