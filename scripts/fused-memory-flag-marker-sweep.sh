#!/usr/bin/env bash
# fused-memory-flag-marker-sweep.sh -- the committed nightly DRAIN action for
# stage1_flag_marker dead-weight records (task 2693, follow-up to task 2596's
# previously-unwired sweep). Invoked by
# scripts/fused-memory-flag-marker-sweep.service's ExecStart (systemd
# .timer-driven, nightly) and directly by
# scripts/install-flag-marker-sweep-timer.sh for the immediate one-time
# drain of the current backlog.
#
# Sets up the fused-memory service environment (sourced .env + CONFIG_PATH /
# PROJECT_ROOT / FALKORDB_URI roots) then runs sweep_orphan_flag_markers.py
# under `uv run` so `fused_memory.*` imports resolve -- mirrors the
# fused-memory/scripts/cgl_eta_auto_apply.sh runbook lesson: a fused-memory
# maintenance action must run under the SERVICE env, not a bare shell, or
# the census silently narrows.
#
# Runs `--apply --terminal-drain` WITHOUT `--check` on purpose: the sweep's
# own docstring/WARNING notes undated markers can never be drained by
# find_stale_markers, so a `--check --max-backlog 0` recurring service would
# enter systemd `failed` state forever whenever any undated marker exists.
# Backlog visibility is instead left to the existing reconciliation
# Stage-1/2 re-flag net (the mechanism that filed this task).
#
# FLAG_MARKER_SWEEP_CMD overrides the default `uv run --frozen --project
# $FM python` interpreter prefix (tests inject a fake recorder here to
# assert the sweep is invoked correctly, without uv/live stores) --
# mirrors install-trickle-timer.sh's INSTALL_TRICKLE_TIMER_PYTHON /
# watcher-rearm.sh's WATCHER_REARM_PYTHON override convention. REPO is
# similarly overridable so tests can point it at a tmp dir with no `.env`
# (a no-op source). FLAG_MARKER_SWEEP_PROJECT_IDS (whitespace-separated)
# overrides the per-project sweep list -- see the loop at the bottom.
#
# Task 2917 EDIT 3 -- `uv` is resolved to an ABSOLUTE path below rather than
# trusted to be on PATH. OBSERVED (journalctl --user -u
# fused-memory-flag-marker-sweep.service):
#
#   Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.sh[65377]:
#       .../fused-memory-flag-marker-sweep.sh: line 46: exec: uv: not found
#   Aug 18 09:02:44 ... fused-memory-flag-marker-sweep.service:
#       Main process exited, code=exited, status=127/n/a
#
# That line is immediately preceded by a `-- Boot ... --` marker, and the
# next normal timer firing (Aug 19 03:34:58) succeeded -- so the failure is
# specific to the unit's `Persistent=true` BOOT CATCH-UP run, which fires
# before the login session pushes the user PATH into the systemd user
# manager. `uv` lives in $HOME/.local/bin, absent from that minimal boot
# PATH. UV_BIN overrides the resolution outright (the test seam); otherwise
# `command -v uv` wins, then the measured real location, then
# /usr/local/bin. An unresolvable `uv` is reported LOUDLY (an ERROR: line
# naming uv, the PATH searched, and the boot-catch-up cause) rather than
# left as a bare shell 127 that says nothing about why.
#
# This is belt-and-braces with the `Environment=PATH=` line the .service
# unit now carries: the unit-level PATH covers everything else the wrapper
# shells out to, while this wrapper-level resolution survives a STALE
# installed unit that predates that line.
set -euo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

resolve_uv_bin() {
  # Order: explicit override, then PATH, then the two known install roots.
  if [ -n "${UV_BIN:-}" ] && [ -x "${UV_BIN}" ]; then
    printf '%s' "${UV_BIN}"
    return 0
  fi
  local from_path
  if from_path="$(command -v uv 2>/dev/null)" && [ -x "$from_path" ]; then
    printf '%s' "$from_path"
    return 0
  fi
  local candidate
  for candidate in "$HOME/.local/bin/uv" /usr/local/bin/uv; do
    if [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

if [ -n "${FLAG_MARKER_SWEEP_CMD:-}" ]; then
  # The documented test seam: an unquoted expansion so a multi-word prefix
  # word-splits into the array.
  # shellcheck disable=SC2206
  SWEEP_CMD=(${FLAG_MARKER_SWEEP_CMD})
else
  if ! UV_RESOLVED="$(resolve_uv_bin)"; then
    echo "fused-memory-flag-marker-sweep.sh: ERROR: cannot resolve \`uv\` -- not at \$UV_BIN (${UV_BIN:-unset}), not on PATH (${PATH}), and not at \$HOME/.local/bin/uv or /usr/local/bin/uv. This is the \`exec: uv: not found\` / status=127 boot-catch-up failure OBSERVED 2026-08-18 09:02:44: the unit's Persistent=true catch-up run fires before the login session pushes the user PATH into the systemd user manager. Install uv, or set UV_BIN to its absolute path." >&2
    exit 127
  fi
  # Built literally (not via a \${X:-...} default inside an unquoted array
  # expansion) so "$FM" survives verbatim even when the repo path contains
  # spaces -- see test_wrapper_default_prefix_invokes_uv_run_frozen_project.
  SWEEP_CMD=("$UV_RESOLVED" run --frozen --project "$FM" python)
fi

# --- Per-project sweep loop (task 2917 EDIT 1) ------------------------------
#
# This used to be a single `exec ... --apply --terminal-drain` with no
# --project-id, so it rode the sweep parser's own `dark_factory` default while
# the per-project census it printed read as if the whole registered fleet had
# been drained. Every registered project now gets its own invocation.
#
# FLAG_MARKER_SWEEP_PROJECT_IDS (whitespace-separated) is the explicit
# override, mirroring the FLAG_MARKER_SWEEP_CMD / REPO convention above.
#
# `exec` is deliberately DROPPED and the loop deliberately does NOT `set -e`
# out on the first failure: one project's sweep failing must not silently
# truncate the fleet. Each failure is named on stderr with its exit code, the
# remaining projects are still attempted, and the wrapper exits non-zero
# overall so a PARTIAL nightly drain is loud rather than swallowed.
PROJECT_IDS=()
if [ -n "${FLAG_MARKER_SWEEP_PROJECT_IDS:-}" ]; then
  # shellcheck disable=SC2206
  PROJECT_IDS=(${FLAG_MARKER_SWEEP_PROJECT_IDS})
fi

if [ "${#PROJECT_IDS[@]}" -eq 0 ]; then
  PROJECT_IDS=(dark_factory)
fi

overall_status=0
for project_id in "${PROJECT_IDS[@]}"; do
  # Census honesty: name every project actually attempted, so coverage is
  # readable straight off the journal. --terminal-drain is REQUESTED
  # uniformly; the sweep itself decides and logs the EFFECTIVE mode, because
  # terminal ids come from the one task store this process is configured with
  # and are applied to that primary project ONLY -- every other project
  # narrows to an age-only sweep (task 2917, esc-2917-3 ruling; the guard is
  # _resolve_terminal_task_ids in sweep_orphan_flag_markers.py).
  echo "fused-memory-flag-marker-sweep.sh: sweeping project_id=$project_id (requested: --apply --terminal-drain; the sweep logs the EFFECTIVE mode, which narrows to age-only for every non-primary project)"
  status=0
  "${SWEEP_CMD[@]}" "$FM/scripts/sweep_orphan_flag_markers.py" \
    --apply --terminal-drain --project-id "$project_id" || status=$?
  if [ "$status" -ne 0 ]; then
    echo "fused-memory-flag-marker-sweep.sh: ERROR: sweep FAILED for project_id=$project_id (exit $status). Continuing with the remaining projects; the wrapper exits non-zero overall so this partial drain is not swallowed." >&2
    overall_status="$status"
  fi
done

exit "$overall_status"
