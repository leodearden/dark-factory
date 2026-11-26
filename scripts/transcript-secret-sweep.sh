#!/usr/bin/env bash
# transcript-secret-sweep.sh -- the nightly transcript secret sweep (task 6820).
# Invoked by scripts/transcript-secret-sweep.service's ExecStart (systemd
# .timer-driven, nightly 02:30).
#
# Runs scripts/transcript_secret_sweep.py over the trees
# scripts/transcript_secret_sweep.py::swept_trees names. The sweep matches the
# secret values ITS OWN environment holds, so this wrapper sources $REPO/.env
# (the same file the orchestrators load) before running it. Nothing here
# echoes an environment value.
#
# Arguments are forwarded to the sweep: leo-laptop's drop-in
# (scripts/transcript-secret-sweep.laptop.conf) adds --host-configs-dir and
# --no-escalation, and a hand run can add --dry-run.
#
# ALWAYS EXITS 0. A recurring oneshot that exits non-zero enters systemd
# `failed` state and stays there, silently stopping the timer (the lesson in
# scripts/fused-memory-flag-marker-sweep.sh). The sweep's own exit code is
# narrated as `sweep=<rc>`, and its JSON record line names what it could not do.
#
# TRANSCRIPT_SECRET_SWEEP_CMD overrides the interpreter prefix (tests inject a
# fake recorder there), and REPO is overridable so tests can point it at a tmp
# dir: the seam convention of scripts/exception-owner-sweep.sh. The default
# prefix runs the uv that scripts/lib/resolve_uv.sh resolves, and passes
# --no-sync, not --frozen, so a timer run never re-syncs the shared root .venv
# (tests/scripts/test_uv_run_venv_isolation.py).
set -uo pipefail

case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
_uv_lib="$_self_dir/lib/resolve_uv.sh"
# shellcheck source=lib/resolve_uv.sh
if ! source "$_uv_lib"; then
    echo "${0##*/}: ERROR: cannot load the shared uv resolver $_uv_lib -- refusing to guess which uv to run." >&2
    echo "transcript-secret-sweep: done (sweep=127)"
    exit 0
fi

REPO="${REPO:-/home/leo/src/dark-factory}"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a

sweep_rc=0
if [ -n "${TRANSCRIPT_SECRET_SWEEP_CMD:-}" ]; then
    read -r -a SWEEP_CMD <<<"$TRANSCRIPT_SECRET_SWEEP_CMD"
elif UV_RESOLVED="$(require_uv_bin)"; then
    SWEEP_CMD=("$UV_RESOLVED" run --no-sync --project "$REPO/shared" python)
else
    sweep_rc=127
fi

if [ "$sweep_rc" -eq 0 ]; then
    echo "transcript-secret-sweep: sweeping for live secret values"
    "${SWEEP_CMD[@]}" "$REPO/scripts/transcript_secret_sweep.py" "$@"
    sweep_rc=$?
fi
if [ "$sweep_rc" -ne 0 ]; then
    echo "transcript-secret-sweep: the sweep exited $sweep_rc -- it could not run, could not cover" \
         "every required root, or an escalation was not filed; the lines above say which" >&2
fi

echo "transcript-secret-sweep: done (sweep=$sweep_rc)"
exit 0
