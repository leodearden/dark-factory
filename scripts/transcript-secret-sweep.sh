#!/usr/bin/env bash
# transcript-secret-sweep.sh -- the nightly transcript secret sweep (task 6820).
# Invoked by scripts/transcript-secret-sweep.service's ExecStart (systemd
# .timer-driven, nightly 02:30).
#
# Runs scripts/transcript_secret_sweep.py over every factory project's
# agent-transcript archive and ~/.claude/projects. The sweep matches the
# secret values ITS OWN environment holds, so this wrapper sources $REPO/.env
# (the same file the orchestrators load) before running it. Nothing here
# echoes an environment value.
#
# ALWAYS EXITS 0. A recurring oneshot that exits non-zero enters systemd
# `failed` state and stays there, silently stopping the timer (the lesson in
# scripts/fused-memory-flag-marker-sweep.sh). The sweep's own exit code is
# narrated as `sweep=<rc>`, and its JSON record line names what it could not do.
#
# TRANSCRIPT_SECRET_SWEEP_CMD overrides the interpreter prefix (tests inject a
# fake recorder there), and REPO is overridable so tests can point it at a tmp
# dir: the seam convention of scripts/exception-owner-sweep.sh.
set -uo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a

read -r -a SWEEP_CMD <<<"${TRANSCRIPT_SECRET_SWEEP_CMD:-uv run --frozen --project $REPO/shared python}"

echo "transcript-secret-sweep: sweeping agent transcripts for live secret values"
"${SWEEP_CMD[@]}" "$REPO/scripts/transcript_secret_sweep.py"
sweep_rc=$?
if [ "$sweep_rc" -ne 0 ]; then
    echo "transcript-secret-sweep: the sweep exited $sweep_rc -- it could not run, or its" \
         "escalation was not filed; its record line above says which" >&2
fi

echo "transcript-secret-sweep: done (sweep=$sweep_rc)"
exit 0
