#!/usr/bin/env bash
# exception-owner-sweep.sh -- the nightly INV-12 owner sweep (task 5608).
# Invoked by scripts/exception-owner-sweep.service's ExecStart (systemd
# .timer-driven, nightly 06:00).
#
# Runs scripts/exception_owner_sweep.py against $REPO's own task store and
# stamp. Everything the sweep decides -- which owners are dead, what it files,
# what it could not measure -- is the script's; this wrapper only gives it its
# environment.
#
# ALWAYS EXITS 0. A recurring oneshot that exits non-zero enters systemd
# `failed` state and stays there, silently stopping the timer (the lesson in
# scripts/fused-memory-flag-marker-sweep.sh). The sweep's own failure is not
# hidden by this: it escalates once and leaves its stamp's last_run
# unadvanced, so the stamp's age under `exception_register.py --report` is the
# liveness signal. The sweep's exit code is narrated as `sweep=<rc>`.
#
# EXCEPTION_OWNER_SWEEP_CMD overrides the interpreter prefix (tests inject a
# fake recorder there), and REPO is overridable so tests can point it at a tmp
# dir with no `.env`: the seam convention of
# scripts/memory-metadata-coverage-census.sh.
set -uo pipefail

# The register runs `git ls-files` under $REPO, and an inherited GIT_DIR or
# one of its siblings would redirect it into another repository. The full
# rationale is on the same block in scripts/memory-metadata-coverage-census.sh.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR \
      GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE \
      GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS

REPO="${REPO:-/home/leo/src/dark-factory}"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a

read -r -a SWEEP_CMD <<<"${EXCEPTION_OWNER_SWEEP_CMD:-uv run --frozen --project $REPO/shared python}"

echo "exception-owner-sweep: sweeping the owners cited in $REPO"
"${SWEEP_CMD[@]}" "$REPO/scripts/exception_owner_sweep.py" --project-root "$REPO"
sweep_rc=$?
if [ "$sweep_rc" -ne 0 ]; then
    echo "exception-owner-sweep: the sweep exited $sweep_rc -- its result line names the" \
         "failed stage; it has escalated and left its stamp's last_run unadvanced" >&2
fi

echo "exception-owner-sweep: done (sweep=$sweep_rc)"
exit 0
