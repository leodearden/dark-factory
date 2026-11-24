#!/usr/bin/env bash
# timeout-headroom-selfcheck.sh -- the nightly per-test timeout headroom and
# leg-budget self-check (task 10940). Invoked by
# scripts/timeout-headroom-selfcheck.service's ExecStart (systemd
# .timer-driven, nightly 01:30).
#
# Runs scripts/timeout_headroom_selfcheck.py against $REPO's own verify-log
# archive, code and stamp. Everything the check decides -- which anchors and
# legs breach, what it files, what it could not measure -- is the script's;
# this wrapper only gives it its environment.
#
# ALWAYS EXITS 0. A recurring oneshot that exits non-zero enters systemd
# `failed` state and stays there, silently stopping the timer (the lesson in
# scripts/fused-memory-flag-marker-sweep.sh). The check's own failure is not
# hidden by this: its stamp records ok false and the stage that failed. The
# check's exit code is narrated as `selfcheck=<rc>`.
#
# TIMEOUT_HEADROOM_SELFCHECK_CMD overrides the interpreter prefix (tests inject
# a fake recorder there), and REPO is overridable so tests can point it at a
# tmp dir with no `.env`: the seam convention of
# scripts/memory-metadata-coverage-census.sh. The default prefix passes
# --no-sync, not --frozen, so a timer run never re-syncs the shared root .venv
# (tests/scripts/test_uv_run_venv_isolation.py).
set -uo pipefail

# The check runs `git ls-files` and `git cat-file` under $REPO, and an
# inherited GIT_DIR or one of its siblings would redirect them into another
# repository. The full rationale is on the same block in
# scripts/memory-metadata-coverage-census.sh.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR \
      GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE \
      GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS

REPO="${REPO:-/home/leo/src/dark-factory}"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a

read -r -a SELFCHECK_CMD <<<"${TIMEOUT_HEADROOM_SELFCHECK_CMD:-uv run --no-sync --project $REPO/shared python}"

echo "timeout-headroom-selfcheck: re-measuring the timeout anchors and leg budgets of $REPO"
"${SELFCHECK_CMD[@]}" "$REPO/scripts/timeout_headroom_selfcheck.py" --project-root "$REPO"
selfcheck_rc=$?
if [ "$selfcheck_rc" -ne 0 ]; then
    echo "timeout-headroom-selfcheck: the self-check exited $selfcheck_rc -- its stderr names the" \
         "failed stage, and its stamp records ok false" >&2
fi

echo "timeout-headroom-selfcheck: done (selfcheck=$selfcheck_rc)"
exit 0
