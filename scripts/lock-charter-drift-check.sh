#!/usr/bin/env bash
# lock-charter-drift-check.sh -- the nightly cross-repo lock-charter drift
# check (task 11623). Invoked by scripts/lock-charter-drift-check.service's
# ExecStart (systemd .timer-driven, nightly 01:45).
#
# Runs scripts/lock_charter_drift_heal.py, which reconciles reify's
# lock-charter declarations against $REPO's and routes the outcome: one heal
# task for drift, one escalate_info when it could not look. Everything the
# check decides is the script's; this wrapper only gives it its environment.
# This comparison used to sit on dark-factory's merge gate, where a reify-only
# commit reddened every unrelated merge; see OPERATIONS.md section 9a.
#
# ALWAYS EXITS 0. A recurring oneshot that exits non-zero enters systemd
# `failed` state and stays there, silently stopping the timer (the lesson in
# scripts/fused-memory-flag-marker-sweep.sh). The router's own failure is not
# hidden by this: it is narrated to stderr, and its exit code as `heal=<rc>`.
#
# LOCK_CHARTER_DRIFT_CHECK_CMD overrides the interpreter prefix (tests inject a
# fake recorder there), REPO is overridable so tests can point it at a tmp dir
# with no `.env`, and REIFY_REPO names the reify checkout to compare against:
# the seam convention of scripts/memory-metadata-coverage-census.sh. The
# default prefix passes --no-sync, not --frozen, so a timer run never re-syncs
# the shared root .venv (tests/scripts/test_uv_run_venv_isolation.py).
set -uo pipefail

# The check runs `git ls-files` under the reify checkout, and an inherited
# GIT_DIR or one of its siblings would redirect it into another repository.
# The full rationale is on the same block in
# scripts/memory-metadata-coverage-census.sh.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR \
      GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE \
      GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS

REPO="${REPO:-/home/leo/src/dark-factory}"
REIFY_REPO="${REIFY_REPO:-/home/leo/src/reify}"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a

read -r -a HEAL_CMD <<<"${LOCK_CHARTER_DRIFT_CHECK_CMD:-uv run --no-sync --project $REPO/shared python}"

echo "lock-charter-drift-check: reconciling $REIFY_REPO against $REPO"
"${HEAL_CMD[@]}" "$REPO/scripts/lock_charter_drift_heal.py" --project-root "$REPO" --reify-repo "$REIFY_REPO"
heal_rc=$?
if [ "$heal_rc" -ne 0 ]; then
    echo "lock-charter-drift-check: the heal router exited $heal_rc -- its stderr names the" \
         "tool call that failed, so this outcome reached no owner" >&2
fi

echo "lock-charter-drift-check: done (heal=$heal_rc)"
exit 0
