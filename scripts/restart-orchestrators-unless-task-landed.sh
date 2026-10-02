#!/usr/bin/env bash
set -euo pipefail

# Restart the orchestrator fleet ONLY IF a named task has not yet landed on main.
#
# Usage:
#   restart-orchestrators-unless-task-landed.sh TASK_ID [args forwarded to restart-all-orchestrators.sh]
#
# Exit 0 having restarted NOTHING when TASK_ID is already on main; otherwise
# exec scripts/restart-all-orchestrators.sh with any extra args (callers pass
# --drain so the per-unit merge-drain gate applies). Diagnostics go to stderr;
# the exit code is the signal.
#
# INTENDED CALLER: a task_kind='deterministic' deploy task's `before_done.script`,
# with `target_unit='orchestrator-dark-factory.service'` so DeterministicRunner
# routes through its cgroup-escaping detached `systemd-run --user` path. The
# constraint is inherited wholesale from restart-all-orchestrators.sh: a BLOCKING
# invocation from inside an orchestrator's own cgroup is killed mid-script under
# KillMode=control-group. This wrapper adds a condition; it changes nothing else.
#
# WHY A WRAPPER RATHER THAN A CONDITION AT THE CALL SITE: a deploy milestone that
# exists to make a restart-only config change live is pointless if the task it
# gates has already landed by other means, and a gratuitous restart kills every
# in-flight agent. The condition belongs in exactly one place (SPOT).
#
# Overrides: MAIN_BRANCH (default main), BRANCH_PREFIX (default task/),
# RESTART_SCRIPT (default the sibling restart-all-orchestrators.sh) — the last
# exists so the tests can inject a stub instead of touching live units.

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
BRANCH_PREFIX="${BRANCH_PREFIX:-task/}"
RESTART_SCRIPT="${RESTART_SCRIPT:-$SCRIPT_DIR/restart-all-orchestrators.sh}"

if [ "$#" -lt 1 ] || [ -z "${1:-}" ]; then
  echo "usage: $(basename "$0") TASK_ID [restart args...]" >&2
  exit 2
fi
readonly TASK_ID="$1"
shift

# Resolve the repo from the script's own location, never the caller's cwd: the
# transient systemd unit runs under the user manager, whose cwd defaults to $HOME.
cd -- "$SCRIPT_DIR/.."

# Is TASK_ID landed on $MAIN_BRANCH? Mirrors the canonical ladder; the arm order
# is load-bearing.
task_is_landed() {
  local branch="${BRANCH_PREFIX}${TASK_ID}"

  # Arm 1 — exact-subject merge marker. FIRST because merge cleanup DELETES a
  # landed branch, and an ancestry probe against a missing ref exits 128, which
  # the two-way `&& echo landed || echo not` idiom silently reports as not-landed
  # — inverting the truth for the most common post-merge state.
  local marker
  if marker="$(git log "$MAIN_BRANCH" --fixed-strings \
        --grep="Merge ${branch} into ${MAIN_BRANCH}" \
        --max-count=1 --format=%H 2>/dev/null)" && [ -n "$marker" ]; then
    echo "landed: found merge marker ${marker} for ${branch}" >&2
    return 0
  fi

  # Arm 2 — ancestry, only while the ref still resolves. rc is captured and
  # echoed so all three outcomes stay distinguishable (0 landed, 1 not landed,
  # 128 ref missing); --is-ancestor prints nothing on either 0 or 1.
  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    local rc=0
    git merge-base --is-ancestor "$branch" "$MAIN_BRANCH" || rc=$?
    echo "ancestry rc=${rc} for ${branch}" >&2
    if [ "$rc" -eq 0 ]; then
      echo "landed: ${branch} is an ancestor of ${MAIN_BRANCH}" >&2
      return 0
    fi
    # rc=1 (genuinely not on main) and rc=128 (ref vanished mid-check) both
    # fall through to NOT LANDED, which is the fail-safe direction.
  fi

  return 1
}

# Fail SAFE: treat any error, and any unresolvable task, as NOT LANDED. A missed
# restart silently strands the gated task; a surplus restart is merely wasteful.
if task_is_landed; then
  echo "task ${TASK_ID} is already on ${MAIN_BRANCH} — restarting nothing." >&2
  exit 0
fi

echo "task ${TASK_ID} is NOT on ${MAIN_BRANCH} — restarting orchestrators." >&2
exec "$RESTART_SCRIPT" "$@"
