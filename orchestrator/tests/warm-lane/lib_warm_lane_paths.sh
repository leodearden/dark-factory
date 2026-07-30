#!/usr/bin/env bash
# orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
#
# The single script-under-test path resolver for dark-factory's ported
# warm-lane bash tests (task 3073, PRD leaf α2).
#
# All eight reify sources share one bootstrap idiom:
#
#     SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#     REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
#     SCRIPT="$REPO_ROOT/scripts/<name>.sh"
#
# In dark-factory the tests sit one level deeper (orchestrator/tests/warm-lane/)
# and the scripts sit at orchestrator/scripts/warm-lane/.  Hoisting the
# derivation here makes the new depth ONE edit point instead of eight
# independently-driftable ones, and keeps each ported file's diff against its
# reify source down to a two-line delta for the whole α→κ duplication window.
#
# Resolution is by PURE PATH ARITHMETIC from ${BASH_SOURCE[0]}: it reads no
# environment and invokes no `git rev-parse`.  Same discipline, and for the same
# reasons, as α's Delta 1 in orchestrator/scripts/warm-lane/README.md — an
# inherited GIT_DIR (the standard git-hook / `git rebase --exec` environment)
# returns this directory itself, a rev-parse probe can ascend into an enclosing
# repo, and it returns the symlink-resolved path where `..` returns the logical
# one (dark-factory's own `.worktrees` is a case where those disagree).  Every
# one of those failure modes lands in the same wrong-path class, and an
# existence guard catches none of them because every wrong path exists.
#
# In particular this file does NOT honour ORCH_WARM_LANE_SCRIPT_DIR.  The
# orchestrator suite's autouse `_isolate_warm_lane_script_dir` fixture pins that
# variable at a guaranteed-absent sentinel for EVERY test and it leaks into the
# subprocesses the pytest driver spawns; honouring it would run all eight ported
# tests against an empty directory — a vacuous green.  Pinned behaviourally by
# test_warm_lane_bash_suite.py::test_script_dir_resolution_ignores_the_env_override.

_WARM_LANE_PATHS_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# orchestrator/tests/warm-lane -> orchestrator/tests -> orchestrator -> <repo>
DF_REPO_ROOT="$(cd "$_WARM_LANE_PATHS_LIB_DIR/../../.." && pwd)"
WARM_LANE_SCRIPTS_DIR="$DF_REPO_ROOT/orchestrator/scripts/warm-lane"

# Fail loudly rather than let a mislaid harness degrade to a vacuous pass: a
# test whose $SCRIPT does not exist would otherwise just see every invocation
# exit 127 and could, for a permissive assertion, still report green.
if [ ! -d "$WARM_LANE_SCRIPTS_DIR" ]; then
    echo "ERROR: warm-lane script directory not found at $WARM_LANE_SCRIPTS_DIR" >&2
    echo "       (resolved from ${BASH_SOURCE[0]}, repo root $DF_REPO_ROOT)" >&2
    exit 2
fi
