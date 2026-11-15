#!/usr/bin/env bash
# shared-repo-git-maintenance.sh -- the nightly incremental repack +
# commit-graph of every orchestrator-managed shared repo (task 8076).
# Invoked by scripts/shared-repo-git-maintenance.service's ExecStart, and
# runnable by hand with scripts/shared_repo_git_maintenance.py's own flags,
# which are passed through.
#
# ALWAYS EXITS 0: a recurring oneshot that exits non-zero stays `failed` and
# silently stops its timer (the lesson in scripts/fused-memory-flag-marker-sweep.sh).
# The script's exit code is narrated as `maintenance=<rc>` instead.
set -uo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"

python3 "$REPO/scripts/shared_repo_git_maintenance.py" "$@"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "shared-repo-git-maintenance: maintenance exited $rc -- the records above name the failed repo" >&2
fi

echo "shared-repo-git-maintenance: done (maintenance=$rc)"
exit 0
