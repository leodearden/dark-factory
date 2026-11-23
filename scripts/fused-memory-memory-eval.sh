#!/usr/bin/env bash
# fused-memory-memory-eval.sh -- run the scheduled memory-eval runner
# (fused-memory/scripts/memory_eval_runner.py) under the fused-memory service
# environment. ExecStart of scripts/fused-memory-memory-eval.service; the job is
# described in OPERATIONS.md §"Scheduled memory evals".
#
# Seams: REPO (the checkout), MEMORY_EVAL_RUNNER_CMD (replaces the
# `uv run --no-sync --project "$FM" python` prefix). --no-sync, not --frozen,
# so a timer run never re-syncs the shared root .venv
# (tests/scripts/test_uv_run_venv_isolation.py). Exits with the runner's status.
set -euo pipefail

case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
_uv_lib="$_self_dir/lib/resolve_uv.sh"
# shellcheck source=lib/resolve_uv.sh
source "$_uv_lib" || { echo "${0##*/}: ERROR: cannot load the shared uv resolver $_uv_lib -- refusing to guess which uv to run." >&2; exit 127; }

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

if [ -n "${MEMORY_EVAL_RUNNER_CMD:-}" ]; then
  # shellcheck disable=SC2206
  RUNNER_CMD=(${MEMORY_EVAL_RUNNER_CMD})
else
  UV_RESOLVED="$(require_uv_bin)" || exit $?
  RUNNER_CMD=("$UV_RESOLVED" run --no-sync --project "$FM" python)
fi

exec "${RUNNER_CMD[@]}" "$FM/scripts/memory_eval_runner.py" "$@"
