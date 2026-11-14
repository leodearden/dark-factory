#!/usr/bin/env bash
# fused-memory-duplicate-audit.sh -- run the weekly duplicate-cluster report
# (fused-memory/scripts/duplicate_cluster_report.py) under the fused-memory
# service environment. ExecStart of scripts/fused-memory-duplicate-audit.service;
# the job is described in OPERATIONS.md §"Weekly duplicate-cluster report (Sun 06:30)".
#
# Seams: REPO (the checkout), DUPLICATE_AUDIT_CMD (replaces the
# `uv run --no-sync --project "$FM" python` prefix). --no-sync, not --frozen,
# so a timer run never re-syncs the shared root .venv
# (tests/scripts/test_uv_run_venv_isolation.py). Exits with the report's status.
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

if [ -n "${DUPLICATE_AUDIT_CMD:-}" ]; then
  # shellcheck disable=SC2206
  REPORT_CMD=(${DUPLICATE_AUDIT_CMD})
else
  UV_RESOLVED="$(require_uv_bin)" || exit $?
  REPORT_CMD=("$UV_RESOLVED" run --no-sync --project "$FM" python)
fi

exec "${REPORT_CMD[@]}" "$FM/scripts/duplicate_cluster_report.py" "$@"
