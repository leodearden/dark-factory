#!/usr/bin/env bash
# fused-memory-link-heal.sh -- run one scheduled link-heal sweep
# (fused-memory/scripts/link_heal.py sweep) under the fused-memory service
# environment. ExecStart of scripts/fused-memory-link-heal.service; the job is
# described in OPERATIONS.md §"Nightly link-heal sweep (02:15)".
#
# Seams: REPO (the checkout), LINK_HEAL_SWEEP_CMD (replaces the
# `uv run --no-sync --project "$FM" python` prefix). --no-sync, not --frozen,
# so a timer run never re-syncs the shared root .venv
# (tests/scripts/test_uv_run_venv_isolation.py). Arguments pass through after
# `sweep`. Exits with the sweep's status.
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

if [ -n "${LINK_HEAL_SWEEP_CMD:-}" ]; then
  # shellcheck disable=SC2206
  SWEEP_CMD=(${LINK_HEAL_SWEEP_CMD})
else
  UV_RESOLVED="$(require_uv_bin)" || exit $?
  SWEEP_CMD=("$UV_RESOLVED" run --no-sync --project "$FM" python)
fi

exec "${SWEEP_CMD[@]}" "$FM/scripts/link_heal.py" sweep "$@"
