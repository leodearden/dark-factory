#!/usr/bin/env bash
# sync-orchestrator-env.sh — the ONLY sanctioned way to (re)materialize the
# orchestrator's shared runtime venv (dark-factory/.venv).
#
# Why this exists
# ---------------
# The three orchestrator units run `uv run --frozen ...`, so process start NEVER
# re-syncs the venv: a frozen start against a missing/stale venv fails fast
# instead of bootstrapping. Mutating the runtime env is therefore a deliberate,
# supervised operation — this script — performed with every orchestrator stopped
# so no live process is bound to an interpreter that is being rebuilt.
#
# This is the 2026-05-29 ghost-venv fix. Before it, the shared runtime venv could
# be mutated two ways: (a) a target's verify subprocess inheriting our
# VIRTUAL_ENV and running `uv sync` into OUR venv (fixed in verify.py's
# _target_subprocess_env scrub), and (b) a unit start implicitly re-syncing
# (fixed by --frozen). This script rebuilds the env on the .python-version pin
# (3.13), which also PURGES any leaked torch/insightface stack a pre-fix target
# verify wrote into our runtime, leaving a clean, lean orchestrator env.
#
# Order matters: stop the watchdog TIMER first (else its 60s probe revives a unit
# we just stopped, mid-sync), then the services; sync; then services; then the
# timer last. The watchdog only port-probes — it does NOT repair a missing/stale
# venv, so a frozen unit must be pre-synced by THIS script before it can start.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV=/home/leo/.local/bin/uv

SERVICES=(
  orchestrator-reify.service
  orchestrator-dark-factory.service
  orchestrator-autopilot-video.service
)

echo "==> Stopping watchdog timer (first, so it can't revive a unit mid-sync)"
systemctl --user stop orchestrator-watchdog.timer || true

echo "==> Stopping orchestrator services"
for svc in "${SERVICES[@]}"; do
  systemctl --user stop "$svc" || true
done

echo "==> Confirming no 'orchestrator run' processes remain"
PATTERN='orchestrator run --config'
for _ in $(seq 1 15); do
  pgrep -u "$(id -u)" -f "$PATTERN" >/dev/null 2>&1 || break
  echo "    ... a process still matches; waiting 1s"
  sleep 1
done
if pgrep -u "$(id -u)" -f "$PATTERN" >/dev/null 2>&1; then
  echo "ERROR: orchestrator processes still alive after stop; aborting sync." >&2
  echo "       (A non-systemd 'ghost' may be lingering — kill it, then re-run.)" >&2
  pgrep -au "$(id -u)" -f "$PATTERN" >&2 || true
  exit 1
fi

echo "==> Syncing runtime venv against uv.lock on the .python-version pin (3.13)"
cd "$REPO_ROOT"
"$UV" sync --project orchestrator

echo "==> Runtime interpreter (expect 3.13.x — fails loudly if a ghost reappears):"
"$REPO_ROOT/.venv/bin/python" --version

echo "==> Restarting orchestrator services"
for svc in "${SERVICES[@]}"; do
  systemctl --user reset-failed "$svc" 2>/dev/null || true
  systemctl --user start "$svc"
done

echo "==> Restarting watchdog timer (last)"
systemctl --user start orchestrator-watchdog.timer

echo "==> Done. Frozen units are running against a freshly-synced venv."
