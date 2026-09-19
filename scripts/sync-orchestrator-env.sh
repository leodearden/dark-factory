#!/usr/bin/env bash
# sync-orchestrator-env.sh — the ONLY sanctioned way to (re)materialize the
# orchestrator's shared runtime venv (dark-factory/.venv).
#
# Why this exists
# ---------------
# The seven orchestrator units run `uv run --no-sync ...`, so process start NEVER
# installs into the venv: a start against a missing/stale venv fails with
# ModuleNotFoundError instead of bootstrapping one. Mutating the runtime env is
# therefore a deliberate, supervised operation — this script — performed with every
# orchestrator stopped so no live process is bound to an interpreter being rebuilt.
#
# CORRECTION (task 5553). This header said the units run `uv run --frozen` and that
# THAT is what stops a start re-syncing. Both were false, and were believed from the
# 2026-05-29 ghost-venv fix until 2026-09-19. `--frozen` is a LOCKFILE option ("run
# without updating the uv.lock file"); measured on uv 0.11.6, `uv run --frozen`
# REINSTALLED a package deleted from the venv ("Installed 1 package in 50ms") while
# `--no-sync` left it untouched. So for four months the units were mutating the
# shared venv at every start and this script's premise was wrong.
#
# That fix addressed two hazards. (a) A target's verify subprocess inheriting our
# VIRTUAL_ENV and running `uv sync` into OUR venv — genuinely fixed, in verify.py's
# _target_subprocess_env scrub, and still live. (b) A unit start implicitly
# re-syncing — NOT fixed by --frozen; fixed by --no-sync in task 5553. This script
# rebuilds the env on the .python-version pin (3.13), which also PURGES any leaked
# torch/insightface stack a pre-fix target verify wrote into our runtime, leaving a
# clean, lean orchestrator env.
#
# This script is the repair path precisely BECAUSE the units can no longer repair
# themselves. While a start silently re-synced, a half-synced venv healed itself and
# this script was one option among several; it is now the only one, as the top of
# this header has always claimed. Fleet-wide enforcement of the unit side lives in
# tests/scripts/test_uv_run_venv_isolation.py.
#
# Order matters: stop the watchdog TIMER first (else its 60s probe revives a unit
# we just stopped, mid-sync), then the services; sync; then services; then the
# timer last. The watchdog only port-probes — it does NOT repair a missing/stale
# venv, so a unit must be pre-synced by THIS script before it can start.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV=/home/leo/.local/bin/uv

# Every committed scripts/orchestrator-*.service unit EXCEPT orchestrator-watchdog
# (the probe, whose TIMER is stopped first and started last below). All of them run
# out of the one shared .venv this script rebuilds, so one left running through the
# sync is bound to an interpreter being replaced underneath it, and one never
# restarted afterwards keeps stale bytecode -- which --no-sync means it can no longer
# fix by re-syncing at its next start. This array is the single source of the list
# within this script; tests/scripts/test_sync_orchestrator_env.py derives the expected
# set from the committed units, so an eighth unit turns that guard red on its own.
SERVICES=(
  orchestrator-reify.service
  orchestrator-dark-factory.service
  orchestrator-autopilot-video.service
  orchestrator-know-live.service
  orchestrator-my-solar-challenge.service
  orchestrator-pump-web-ui.service
  orchestrator-solar-challenge-platform.service
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

echo "==> Syncing the WHOLE workspace against uv.lock on the .python-version pin (3.13)"
cd "$REPO_ROOT"
# --all-packages, NEVER a member-scoped --project/--package. Measured on uv
# 0.11.6: `uv sync --project <member>` is EXACT and uninstalls every package no
# selected member needs -- it removed a sibling workspace member and its
# dependency from the shared root .venv. That made the script that exists to
# REPAIR the shared venv prune it instead. `uv sync --inexact --project <member>`
# does not prune, but it also cannot install the other members' deps, so it
# cannot repair the one .venv the seven orchestrators, the dashboard, the load
# sampler and fused-memory all resolve against. --all-packages is the only form
# that does the job. Same norm SETUP.md states from the other direction
# ("Always --all-packages, never a bare uv sync", task 4539).
"$UV" sync --all-packages

echo "==> Runtime interpreter (expect 3.13.x — fails loudly if a ghost reappears):"
"$REPO_ROOT/.venv/bin/python" --version

echo "==> Restarting orchestrator services"
# The installed-check is load-bearing under `set -euo pipefail`, NOT defensive
# tidiness. The stop loop above tolerates absence via `|| true`, but a bare
# `systemctl start` on a host carrying only a SUBSET of these units would abort
# the script mid-restart and leave the remaining orchestrators DOWN -- strictly
# worse than the drift this list widening fixes. So skip what is not installed,
# loudly and by name, making a subset host a reported fact rather than a silent
# one. A genuine start failure for a unit that IS installed is still fatal.
for svc in "${SERVICES[@]}"; do
  if [[ ! -f "$HOME/.config/systemd/user/$svc" ]]; then
    echo "    SKIP $svc — not installed on this host (no ~/.config/systemd/user/$svc)"
    continue
  fi
  systemctl --user reset-failed "$svc" 2>/dev/null || true
  systemctl --user start "$svc"
done

echo "==> Restarting watchdog timer (last)"
systemctl --user start orchestrator-watchdog.timer

echo "==> Done. Units are running against a freshly-synced workspace venv."
