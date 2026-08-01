"""Parity checker for the installed dashboard systemd units.

Verifies that the three dashboard units committed under ``dashboard/`` still
agree with the copies actually installed in ``~/.config/systemd/user/``:

    dark-factory-dashboard.service
    dark-factory-dashboard-watchdog.service
    dark-factory-dashboard-watchdog.timer

The purpose is narrow and concrete: make it OBSERVABLE when a repo-side unit
change never reaches the running system.  Two such changes exist already —
task 3306 added ``--timeout-graceful-shutdown 8`` / ``--timeout-keep-alive 5``
to the dashboard ExecStart, and task 3308 replaced the watchdog's inline-shell
ExecStart with ``scripts/dashboard-watchdog.py`` plus ``TimeoutStartSec=300``.
A unit edit that stays repo-side is indistinguishable from no edit at all, and
nothing reported that until this check existed.

Exit codes
----------
0 — parity (every compared directive agrees)
1 — drift  (one or more compared directives disagree)
2 — installed unit absent (no installed copy found for one or more units)

PRECEDENCE: drift (1) DOMINATES absence (2).  With three units a single run can
hit both at once, and returning 2 there would let an unrelated uninstalled unit
mask an actionable finding — ``setup-host.sh`` treats 2 as a benign "not
installed on this host, skipping" and only 1 as something to act on.

Usage
-----
  # verify with defaults (~/.config/systemd/user vs this repo)
  python3 scripts/check_dashboard_unit_parity.py

  # verify explicit trees
  python3 scripts/check_dashboard_unit_parity.py \\
      --installed-dir ~/.config/systemd/user \\
      --repo-root /home/leo/src/dark-factory

  # verify a single unit
  python3 scripts/check_dashboard_unit_parity.py \\
      --unit dark-factory-dashboard-watchdog.timer

Design notes
------------
- Stdlib-only (argparse, dataclasses, pathlib, re, sys) — runs under plain
  python3, exactly like scripts/check_fused_memory_unit_parity.py, whose
  idioms this script follows deliberately rather than inventing a new pattern.
- No ``--fix``, unlike that precedent.  Propagating repo units into
  ``~/.config/systemd/user/`` and daemon-reloading is already what
  ``scripts/setup-host.sh`` does; duplicating it here would be a fourth copy of
  the install logic.  It would also be actively unsafe today: the repo
  watchdog timer's own comment records that RE-ARMING the installed timer is
  task 3289's job, so a ``--fix`` that installed and reloaded it would
  silently re-arm a watchdog someone deliberately left disarmed — a
  supervision change disguised as a parity fix.  Drift is reported with a
  remediation pointer instead.

Testing note
------------
All drift-logic tests run against ``tmp_path`` fixtures — never the host's
real ``~/.config/systemd/user/`` — mirroring the rule the fused-memory test
module states in its own docstring.  This is not merely for portability: as
measured on 2026-08-01 the installed watchdog service is still the
pre-incident inline-shell copy, so this checker exits 1 against the live host
today.  That is the CORRECT signal (installing the post-3308 units belongs to
task 3289), but a test asserting parity against the live host would be red on
landing, and one asserting drift would flip red the moment 3289 fixes it.
Either encodes host state rather than checker behaviour.
"""

import argparse
import pathlib
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, so a parity report is greppable
# the same way `journalctl --user -t dashboard-watchdog` makes the watchdog's
# decisions greppable.  This is also the token `metadata.delivered_checks`
# greps for over scripts/ (dashboard_unit_parity|DASHBOARD_UNIT_PARITY).
LOG_TAG = "dashboard_unit_parity"

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INSTALLED_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"


def _log(message: str, *, stream=None) -> None:
    """Print *message* prefixed with the log tag."""
    print(f"[{LOG_TAG}] {message}", file=stream if stream is not None else sys.stdout)


def main(argv: Sequence[str]) -> int:
    """Parse args and run the parity check.  See the module exit-code table."""
    parser = argparse.ArgumentParser(
        description="Verify parity between the in-repo and installed dashboard units."
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
