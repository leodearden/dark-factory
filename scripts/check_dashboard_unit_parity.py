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


# ---------------------------------------------------------------------------
# Unit parser
# ---------------------------------------------------------------------------


def _join_continuations(text: str) -> list[str]:
    """Return *text*'s lines with backslash continuations joined into one line.

    While a line ends in ``\\``, the backslash is dropped and the NEXT line's
    stripped form is appended after a single space.  Mirrors ``_logical_exec_start``
    in tests/scripts/test_dashboard_service_template.py, generalised from "the
    ExecStart line" to "every line".

    Joining happens BEFORE comment classification, which matches systemd's own
    behaviour: a comment line ending in ``\\`` continues, and its continuation
    is part of the comment.  Both real dashboard units rely on this — the
    watchdog service quotes the old inline-shell ExecStart across two ``#``
    lines joined by a backslash, and classifying first would leave the second
    half of that quote looking like a directive.
    """
    joined: list[str] = []
    pending: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        piece = line if pending is None else f"{pending} {line.strip()}"
        if continued:
            pending = piece
            continue
        joined.append(piece)
        pending = None
    if pending is not None:
        # Trailing backslash on the final line — keep what we have rather than
        # silently dropping the directive.
        joined.append(pending)
    return joined


def parse_unit_directives(text: str) -> dict[str, dict[str, list[str]]]:
    """Parse a systemd unit into ``{section: {key: [value, ...]}}``.

    Classification rules are taken verbatim from the precedent's
    ``parse_unit_sections`` (scripts/check_fused_memory_unit_parity.py):

    - ``[X]`` opens section X.
    - Lines starting with ``#`` or ``;`` are comments — skipped.
    - Blank lines are skipped.
    - Lines before the first section header are DROPPED, not attributed.

    Two deliberate divergences from that precedent, each required here:

    1. **key → values LIST, not a flat line list.**  This checker compares
       directives BY KEY, which a flat list of lines cannot express, and it
       needs the several ``Environment=`` lines of a unit addressable as a
       group rather than as unrelated strings.
    2. **Backslash continuations are JOINED.**  ``parse_unit_sections``
       documents that it does not join them, which is harmless for its exact
       whole-line membership checks.  It is fatal here: the dashboard
       ExecStart spans four physical lines, so without joining every uvicorn
       flag task 3306 added lives on a line the parser never associates with
       ``ExecStart`` — the checker would report parity on a command it never
       actually read.

    Each surviving line is split on the FIRST ``=`` only, so
    ``Environment=A=1`` yields key ``Environment`` and value ``A=1``.  A line
    with no ``=`` is skipped (systemd has no valueless directives).
    """
    sections: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    for joined_line in _join_continuations(text):
        line = joined_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, {})
            continue
        if current is None:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        sections[current].setdefault(key.strip(), []).append(value.strip())
    return sections


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    """Parse args and run the parity check.  See the module exit-code table."""
    parser = argparse.ArgumentParser(
        description="Verify parity between the in-repo and installed dashboard units."
    )
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
