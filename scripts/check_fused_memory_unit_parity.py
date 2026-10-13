"""Parity checker for the installed fused-memory systemd unit.

Verifies that the installed user unit (~/.config/systemd/user/fused-memory.service)
carries all host-invariant safety directives committed in the template
(scripts/fused-memory.service.template). Alarms on drift; optionally fixes it.

Exit codes
----------
The family's codes are defined once, in
``systemd_unit_parity.py::ParityExit``. Here, drift means a required directive
is missing from the installed unit.

An exit status alone is not enough to read this script's verdict: 2 is also
what `python3` returns for a script it cannot open and what argparse returns
for a rejected flag. So EVERY line this script emits carries the
``[fused_memory_unit_parity]`` tag, and setup-host.sh's parity gate believes a
status only when that tag is present in the captured output. The tag's ABSENCE
is therefore conclusive rather than heuristic, which only holds because every
PHYSICAL line carries it — see ``systemd_unit_parity.emit``.

Usage
-----
  # verify only
  python3 scripts/check_fused_memory_unit_parity.py

  # verify with explicit paths
  python3 scripts/check_fused_memory_unit_parity.py \\
      --installed ~/.config/systemd/user/fused-memory.service \\
      --template  scripts/fused-memory.service.template

  # verify and fix in place (appends missing directives, reloads systemd)
  python3 scripts/check_fused_memory_unit_parity.py --fix

Design notes
------------
- Stdlib-only (pathlib, argparse, subprocess, sys) plus the sibling
  scripts/systemd_unit_parity.py, which is itself stdlib-only — so this still
  runs under a plain python3 with no environment set up.
- Required directives are an explicit curated allow-list of host-INVARIANT safety
  switches. They are NOT auto-derived from the template, because some template lines
  are host-specific (e.g. Environment=...PREDONE_HOOK...) and would produce false
  drift alarms on other machines. Their Environment= part is
  ``systemd_unit_parity.py::ENV_POLICY``'s required projection.
- --fix only APPENDS missing directives; it never removes or reorders existing lines.
  This preserves intentionally host-specific lines (e.g. extra
  DASHBOARD_KNOWN_PROJECT_ROOTS entries) that live only in the installed unit.
- All output goes through the family's one tagger, ``systemd_unit_parity.emit``
  (via ``_log``, or ``emit_block`` for the ``[override]`` block), so the report
  is uniformly tagged; the contract is pinned by
  test_main_every_emitted_line_carries_the_log_tag in
  tests/scripts/test_check_fused_memory_unit_parity.py.
- DROP-INS are consulted even when the unit file is at parity, via the shared
  ``systemd_unit_parity.find_dropins``. `systemctl --user edit` never modifies
  the unit file; it writes ``<unit>.d/override.conf`` beside it, which systemd
  merges OVER the unit at load time — so the whole-line membership comparison
  above can certify every required directive present while the configuration
  that actually runs is a different one. This closed the LAST drop-in-blind
  member of the check_*_unit_parity.py family (task-3775 lineage; defect
  writeups 4382/4388); the dashboard, orchestrator and lms checkers already
  consulted it. Reported, never removed: a drop-in can be load-bearing (task
  3750), so removal has owners with preconditions
  (scripts/remove-lms-arm-worktree-dropin.sh is the precedent) that a
  general-purpose parity checker has no business re-implementing.
"""

import argparse
import pathlib
import subprocess
import sys
from collections.abc import Sequence

# A bare module-name import, the same mechanism the three sibling checkers use
# — do not add a path shim. It resolves in BOTH contexts this script runs in:
# under the CLI python puts the executed script's own directory (``scripts/``)
# at ``sys.path[0]``, and under pytest ``tests/scripts/conftest.py`` inserts
# ``scripts/`` explicitly (load-bearing, because pyproject sets
# ``--import-mode=importlib``, under which pytest deliberately does not perform
# that sys.path mutation itself).
from systemd_unit_parity import (
    ENV_POLICY,
    SERVICE_SECTION,
    ParityExit,
    UnitFamily,
    emit,
    emit_block,
    find_dropins,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, matching
# [dashboard_unit_parity], [orchestrator_unit_parity] and [lms_unit_parity].
# setup-host.sh routes operators to a detailed report BY TAG rather than by
# position, so an untagged line in a long bring-up run has no reliable way to
# point at its own output — and the gate's "no tag, so it did not run" test
# would be answerable only by heuristic.
LOG_TAG = "fused_memory_unit_parity"

_DEFAULT_INSTALLED = pathlib.Path.home() / ".config" / "systemd" / "user" / "fused-memory.service"
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_TEMPLATE = _SCRIPT_DIR / "fused-memory.service.template"

# Host-invariant safety switches that MUST be present in [Service] as
# non-comment directives. The Environment= entries are ENV_POLICY's required
# projection; a host-local name cannot be one, because EnvPolicy raises at import
# (B5, plans/review-all-s25-scripts-instruments-prd.md D5).
#
# Restart=on-failure / RestartSec=5 / RestartSteps=4 / TimeoutStartSec=300 /
# TimeoutStopSec=90 are host-invariant literal strings (already present
# verbatim in scripts/fused-memory.service.template) — exact membership
# matching both detects a divergent value (e.g. a wrong TimeoutStopSec) and
# lets --fix append the correct line.
#
# RestartSteps=4 is host-invariant for the same reason the others are, and is
# listed here rather than left to the template alone because the template is
# not what runs: the unit above RestartSteps in the template declares
# RestartMaxDelaySec=60, and systemd DISCARDS that cap on any unit that does
# not also declare RestartSteps= (it logs "Service has RestartMaxDelaySec= but
# no RestartSteps= setting. Ignoring." at load and moves on). An installed unit
# missing this line therefore has no growing backoff at all, silently, and the
# only way that gets corrected on the host is for this checker to report it.
REQUIRED_SERVICE_DIRECTIVES: tuple[str, ...] = (
    *ENV_POLICY.required_directives(UnitFamily.FUSED_MEMORY),
    "WatchdogSec=120",
    "Restart=on-failure",
    "RestartSec=5",
    "RestartSteps=4",
    "TimeoutStartSec=300",
    "TimeoutStopSec=90",
)

# Directive PREFIXES that MUST be present (as the start of some non-comment
# [Service] line) but whose full value is host-specific and therefore cannot
# be exact-matched or synthesized by --fix. ExecStartPre= carries
# host-specific paths (__REPO_ROOT__, /home/leo/bin) — only its presence can
# be asserted. ExecStopPost= is the hook that stops fused-memory.socket on a
# deliberate stop; without it a stopped service is restarted by the next
# connection to its still-listening port.
REQUIRED_SERVICE_DIRECTIVE_PREFIXES: tuple[str, ...] = ("ExecStartPre=", "ExecStopPost=")


def _log(message: str, *, stream=None) -> None:
    """Tag every physical line of *message*; the contract lives in systemd_unit_parity.py::emit."""
    emit(LOG_TAG, message, out=stream if stream is not None else sys.stdout)


# ---------------------------------------------------------------------------
# Unit parser
# ---------------------------------------------------------------------------


def parse_unit_sections(text: str) -> dict[str, list[str]]:
    """Parse a systemd unit file text into a dict of section → non-comment lines.

    Rules applied (mirrors fused-memory/tests/test_systemd_unit_config.py::_parse_systemd_unit):
    - Lines whose stripped form starts with '[' and ends with ']' open a new section.
    - Lines whose stripped form starts with '#' or ';' are comments — skipped.
    - Blank lines are skipped.
    - All other lines belong to the current section (None before the first header).

    Limitation: line continuations (trailing '\\') are NOT joined.  Each
    physical line is recorded separately.  This is harmless for exact-string
    membership checks (e.g. ``Environment=MEM0_TELEMETRY=false``).
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def find_drift(
    unit_text: str,
    required: tuple[str, ...] = REQUIRED_SERVICE_DIRECTIVES,
    required_prefixes: tuple[str, ...] = REQUIRED_SERVICE_DIRECTIVE_PREFIXES,
) -> list[str]:
    """Return required directives NOT present as non-comment lines in [Service].

    A directive is considered absent if it does not appear as a verbatim
    non-comment line inside the [Service] section — a commented-out copy or
    a line in another section is treated as missing.

    A prefix in ``required_prefixes`` is considered absent if no non-comment
    [Service] line starts with it — used for directives whose full value is
    host-specific (e.g. ExecStartPre=) and therefore cannot be exact-matched.
    Prefix misses are appended, in order, after the exact-match misses.

    Args:
        unit_text: The full text of the systemd unit file.
        required:  Ordered tuple of exact directive strings to check.
        required_prefixes: Ordered tuple of directive prefixes whose mere
            presence (not exact value) must be checked. Pass ``()`` to skip
            prefix checking entirely (e.g. from fix_unit_text, which can only
            ever synthesize exact directives).

    Returns:
        List of missing directives/prefixes: exact misses (in ``required``
        order) followed by prefix misses (in ``required_prefixes`` order).
        Empty if everything required is present.
    """
    sections = parse_unit_sections(unit_text)
    service_lines = sections.get(SERVICE_SECTION, [])
    missing = [d for d in required if d not in service_lines]
    missing.extend(
        p for p in required_prefixes if not any(line.startswith(p) for line in service_lines)
    )
    return missing


# ---------------------------------------------------------------------------
# Fix
# ---------------------------------------------------------------------------


def fix_unit_text(
    unit_text: str,
    required: tuple[str, ...] = REQUIRED_SERVICE_DIRECTIVES,
) -> str:
    """Return a new unit text with missing required directives appended to [Service].

    Behaviour:
    - Computes find_drift; if empty returns unit_text unchanged (idempotent).
    - Appends each missing directive immediately after the last non-blank
      line of the [Service] section, before the next section header.
    - Never removes or reorders any existing line.
    - Only ever synthesizes EXACT directives (required_prefixes=() is passed
      to find_drift) — a prefix-checked directive like ExecStartPre= carries
      a host-specific value that cannot be guessed, so a missing prefix is
      surfaced by find_drift() for reporting but never appended here.

    Args:
        unit_text: The original unit file text.
        required:  Directives to ensure are present in [Service].

    Returns:
        Updated text string (or the original if nothing was missing).
    """
    missing = find_drift(unit_text, required, required_prefixes=())
    if not missing:
        return unit_text

    lines = unit_text.splitlines(keepends=True)
    # Find the insertion index: just before the next section header after [Service],
    # or at end-of-file if [Service] is the last section.
    in_service = False
    next_section_idx = len(lines)  # default: append at end

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_service:
                # Hit the section after [Service]
                next_section_idx = i
                break
            if stripped == f"[{SERVICE_SECTION}]":
                in_service = True

    # Ensure the line just before the insertion point ends with '\n' so the
    # appended directives are not concatenated onto it (edge case: [Service]
    # is the last section and the file lacks a trailing newline).
    if lines and next_section_idx > 0 and not lines[next_section_idx - 1].endswith("\n"):
        lines[next_section_idx - 1] += "\n"

    insertion_lines = [d + "\n" for d in missing]
    new_lines = lines[:next_section_idx] + insertion_lines + lines[next_section_idx:]
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# Drop-in overrides
# ---------------------------------------------------------------------------


_OVERRIDE_TAIL: tuple[str, ...] = (
    "systemd merges these over the unit at load time, so a directive set here "
    "silently wins over the committed value. Inspect the merged result with: "
    "systemctl --user cat fused-memory.service",
    "Nothing was removed: this checker is read-only about drop-ins BY DESIGN, "
    "because a drop-in can be load-bearing (task 3750). --fix appends to the "
    "unit FILE and can neither synthesize nor resolve an override living in a "
    "different one — remove it by hand, or move the setting into the committed "
    "template.",
)


def _report_override(
    installed_path: pathlib.Path, dropins: list[pathlib.Path]
) -> None:
    """Name every drop-in layered over the unit: its EFFECTIVE configuration went unverified."""
    emit_block(
        LOG_TAG,
        "override",
        header=(
            f"{len(dropins)} drop-in(s) apply to {installed_path} — the unit file "
            "was compared, but the EFFECTIVE configuration was NOT verified by "
            "that comparison:"
        ),
        entries=[f"  {dropin}" for dropin in dropins],
        tail=_OVERRIDE_TAIL,
        out=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Systemd reload
# ---------------------------------------------------------------------------


def daemon_reload() -> None:
    """Run `systemctl --user daemon-reload` (best-effort; tolerant when absent)."""
    try:
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        # systemctl not available (e.g. CI without systemd)
        pass
    except subprocess.CalledProcessError as exc:
        _log(
            f"[warn] systemctl --user daemon-reload failed (exit {exc.returncode}): "
            f"{exc.stderr.decode(errors='replace').strip()}",
            stream=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    """Parse args and run parity check (and optional fix); return a ParityExit.

    systemd_unit_parity.py::ParityExit is the one statement of what each code
    means — a drop-in alone is FINDING, never PARITY.
    """
    parser = argparse.ArgumentParser(
        description="Verify/fix parity of installed fused-memory systemd unit."
    )
    parser.add_argument(
        "--installed",
        type=pathlib.Path,
        default=_DEFAULT_INSTALLED,
        help="Path to the installed unit (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        type=pathlib.Path,
        default=_DEFAULT_TEMPLATE,
        help="Path to the source-of-truth template (default: %(default)s)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Append missing required directives to the installed unit and daemon-reload.",
    )
    args = parser.parse_args(argv)

    installed_path: pathlib.Path = args.installed
    template_path: pathlib.Path = args.template

    # Exit code 2: installed unit absent
    if not installed_path.exists():
        _log(
            f"[skip] Installed unit not found at {installed_path} "
            "(unit may not be installed on this host)",
            stream=sys.stderr,
        )
        return ParityExit.NOT_INSTALLED

    unit_text = installed_path.read_text(encoding="utf-8")
    drift = find_drift(unit_text)

    # Consulted EVEN WHEN the unit file itself is at parity: a drop-in is
    # layered OVER a matching unit file, so it is invisible to the whole-line
    # membership comparison above. `--installed` names a FILE here (the three
    # sibling checkers take an installed-DIR), so the drop-in directory is
    # derived from it rather than passed in.
    dropins = find_dropins(installed_path.parent, installed_path.name)
    if dropins:
        _report_override(installed_path, dropins)

    # Also verify that the template itself is not drifted (self-sanity check).
    if template_path.exists():
        template_drift = find_drift(template_path.read_text(encoding="utf-8"))
        if template_drift:
            _log(
                f"[warn] Template {template_path} is itself missing: {template_drift}",
                stream=sys.stderr,
            )

    if not drift:
        if dropins:
            # NOT parity, and deliberately not exit 0: the required directives
            # all matched, but what this run could not establish is that the
            # EFFECTIVE configuration matches. Reporting `[ok] ... parity` here
            # would be a green claim covering exactly the configuration that is
            # not running.
            return ParityExit.FINDING
        _log(f"[ok] {installed_path}: parity — all required directives present.")
        return ParityExit.PARITY

    # rstrip: the joined list ends in a newline, which under a per-line _log
    # would render one final line holding nothing but the tag.
    _log(
        (
            f"[drift] {installed_path}: missing required directives:\n"
            + "".join(f"  - {d}\n" for d in drift)
        ).rstrip("\n")
    )

    if args.fix:
        # fix_unit_text only ever synthesizes EXACT directives — a host-specific
        # prefix directive (e.g. ExecStartPre=, whose value carries __REPO_ROOT__
        # / host paths) cannot be guessed. So the count of lines actually
        # appended is the number of missing EXACT directives, NOT len(drift)
        # (which also counts un-synthesizable prefix misses).
        appended = find_drift(unit_text, required_prefixes=())
        fixed_text = fix_unit_text(unit_text)
        installed_path.write_text(fixed_text, encoding="utf-8")
        _log(f"[fixed] Appended {len(appended)} directive(s) to {installed_path}")
        daemon_reload()

        # Re-check the written text with the default (prefix-aware) config. Any
        # residual drift is an un-synthesizable prefix directive that --fix
        # could not resolve; report it and exit 1 (drift) rather than falsely
        # signalling parity with exit 0 — a follow-up plain verify would exit 1.
        residual = find_drift(fixed_text)
        if residual:
            _log(
                (
                    f"[drift] {installed_path}: --fix cannot synthesize host-specific "
                    f"directive(s) (value carries host paths — add them by hand):\n"
                    + "".join(f"  - {d}\n" for d in residual)
                ).rstrip("\n"),
                stream=sys.stderr,
            )
            return ParityExit.FINDING

        # An override is neither synthesizable nor removable HERE, so a --fix
        # run that repaired everything it could must still decline to report
        # success. The repair was real and is kept; what it cannot establish is
        # that the values it just wrote are the ones that would take effect,
        # because the drop-in lives in a DIFFERENT FILE and systemd merges it
        # over them at load time. Reporting 0 would be the checker
        # manufacturing the reassurance.
        #
        # Reported and NOT removed, following the lms precedent: a drop-in can
        # be load-bearing (task 3750), so removal has a correct owner with
        # preconditions — scripts/remove-lms-arm-worktree-dropin.sh — that a
        # general-purpose parity checker has no business re-implementing.
        # The [override] block was already emitted above and is worded APART
        # from the residual-drift report just above: they share exit 1 but send
        # the operator to different places (hand-add a host-specific directive
        # vs. inspect and remove a drop-in).
        if dropins:
            return ParityExit.FINDING
        return ParityExit.PARITY

    _log(
        "Run with --fix to append missing directives without clobbering "
        "host-specific lines.",
        stream=sys.stderr,
    )
    return ParityExit.FINDING


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
