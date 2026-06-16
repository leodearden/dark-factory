"""Parity checker for the installed fused-memory systemd unit.

Verifies that the installed user unit (~/.config/systemd/user/fused-memory.service)
carries all host-invariant safety directives committed in the template
(scripts/fused-memory.service.template). Alarms on drift; optionally fixes it.

Exit codes
----------
0 — parity (all required directives present)
1 — drift   (one or more required directives missing)
2 — installed unit absent (no installed unit found at the given path)

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
- Stdlib-only (pathlib, argparse, subprocess, sys) — runs under plain python3.
- Required directives are an explicit curated allow-list of host-INVARIANT safety
  switches. They are NOT auto-derived from the template, because some template lines
  are host-specific (e.g. Environment=...PREDONE_HOOK...) and would produce false
  drift alarms on other machines.
- --fix only APPENDS missing directives; it never removes or reorders existing lines.
  This preserves intentionally host-specific lines (e.g. extra
  DASHBOARD_KNOWN_PROJECT_ROOTS entries) that live only in the installed unit.
"""

import argparse
import pathlib
import subprocess
import sys
from typing import Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_INSTALLED = pathlib.Path.home() / ".config" / "systemd" / "user" / "fused-memory.service"
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_DEFAULT_TEMPLATE = _SCRIPT_DIR / "fused-memory.service.template"

# Host-invariant safety switches that MUST be present in [Service] as
# non-comment directives.  Extend this list to guard additional safety flags.
REQUIRED_SERVICE_DIRECTIVES: tuple[str, ...] = (
    "Environment=MEM0_TELEMETRY=false",
    "WatchdogSec=120",
)

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
) -> list[str]:
    """Return required directives NOT present as non-comment lines in [Service].

    A directive is considered absent if it does not appear as a verbatim
    non-comment line inside the [Service] section — a commented-out copy or
    a line in another section is treated as missing.

    Args:
        unit_text: The full text of the systemd unit file.
        required:  Ordered tuple of exact directive strings to check.

    Returns:
        Sorted list of missing directives (empty if all are present).
    """
    sections = parse_unit_sections(unit_text)
    service_lines = sections.get("Service", [])
    return [d for d in required if d not in service_lines]


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

    Args:
        unit_text: The original unit file text.
        required:  Directives to ensure are present in [Service].

    Returns:
        Updated text string (or the original if nothing was missing).
    """
    missing = find_drift(unit_text, required)
    if not missing:
        return unit_text

    lines = unit_text.splitlines(keepends=True)
    # Find the insertion index: the line AFTER the last [Service] content line,
    # i.e. just before the next section header or end-of-file.
    in_service = False
    last_service_content_idx = -1
    next_section_idx = len(lines)  # default: append at end

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_service:
                # We were in [Service] and hit the next section
                next_section_idx = i
                break
            if stripped == "[Service]":
                in_service = True
            continue
        if in_service and stripped and not stripped.startswith("#") and not stripped.startswith(";"):
            last_service_content_idx = i

    # Insert the missing directives just before the next section header
    # (or at end-of-file if [Service] is the last section).
    insertion_lines = [d + "\n" for d in missing]
    new_lines = lines[:next_section_idx] + insertion_lines + lines[next_section_idx:]
    return "".join(new_lines)


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
        print(
            f"[warn] systemctl --user daemon-reload failed (exit {exc.returncode}): "
            f"{exc.stderr.decode(errors='replace').strip()}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    """Parse args and run parity check (and optional fix).

    Returns:
        0 — parity
        1 — drift
        2 — installed unit absent
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
        print(
            f"[skip] Installed unit not found at {installed_path} "
            "(unit may not be installed on this host)",
            file=sys.stderr,
        )
        return 2

    unit_text = installed_path.read_text(encoding="utf-8")
    drift = find_drift(unit_text)

    # Also verify that the template itself is not drifted (self-sanity check).
    if template_path.exists():
        template_drift = find_drift(template_path.read_text(encoding="utf-8"))
        if template_drift:
            print(
                f"[warn] Template {template_path} is itself missing: {template_drift}",
                file=sys.stderr,
            )

    if not drift:
        print(f"[ok] {installed_path}: parity — all required directives present.")
        return 0

    print(
        f"[drift] {installed_path}: missing required directives:\n"
        + "".join(f"  - {d}\n" for d in drift)
    )

    if args.fix:
        fixed_text = fix_unit_text(unit_text)
        installed_path.write_text(fixed_text, encoding="utf-8")
        print(f"[fixed] Appended {len(drift)} directive(s) to {installed_path}")
        daemon_reload()
        return 0

    print(
        "Run with --fix to append missing directives without clobbering "
        "host-specific lines.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
