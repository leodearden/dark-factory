"""Config-contract tests for fused-memory/fused-memory.service.example-systemd-config.

Guards the operational guarantee that MEM0_TELEMETRY=false is pinned at the
systemd/process layer — defense-in-depth so the telemetry-off guarantee does not
depend solely on Python import ordering in server/main.py.

Also guards WatchdogSec=120 in both committed unit files (task 1731 palliative):
ensures that even if the dedicated-thread heartbeat is delayed, systemd gives
the process 120s before SIGABRT (was 30s — too tight with the old on-loop design).
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_systemd_unit(path: Path) -> dict[str, list[str]]:
    """Parse a systemd unit file into a dict of section → non-comment lines.

    Rules applied:
    - Lines whose stripped form starts with '[' open a new section.
    - Lines whose stripped form starts with '#' or ';' are comments — skip.
    - Blank lines are skipped.
    - All other lines belong to the current section (None before the first header).

    Limitation: line continuations (trailing '\\') are NOT joined.  Each
    physical line is recorded separately.  This is harmless for exact-string
    membership checks like ``Environment=MEM0_TELEMETRY=false``, but callers
    that need reassembled values for multi-line directives (e.g. ExecStart)
    must handle joining themselves.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
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
# Tests
# ---------------------------------------------------------------------------

class TestSystemdUnitMem0Telemetry:
    """Guard that MEM0_TELEMETRY=false is present as a functional directive in [Service]."""

    # Locate unit file: tests/ → fused-memory/ (parent.parent) → file
    UNIT_FILE = (
        Path(__file__).resolve().parent.parent
        / "fused-memory.service.example-systemd-config"
    )
    EXPECTED_DIRECTIVE = "Environment=MEM0_TELEMETRY=false"

    def test_mem0_telemetry_false_in_service_section(self):
        """[Service] must contain Environment=MEM0_TELEMETRY=false as a non-comment directive.

        systemd only honours Environment= directives under [Service]; a directive
        placed in another section or hidden in a comment is silently ignored.
        A bare substring search anywhere in the file would produce a false-green
        on a misplaced or commented-out line — so we parse section-by-section.

        Background: mem0's Posthog telemetry historically spawned ~8000 threads/day
        and pegged CPU at 1400%. The in-process dotenv suppression in server/main.py
        is contingent on import ordering. Pinning the var at the systemd layer removes
        that fragile dependency (belt-and-suspenders; both mechanisms are kept active).
        The live/deployed unit must carry this identical line.

        Scope note: this test pins the *checked-in example* file
        (fused-memory.service.example-systemd-config), not the live/deployed unit.
        It cannot detect drift between the example and a deployed unit.  That gap
        is accepted as an inherent in-repo limitation; deployment discipline (copying
        the example to the actual unit) remains the responsibility of the operator.
        """
        sections = _parse_systemd_unit(self.UNIT_FILE)
        assert "Service" in sections, (
            f"[Service] section not found in {self.UNIT_FILE}"
        )
        service_directives = sections["Service"]
        assert self.EXPECTED_DIRECTIVE in service_directives, (
            f"'{self.EXPECTED_DIRECTIVE}' not found as a non-comment directive "
            f"inside the [Service] section of {self.UNIT_FILE.name}.\n"
            "Add:\n"
            "    Environment=MEM0_TELEMETRY=false\n"
            "under [Service] (alongside the existing Environment= lines) to pin "
            "mem0 Posthog telemetry OFF at the systemd/process layer. "
            "The live/deployed unit must carry the identical line.\n"
            f"Current [Service] directives: {service_directives}"
        )


# ---------------------------------------------------------------------------
# WatchdogSec contract — task 1731 palliative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit_path,label", [
    (
        Path(__file__).resolve().parent.parent / "fused-memory.service.example-systemd-config",
        "fused-memory.service.example-systemd-config",
    ),
    (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "fused-memory.service.template",
        "scripts/fused-memory.service.template",
    ),
])
class TestWatchdogSecContract:
    """Guard that WatchdogSec=120 is present as a non-comment [Service] directive
    in both committed unit files (task 1731 palliative: 30->120 to give the
    dedicated-thread watchdog heartbeat adequate headroom).

    Section-aware parse prevents a commented/misplaced line from false-greening.
    """

    EXPECTED_DIRECTIVE = "WatchdogSec=120"

    def test_watchdog_sec_120_in_service_section(self, unit_path: Path, label: str) -> None:
        """[Service] must contain WatchdogSec=120 as a non-comment directive.

        WatchdogSec=30 was too tight: the old on-loop coroutine heartbeat was
        silenced when the asyncio loop was busy/blocked >30s, causing systemd
        to SIGABRT the whole service ~6-8x/day (task 1731).  With the dedicated
        OS-thread heartbeat, 120s is belt-and-suspenders headroom rather than
        the primary guard.

        The live/deployed unit must be updated out-of-band by the operator
        (installed-vs-committed split; the worktree cannot commit ~/.config/...).
        """
        sections = _parse_systemd_unit(unit_path)
        assert "Service" in sections, (
            f"[Service] section not found in {label}"
        )
        service_directives = sections["Service"]
        assert self.EXPECTED_DIRECTIVE in service_directives, (
            f"'{self.EXPECTED_DIRECTIVE}' not found as a non-comment directive "
            f"inside the [Service] section of {label}.\n"
            "Change WatchdogSec=30 -> WatchdogSec=120 under [Service] "
            "(task 1731 palliative: gives the dedicated-thread heartbeat "
            "adequate headroom). The live/deployed unit must be updated "
            "out-of-band by the operator.\n"
            f"Current [Service] directives: {service_directives}"
        )
