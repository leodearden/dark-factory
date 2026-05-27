"""Config-contract tests for fused-memory/fused-memory.service.example-systemd-config.

Guards the operational guarantee that MEM0_TELEMETRY=false is pinned at the
systemd/process layer — defense-in-depth so the telemetry-off guarantee does not
depend solely on Python import ordering in server/main.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_systemd_unit(path: Path) -> dict[str, list[str]]:
    """Parse a systemd unit file into a dict of section → non-comment lines.

    Rules (faithful to systemd INI dialect):
    - Lines whose stripped form starts with '[' open a new section.
    - Lines whose stripped form starts with '#' or ';' are comments — skip.
    - Blank lines are skipped.
    - All other lines belong to the current section (None before the first header).
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

    def test_unit_file_exists(self):
        """Sanity-check that the unit file itself is present in the repo."""
        assert self.UNIT_FILE.is_file(), (
            f"Systemd unit example not found at {self.UNIT_FILE}. "
            "Was the file renamed or moved?"
        )

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
