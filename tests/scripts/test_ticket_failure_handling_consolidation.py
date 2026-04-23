"""Structural integrity tests for the ticket-failure-handling consolidation.

These tests read the source-controlled skill files directly — no runtime
dependencies.  They guard against:
1.  The canonical file disappearing or losing a reason name.
2.  Any of the four caller files silently dropping the link to the canonical doc.

Pattern: same as tests/scripts/test_dashboard_service_template.py — plain
``str in text`` / ``re.search`` assertions, no prose pinning.

See also:
  - skills/_shared/ticket-failure-handling.md  — canonical reference
  - skills/review/references/phase3-triage.md  — caller: review
  - skills/unblock/SKILL.md                    — caller: unblock
  - skills/escalation-watcher/SKILL.md         — caller: escalation-watcher
  - skills/orchestrate/SKILL.md                — caller: orchestrate
"""

import pathlib

REPO_ROOT = pathlib.Path(__file__).parents[2]

CANONICAL = REPO_ROOT / "skills" / "_shared" / "ticket-failure-handling.md"

# All five canonical reason names that must appear in the canonical doc.
REASON_NAMES = [
    "server_restart",
    "timeout",
    "unknown_ticket",
    "server_closed",
    "expired",
]

# R4 gate must be referenced by artefact name.
R4_MARKERS = ["_check_escalation_idempotency", "R4"]

# Relative path fragment that each caller file must contain as a link to the canonical doc.
CANONICAL_LINK_FRAGMENT = "_shared/ticket-failure-handling.md"

# The four caller files.
PHASE3_TRIAGE = REPO_ROOT / "skills" / "review" / "references" / "phase3-triage.md"
UNBLOCK_SKILL = REPO_ROOT / "skills" / "unblock" / "SKILL.md"
ESCALATION_WATCHER_SKILL = REPO_ROOT / "skills" / "escalation-watcher" / "SKILL.md"
ORCHESTRATE_SKILL = REPO_ROOT / "skills" / "orchestrate" / "SKILL.md"


def test_canonical_file_exists_and_names_all_reasons() -> None:
    """skills/_shared/ticket-failure-handling.md must exist and mention all five reason names plus R4.

    This catches:
    - The file being deleted or moved.
    - A reason name being renamed/dropped without updating the canonical reference.
    - The R4 idempotency concept being stripped from the doc.
    """
    assert CANONICAL.exists(), (
        f"Canonical ticket-failure-handling doc not found at {CANONICAL}.\n"
        "Create skills/_shared/ticket-failure-handling.md with the full reason matrix."
    )

    text = CANONICAL.read_text(encoding="utf-8")

    for reason in REASON_NAMES:
        assert reason in text, (
            f"Reason name {reason!r} not found in {CANONICAL}.\n"
            "The canonical doc must mention every reason name so reviewers notice "
            "if one disappears."
        )

    assert any(marker in text for marker in R4_MARKERS), (
        f"Neither {R4_MARKERS} found in {CANONICAL}.\n"
        "The canonical doc must reference the R4 idempotency gate by artefact name "
        "(_check_escalation_idempotency or 'R4')."
    )


def test_phase3_triage_links_to_canonical() -> None:
    """skills/review/references/phase3-triage.md must link to the canonical ticket-failure doc.

    Catches a future editor removing the link during an unrelated edit, which
    would silently allow duplication to creep back.
    """
    text = PHASE3_TRIAGE.read_text(encoding="utf-8")
    assert CANONICAL_LINK_FRAGMENT in text, (
        f"{CANONICAL_LINK_FRAGMENT!r} not found in {PHASE3_TRIAGE}.\n"
        "Add a link to skills/_shared/ticket-failure-handling.md for the full reason matrix."
    )


def test_unblock_links_to_canonical() -> None:
    """skills/unblock/SKILL.md must link to the canonical ticket-failure doc."""
    text = UNBLOCK_SKILL.read_text(encoding="utf-8")
    assert CANONICAL_LINK_FRAGMENT in text, (
        f"{CANONICAL_LINK_FRAGMENT!r} not found in {UNBLOCK_SKILL}.\n"
        "Add a link to skills/_shared/ticket-failure-handling.md for the full reason matrix."
    )


def test_escalation_watcher_links_to_canonical() -> None:
    """skills/escalation-watcher/SKILL.md must link to the canonical ticket-failure doc."""
    text = ESCALATION_WATCHER_SKILL.read_text(encoding="utf-8")
    assert CANONICAL_LINK_FRAGMENT in text, (
        f"{CANONICAL_LINK_FRAGMENT!r} not found in {ESCALATION_WATCHER_SKILL}.\n"
        "Add a link to skills/_shared/ticket-failure-handling.md for the full reason matrix."
    )


def test_orchestrate_links_to_canonical() -> None:
    """skills/orchestrate/SKILL.md must link to the canonical ticket-failure doc."""
    text = ORCHESTRATE_SKILL.read_text(encoding="utf-8")
    assert CANONICAL_LINK_FRAGMENT in text, (
        f"{CANONICAL_LINK_FRAGMENT!r} not found in {ORCHESTRATE_SKILL}.\n"
        "Add a link to skills/_shared/ticket-failure-handling.md for the full reason matrix."
    )
