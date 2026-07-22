"""Tests for scripts/scan_task_toolcall_leaks.py — the READ-ONLY,
detection-only sweep for leaked serialized tool-call XML fragments (e.g.
``</description>\\n<parameter name="priority">low``) in Taskmaster task text
columns.

Task 2939: builds the proactive sweep after the pattern recurred on ~32 live
tasks (first seen as an apparent one-off on task 2080, then recurring on
task 2865). This module — and its tests — never mutate task text; they only
detect and report.

Mirrors test_recon_busy_check.py: pure functions (detect_leak, scan_db,
discover_db_paths, format_report, format_json) get direct pytest coverage;
main() gets subprocess coverage (added alongside the CLI in a later step).

Fixture shapes below are modeled on the real live-DB leak shapes confirmed
while planning this task (tasks 992, 1068, 1067, 2691) and the real
false-positive prose mentions (tasks 2938/2939) — not invented shapes.
"""
from __future__ import annotations

from scan_task_toolcall_leaks import detect_leak

# ---------------------------------------------------------------------------
# Genuine-leak fixtures: a stray closing tag, a REAL newline, then one or more
# serialized <parameter name="..."> fragments running to end-of-string.
# ---------------------------------------------------------------------------

# Shape (a) — task 992: a description leak ending in a bare priority param.
GENUINE_DESCRIPTION_LEAK_FRAGMENT = '</description>\n<parameter name="priority">low'
GENUINE_DESCRIPTION_LEAK = (
    "Add a new test that writes >limit records and asserts only the newest "
    "`limit` are materialised."
) + GENUINE_DESCRIPTION_LEAK_FRAGMENT

# Shape (b) — task 1068: a chained leak where the stray tag is itself
# </parameter> rather than </description>.
CHAINED_PARAMETER_LEAK_FRAGMENT = '</parameter>\n<parameter name="priority">high'
CHAINED_PARAMETER_LEAK = (
    "Root cause is documented in a code comment; a regression test exercises "
    "the orphaned-commit path."
) + CHAINED_PARAMETER_LEAK_FRAGMENT

# Shape (c) — task 1067: the leak lands in `details` (closing tag </details>).
DETAILS_LEAK_FRAGMENT = '</details>\n<parameter name="priority">polish'
DETAILS_LEAK_TEXT = (
    "Verify with `pytest fused-memory/tests/test_bulk_reset_guard.py -q`."
) + DETAILS_LEAK_FRAGMENT

# Shape (d) — task 2691: the "swallowed-details" variant, where an entire
# serialized <parameter name="details">...</parameter> tool-call payload
# leaked into `description` and runs all the way to end-of-string, while the
# real `details` column is left empty.
SWALLOWED_DETAILS_FRAGMENT = (
    '</description>\n<parameter name="details">Run fused-memory/scripts/'
    "audit_duplicate_memories.py or a direct Qdrant admin query against "
    "memory_id=028edb1f-299c-4755-9438-deadbeefcafe (null category, "
    "unretrievable content matching the fingerprint) to confirm-safe the "
    "deletion.\n\nClose out this one-off cleanup once resolved so it stops "
    "recurring in Stage 1/2 payloads every cycle."
)
SWALLOWED_DETAILS_LEAK = (
    "The dashboard shows an orphaned memory row with a null agent_id and "
    "unretrievable content matching the fingerprint."
) + SWALLOWED_DETAILS_FRAGMENT

# ---------------------------------------------------------------------------
# False-positive fixtures: prose that MENTIONS the fragment (quoting the
# ESCAPED literal backslash-n — two real characters, "\" then "n" — not an
# actual newline) and keeps going with trailing prose afterward. Exact
# 2938/2939 shape.
# ---------------------------------------------------------------------------

PROSE_MENTION_FALSE_POSITIVE = (
    "Stage 1 found that task 2865's description had a leaked plain-text "
    'tool-call/XML fragment appended: `</description>\\n<parameter '
    'name="priority">low`. Stage 2 (this run) verified the fragment '
    "verbatim via live get_task and stripped it from the description via "
    "update_task."
)

BARE_PARAMETER_MENTION = (
    'The bug report mentions <parameter name="priority"> appearing in raw '
    "form somewhere upstream, but this description itself is clean."
)

CLEAN_TEXT = "This is a perfectly normal task description with no XML leakage at all."


# ---------------------------------------------------------------------------
# detect_leak(text) -> str | None
# ---------------------------------------------------------------------------

def test_detect_leak_returns_fragment_for_description_leak():
    assert detect_leak(GENUINE_DESCRIPTION_LEAK) == GENUINE_DESCRIPTION_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_chained_parameter_leak():
    assert detect_leak(CHAINED_PARAMETER_LEAK) == CHAINED_PARAMETER_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_details_leak():
    assert detect_leak(DETAILS_LEAK_TEXT) == DETAILS_LEAK_FRAGMENT


def test_detect_leak_returns_fragment_for_swallowed_details_leak():
    assert detect_leak(SWALLOWED_DETAILS_LEAK) == SWALLOWED_DETAILS_FRAGMENT


def test_detect_leak_returns_none_for_prose_mention_false_positive():
    """The exact 2938/2939 shape: escaped literal backslash-n (not a real
    newline) plus trailing prose after the fragment-looking substring."""
    assert detect_leak(PROSE_MENTION_FALSE_POSITIVE) is None


def test_detect_leak_returns_none_for_bare_parameter_mid_prose():
    assert detect_leak(BARE_PARAMETER_MENTION) is None


def test_detect_leak_returns_none_for_clean_text():
    assert detect_leak(CLEAN_TEXT) is None


def test_detect_leak_returns_none_for_empty_string():
    assert detect_leak("") is None


def test_detect_leak_returns_none_for_none():
    assert detect_leak(None) is None


def test_trailing_whitespace_after_fragment_is_tolerated():
    """detect_leak rstrips before matching, so trailing whitespace tacked on
    after the fragment (e.g. a task text with stray trailing blank lines)
    does not defeat detection."""
    text_with_trailing_ws = GENUINE_DESCRIPTION_LEAK + "\n\n   \n"
    assert detect_leak(text_with_trailing_ws) == GENUINE_DESCRIPTION_LEAK_FRAGMENT
