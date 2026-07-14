"""Tests for scripts/legibility/coder.py — the Haiku trickle coder (digest
-> strict-JSON §7.3 coding record).

See plans/confusion-reduction-prd.md §7.3 (coding record contract), §8.1
(consumer-side boundary test), §8.6 (coder failure storm). Task delta of
the confusion-reduction PRD decomposition.

Import the target module as `import coder as mod`, resolved via
scripts/tests/conftest.py's sys.path insertion (both scripts/ and
scripts/legibility/ are on sys.path; no package __init__ needed) — mirrors
test_legibility_digest.py's import style.

The LLM is ALWAYS mocked in this file: every code_digest/code_digests call
below injects a fake `invoke` callable (or monkeypatches mod._invoke_cli)
rather than ever shelling out to a real `claude` process.
"""
from __future__ import annotations

import coder as mod


# ---------------------------------------------------------------------------
# step-1: RED — build_codebook_index() compact rendering
# ---------------------------------------------------------------------------

def test_build_codebook_index_compact_one_line_per_entry():
    codebook = {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Short title A",
                "cause": (
                    "Paragraph one explaining the root cause in some "
                    "detail.\n\nParagraph two continues on and on with "
                    "even more detail than anyone could possibly need "
                    "here, well past the one-line summary cap so "
                    "truncation must kick in eventually for sure yes "
                    "indeed this needs to be quite a bit longer still "
                    "to actually cross the two hundred character cap."
                ),
                "fix": "apply patch XYZ",
                "fix_where": ["some/file.py:12"],
                "status": "open",
                "severity": "high",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [
                    {
                        "date": "2026-01-01", "project": "p", "session": "s",
                        "origin_phase": "implement", "manifested_phase": "merge",
                    }
                ],
            },
            {
                "id": "entry-b",
                "title": "Title B",
                "cause": "one line cause for B",
                "status": "retired",
                "severity": "low",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
        "candidates": [
            {
                "id": "cand-20260101-1",
                "title": "SECRET_CANDIDATE_TITLE_MARKER",
                "cause": "a novel candidate cause, never in the entry index",
                "first_seen": "2026-01-01",
                "disposition": "pending",
                "sightings": [],
            }
        ],
    }

    index = mod.build_codebook_index(codebook)
    lines = [line for line in index.splitlines() if line.strip()]

    assert len(lines) == 2, f"expected exactly one line per entry, got: {lines!r}"
    assert "entry-a" in lines[0] and "Short title A" in lines[0]
    assert "entry-b" in lines[1] and "Title B" in lines[1]

    # cause collapsed to a single line -- no embedded newlines anywhere.
    assert "\n" not in lines[0]
    assert "\n" not in lines[1]

    # retired entries are still included (a census re-observes pre-fix
    # traces, so a live sighting can still match a retired cause).
    assert "entry-b" in index

    # heavy fields (and candidates) never leak into the compact index.
    for banned in (
        "fix_where", "apply patch XYZ", "some/file.py", "sightings",
        "2026-01-01", "SECRET_CANDIDATE_TITLE_MARKER",
    ):
        assert banned not in index, f"heavy/candidate field leaked into index: {banned!r}"


def test_build_codebook_index_empty_entries_yields_empty_or_header_only():
    codebook = {"version": 2, "entries": [], "candidates": []}
    index = mod.build_codebook_index(codebook)
    assert index.strip() == ""
