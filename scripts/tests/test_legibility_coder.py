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

import json

import pytest

import codebook as codebook_mod
import coder as mod
import digest as digest_mod

# ---------------------------------------------------------------------------
# Shared fixture helpers — synthetic transcript -> real digest text, mirrors
# test_legibility_digest.py's helper shapes (own copies, per this repo's
# convention of each test file owning its scaffolding, e.g. test_codebook.py's
# _minimal_v2()).
# ---------------------------------------------------------------------------

_SESSION_ID = "cafe1234-0000-4000-8000-000000000000"
_CWD = "/home/leo/src/dark-factory"
_TIMESTAMP = "2026-07-14T06:02:29.796Z"


def _user_text(s, *, cwd=_CWD, session_id=_SESSION_ID, timestamp=_TIMESTAMP):
    """Build a synthetic genuine (non-meta, non-sidechain) human user turn
    carrying real-transcript-shaped top-level cwd/sessionId/timestamp
    fields (observed on every non-queue-operation record in a real Claude
    Code transcript)."""
    return {
        "type": "user",
        "message": {"role": "user", "content": s},
        "isSidechain": False,
        "isMeta": False,
        "cwd": cwd,
        "sessionId": session_id,
        "timestamp": timestamp,
    }


def _write_jsonl(tmp_path, records, name="transcript.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")
    return path


def _build_digest_text(tmp_path, *, session_id=_SESSION_ID, agent_class="interactive", name="transcript.jsonl"):
    """Build a real digest string via digest.build_digest on a minimal
    synthetic single-user-turn transcript — session/date/agent_class in
    the resulting frontmatter are deterministic from the inputs given
    here."""
    records = [_user_text("Please fix this, it is wrong.", session_id=session_id)]
    path = _write_jsonl(tmp_path, records, name=name)
    return digest_mod.build_digest(path, agent_class_override=agent_class)


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


# ---------------------------------------------------------------------------
# step-3: RED — parse_frontmatter() digest frontmatter -> meta dict
# ---------------------------------------------------------------------------

def test_parse_frontmatter_returns_session_date_agent_class(tmp_path):
    digest_text = _build_digest_text(tmp_path, agent_class="orchestrated-task")

    meta = mod.parse_frontmatter(digest_text)

    assert meta["session"] == _SESSION_ID
    assert meta["date"] == "2026-07-14"
    assert meta["agent_class"] == "orchestrated-task"


def test_parse_frontmatter_raises_on_missing_delimiters():
    with pytest.raises(mod.CoderParseError):
        mod.parse_frontmatter("no frontmatter here, just prose")


def test_parse_frontmatter_raises_on_non_mapping_block():
    malformed = "---\n- just\n- a\n- list\n---\nbody\n"
    with pytest.raises(mod.CoderParseError):
        mod.parse_frontmatter(malformed)


# ---------------------------------------------------------------------------
# step-5: RED — build_prompt() embeds digest + codebook index + phase enum
# ---------------------------------------------------------------------------

def test_build_prompt_embeds_digest_and_index_verbatim():
    digest_text = '---\nsession: "s1"\n---\n\n## User Corrections\n- unique marker UC123'
    index = "- entry-a: Title A — cause a\n- entry-b: Title B — cause b"

    prompt = mod.build_prompt(digest_text, index)

    assert digest_text in prompt
    assert index in prompt


def test_build_prompt_embeds_phase_enum_including_unknown():
    prompt = mod.build_prompt("digest text", "codebook index")

    for phase in codebook_mod.PHASES:
        assert phase in prompt
    assert "unknown" in prompt


# ---------------------------------------------------------------------------
# step-7: RED — parse_coder_output() raw LLM stdout -> judgment dict
# ---------------------------------------------------------------------------

def test_parse_coder_output_clean_json_object():
    raw = '{"matches": [], "candidates": []}'
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_json_fenced_block():
    raw = '```json\n{"matches": [], "candidates": []}\n```'
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_json_embedded_in_prose():
    raw = (
        "Sure, here is my judgment:\n"
        '{"matches": [], "candidates": []}\n'
        "Let me know if you need anything else."
    )
    assert mod.parse_coder_output(raw) == {"matches": [], "candidates": []}


def test_parse_coder_output_pure_garbage_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output("I cannot help with that request, sorry.")


def test_parse_coder_output_top_level_array_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output('[{"entry_id": "x"}]')


def test_parse_coder_output_top_level_scalar_raises():
    with pytest.raises(mod.CoderParseError):
        mod.parse_coder_output('"just a string"')
