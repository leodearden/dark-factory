"""Tests for scripts/legibility/codebook.py — confusion-codebook v2 schema,
validator, and deterministic sole-writer merger (task 2575).

See plans/confusion-reduction-prd.md §7.1 (codebook v2), §7.3 (coding
record), §8.2/§8.3 (idempotency + never-delete boundary tests).

Imported as a namespace package (`from legibility import codebook as mod`)
since scripts/legibility/ is a subdir of scripts/ (on sys.path via
scripts/tests/conftest.py) with no __init__.py — confirmed empirically to
resolve under pytest's --import-mode=importlib.
"""
from __future__ import annotations

import copy

import pytest

from legibility import codebook as mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_v2() -> dict:
    """A minimal well-formed v2 codebook: one entry, no candidates."""
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": "high",
                "status": "open",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [],
            }
        ],
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# step-1: RED — validate() structural + semantic checks
# ---------------------------------------------------------------------------

def test_validate_minimal_v2_codebook_is_valid():
    assert mod.validate(_minimal_v2()) == []


def test_validate_rejects_wrong_version():
    codebook = _minimal_v2()
    codebook["version"] = 1
    assert mod.validate(codebook) != []


def test_validate_rejects_out_of_enum_origin_phase():
    codebook = _minimal_v2()
    codebook["entries"][0]["origin_phase"] = "not-a-phase"
    assert mod.validate(codebook) != []


def test_validate_rejects_out_of_enum_manifested_phase():
    codebook = _minimal_v2()
    codebook["entries"][0]["manifested_phase"] = "not-a-phase"
    assert mod.validate(codebook) != []


def test_validate_rejects_bad_status():
    codebook = _minimal_v2()
    codebook["entries"][0]["status"] = "yes"  # v1 value, not in v2 enum
    assert mod.validate(codebook) != []


def test_validate_rejects_duplicate_entry_id():
    codebook = _minimal_v2()
    dup = copy.deepcopy(codebook["entries"][0])
    codebook["entries"].append(dup)
    assert mod.validate(codebook) != []


def test_validate_rejects_malformed_candidate_id():
    codebook = _minimal_v2()
    codebook["candidates"].append(
        {
            "id": "not-a-valid-id",
            "title": "novel shape",
            "first_seen": "2026-07-14",
            "disposition": "pending",
            "sightings": [],
        }
    )
    assert mod.validate(codebook) != []


def test_validate_rejects_bad_candidate_disposition():
    codebook = _minimal_v2()
    codebook["candidates"].append(
        {
            "id": "cand-20260714-1",
            "title": "novel shape",
            "first_seen": "2026-07-14",
            "disposition": "maybe",
            "sightings": [],
        }
    )
    assert mod.validate(codebook) != []


def test_validate_permissive_on_v1_free_form_fields():
    """A v1-style entry with filed_tasks as a bare string and no area/cause
    still validates — the validator is strict only on v2-relevant fields."""
    codebook = _minimal_v2()
    codebook["entries"][0]["filed_tasks"] = "2547, 2548"
    # deliberately no area/cause/fix/fix_where/known_cause_match
    assert mod.validate(codebook) == []


# ---------------------------------------------------------------------------
# step-3: RED — load()/dump() round-trip + deterministic serialization
# ---------------------------------------------------------------------------

def test_dump_load_roundtrip(tmp_path):
    codebook = _minimal_v2()
    path = tmp_path / "codebook.yaml"
    mod.dump(codebook, path)
    assert mod.load(path) == codebook


def test_dump_is_byte_stable_across_calls(tmp_path):
    """Dumping the same dict twice yields byte-identical file contents —
    a no-change night must produce zero diff (PRD §6.7)."""
    codebook = _minimal_v2()
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    mod.dump(codebook, path_a)
    mod.dump(codebook, path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_dump_starts_with_canonical_header(tmp_path):
    codebook = _minimal_v2()
    path = tmp_path / "codebook.yaml"
    mod.dump(codebook, path)
    assert path.read_text(encoding="utf-8").startswith(mod.HEADER)


def test_dump_uses_block_style_not_flow_style(tmp_path):
    """The emitted file must use block style for entries — no inline
    `{...}` flow mappings."""
    codebook = _minimal_v2()
    codebook["entries"][0]["sightings"].append(
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-1",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    )
    path = tmp_path / "codebook.yaml"
    mod.dump(codebook, path)
    text = path.read_text(encoding="utf-8")
    assert "{" not in text
    assert "}" not in text
