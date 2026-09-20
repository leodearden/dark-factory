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
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from legibility import codebook as mod
from legibility import coder

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


# ---------------------------------------------------------------------------
# amendment: dump() writes atomically (temp file + os.replace) instead of
# truncating the destination in place, so a crash/kill mid-write can never
# leave a partial/corrupt registry behind for the next load()/validate().
# ---------------------------------------------------------------------------

def test_dump_leaves_no_temp_file_behind_on_success(tmp_path):
    path = tmp_path / "codebook.yaml"
    mod.dump(_minimal_v2(), path)
    assert list(tmp_path.iterdir()) == [path]


def test_dump_preserves_original_and_cleans_up_temp_file_on_failure(tmp_path, monkeypatch):
    """If the atomic replace step fails, the destination file must be left
    exactly as it was (never truncated/partially written), and the temp
    file used to stage the new content must not be left behind."""
    path = tmp_path / "codebook.yaml"
    mod.dump(_minimal_v2(), path)
    original_bytes = path.read_bytes()

    def _boom(*_args, **_kwargs):
        raise OSError("simulated os.replace failure")

    monkeypatch.setattr(mod.os, "replace", _boom)

    changed = _minimal_v2()
    changed["entries"][0]["title"] = "a different title"
    with pytest.raises(OSError):
        mod.dump(changed, path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]  # no stray temp file


# ---------------------------------------------------------------------------
# step-5: RED — migrate_v1_to_v2()
# ---------------------------------------------------------------------------

def _v1_fixture() -> dict:
    return {
        "version": 1,
        "updated": "2026-07-13",
        "entries": [
            {
                "id": "one-shot-subagent-contract",
                "title": "Full entry",
                "severity": "high",
                "area": "orchestrator-prompt",
                "cause": "...",
                "status": "partially",
                "sightings_2026_06": 17,
                "affected": ["a", "b"],
                "fix": "...",
                "fix_where": ["x.py:1"],
                "fix_effort": "S",
                "known_cause_match": "...",
                "filed_tasks": "2547, 2548",
            },
            {
                "id": "recon-prompt-schema-drift",
                "title": "Yes-status entry",
                "severity": "medium",
                "status": "yes",
                "sightings_2026_06": 16,
                "filed_tasks": [2559],
            },
            {
                # Minimal oneoff shape: only id/title/severity/status/sightings_2026_06
                "id": "oneoff-2026-07-01",
                "title": "A one-off",
                "severity": "low",
                "status": "mined-unverified",
                "sightings_2026_06": 1,
            },
        ],
    }


def test_migrate_v1_to_v2_sets_version_and_defaults():
    v1 = _v1_fixture()
    result = mod.migrate_v1_to_v2(v1)
    assert result["version"] == 2
    for entry in result["entries"]:
        assert entry["origin_phase"] == "unknown"
        assert entry["manifested_phase"] == "unknown"
        assert entry["sightings"] == []
    assert result["candidates"] == []


def test_migrate_v1_to_v2_maps_yes_status_to_open():
    v1 = _v1_fixture()
    result = mod.migrate_v1_to_v2(v1)
    by_id = {e["id"]: e for e in result["entries"]}
    assert by_id["recon-prompt-schema-drift"]["status"] == "open"
    # unchanged statuses stay as-is
    assert by_id["one-shot-subagent-contract"]["status"] == "partially"
    assert by_id["oneoff-2026-07-01"]["status"] == "mined-unverified"


def test_migrate_v1_to_v2_retains_all_v1_fields_and_order():
    v1 = _v1_fixture()
    result = mod.migrate_v1_to_v2(v1)
    assert [e["id"] for e in result["entries"]] == [e["id"] for e in v1["entries"]]
    full_entry = next(e for e in result["entries"] if e["id"] == "one-shot-subagent-contract")
    assert full_entry["sightings_2026_06"] == 17
    assert full_entry["filed_tasks"] == "2547, 2548"
    assert full_entry["affected"] == ["a", "b"]
    assert full_entry["fix_where"] == ["x.py:1"]
    oneoff = next(e for e in result["entries"] if e["id"] == "oneoff-2026-07-01")
    assert oneoff["sightings_2026_06"] == 1
    assert "area" not in oneoff
    assert "cause" not in oneoff


def test_migrate_v1_to_v2_output_validates_green():
    v1 = _v1_fixture()
    result = mod.migrate_v1_to_v2(v1)
    assert mod.validate(result) == []


def test_migrate_v1_to_v2_is_idempotent():
    v1 = _v1_fixture()
    once = mod.migrate_v1_to_v2(v1)
    twice = mod.migrate_v1_to_v2(once)
    assert once == twice


def test_migrate_v1_to_v2_does_not_mutate_input():
    v1 = _v1_fixture()
    original = copy.deepcopy(v1)
    mod.migrate_v1_to_v2(v1)
    assert v1 == original


def test_migrate_v1_to_v2_maps_yaml_boolean_true_status_to_open():
    """PyYAML's default (YAML-1.1) resolver coerces an unquoted `status: yes`
    to the Python bool True, not the string "yes" — confirmed empirically
    against the real docs/legibility/confusion-codebook.yaml (both of its
    `status: yes` entries load as status=True). migrate_v1_to_v2 must treat
    the loaded bool True the same as the string "yes" and map it to 'open'."""
    v1 = {
        "version": 1,
        "entries": [
            {
                "id": "bool-coerced-yes",
                "title": "A v1 entry whose status was unquoted `yes` in YAML",
                "severity": "medium",
                "status": True,
                "sightings_2026_06": 3,
            }
        ],
    }
    result = mod.migrate_v1_to_v2(v1)
    assert result["entries"][0]["status"] == "open"
    assert mod.validate(result) == []


def test_migrate_v1_to_v2_drops_stale_updated_field():
    """Amendment: under the sole-writer design, dump()/apply_coding_record()
    never stamp the top-level `updated` field, so retaining it would freeze
    at the migration date and mislead readers as sightings accumulate.
    migrate_v1_to_v2 drops it rather than retaining or re-stamping it (a
    mutable timestamp would break dump()'s byte-stable no-change-night
    guarantee). Idempotent: already-absent on a second migration pass."""
    v1 = _v1_fixture()
    assert "updated" in v1  # fixture carries the v1 top-level field

    once = mod.migrate_v1_to_v2(v1)
    assert "updated" not in once
    assert mod.validate(once) == []

    twice = mod.migrate_v1_to_v2(once)
    assert "updated" not in twice
    assert once == twice


# ---------------------------------------------------------------------------
# step-7: RED — validate_coding_record() against §7.3 schema
# ---------------------------------------------------------------------------

def _well_formed_record() -> dict:
    return {
        "session": "sess-1",
        "date": "2026-07-14",
        "project": "dark_factory",
        "agent_class": "orchestrated-task",
        "matches": [
            {
                "entry_id": "entry-a",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "invariant_violated": None,
                "note": "matched on X",
            }
        ],
        "candidates": [
            {
                "title": "novel shape",
                "cause": "...",
                "area": "...",
                "origin_phase": "architect",
                "manifested_phase": "verify",
                "evidence_quote": "...",
            }
        ],
    }


def test_validate_coding_record_well_formed_is_valid():
    assert mod.validate_coding_record(_well_formed_record()) == []


def test_validate_coding_record_missing_session():
    record = _well_formed_record()
    del record["session"]
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_missing_date():
    record = _well_formed_record()
    del record["date"]
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_match_out_of_enum_origin_phase():
    record = _well_formed_record()
    record["matches"][0]["origin_phase"] = "not-a-phase"
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_match_missing_entry_id():
    record = _well_formed_record()
    del record["matches"][0]["entry_id"]
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_candidate_missing_title():
    record = _well_formed_record()
    del record["candidates"][0]["title"]
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_matches_not_a_list():
    record = _well_formed_record()
    record["matches"] = "not-a-list"
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_candidates_not_a_list():
    record = _well_formed_record()
    record["candidates"] = "not-a-list"
    assert mod.validate_coding_record(record) != []


def test_validate_coding_record_invariant_violated_accepts_string():
    record = _well_formed_record()
    record["matches"][0]["invariant_violated"] = "corroborate-before-acting"
    assert mod.validate_coding_record(record) == []


def test_validate_coding_record_invariant_violated_accepts_null():
    record = _well_formed_record()
    record["matches"][0]["invariant_violated"] = None
    assert mod.validate_coding_record(record) == []


# ---------------------------------------------------------------------------
# step-9: RED (§8.2 idempotency, match path) — apply_coding_record()
# ---------------------------------------------------------------------------

def _codebook_with_entry_a() -> dict:
    codebook = _minimal_v2()
    codebook["entries"][0]["id"] = "entry-a"
    return codebook


def _match_record(entry_id="entry-a", session="sess-1", date="2026-07-14"):
    return {
        "session": session,
        "date": date,
        "project": "dark_factory",
        "agent_class": "orchestrated-task",
        "matches": [
            {
                "entry_id": entry_id,
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "invariant_violated": None,
            }
        ],
    }


def test_apply_coding_record_appends_one_sighting():
    codebook = _codebook_with_entry_a()
    record = _match_record()

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["sightings"] == [
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-1",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    ]
    assert mod.validate(result) == []


def test_apply_coding_record_match_is_idempotent_on_session_and_entry():
    codebook = _codebook_with_entry_a()
    record = _match_record()

    once, _ = mod.apply_coding_record(codebook, record)
    twice, _ = mod.apply_coding_record(once, record)

    entry = next(e for e in twice["entries"] if e["id"] == "entry-a")
    assert len(entry["sightings"]) == 1
    assert mod.validate(twice) == []


def test_apply_coding_record_unknown_entry_id_is_skipped_and_counted():
    codebook = _codebook_with_entry_a()
    record = _match_record(entry_id="entry-zzz")

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["entries"]) == 1  # no entry fabricated
    assert result["entries"][0]["sightings"] == []
    assert stats["skipped_unknown_entry"] == 1
    assert mod.validate(result) == []


def test_apply_coding_record_does_not_mutate_input():
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _match_record()

    mod.apply_coding_record(codebook, record)

    assert codebook == original


# ---------------------------------------------------------------------------
# step-11: RED (§8.3 candidate append) — apply_coding_record() candidates
# ---------------------------------------------------------------------------

def _candidate_record(title="novel shape", session="sess-1", date="2026-07-14"):
    return {
        "session": session,
        "date": date,
        "project": "dark_factory",
        "agent_class": "orchestrated-task",
        "candidates": [
            {
                "title": title,
                "cause": "...",
                "area": "...",
                "origin_phase": "architect",
                "manifested_phase": "verify",
                "evidence_quote": "...",
            }
        ],
    }


def test_apply_coding_record_appends_one_candidate():
    codebook = _codebook_with_entry_a()
    record = _candidate_record()

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["id"] == "cand-20260714-1"
    assert candidate["first_seen"] == "2026-07-14"
    assert candidate["disposition"] == "pending"
    assert candidate["title"] == "novel shape"
    assert len(candidate["sightings"]) == 1
    assert candidate["sightings"][0]["session"] == "sess-1"
    assert stats["candidates_applied"] == 1
    assert mod.validate(result) == []


def test_apply_coding_record_candidate_is_idempotent_on_session_and_title():
    codebook = _codebook_with_entry_a()
    record = _candidate_record()

    once, _ = mod.apply_coding_record(codebook, record)
    twice, _ = mod.apply_coding_record(once, record)

    assert len(twice["candidates"]) == 1
    assert len(twice["candidates"][0]["sightings"]) == 1
    assert mod.validate(twice) == []


def test_apply_coding_record_different_candidate_same_day_increments_id():
    codebook = _codebook_with_entry_a()
    record_a = _candidate_record(title="novel shape")
    once, _ = mod.apply_coding_record(codebook, record_a)

    record_b = _candidate_record(title="a different shape")
    twice, _ = mod.apply_coding_record(once, record_b)

    assert len(twice["candidates"]) == 2
    ids = {c["title"]: c["id"] for c in twice["candidates"]}
    assert ids["novel shape"] == "cand-20260714-1"
    assert ids["a different shape"] == "cand-20260714-2"
    assert mod.validate(twice) == []


# ---------------------------------------------------------------------------
# task-4144 step-1: RED — the merger must never resurrect an adjudicated
# candidate by fabricating a byte-identical-title pending twin. This is the
# defect that turned rejected `cand-20260722-28` back into pending
# `cand-20260724-2` in the live registry: `same_title` spans every
# disposition, but `pending_match` filters to `pending` only, so a title
# whose records are all rejected/promoted fell through to the create branch.
# ---------------------------------------------------------------------------

def _codebook_with_adjudicated_candidate(disposition: str) -> dict:
    """A v2 codebook carrying exactly one already-adjudicated candidate —
    the shape `census.reject_candidate`/`promote_candidate` leave behind."""
    codebook = _codebook_with_entry_a()
    codebook["candidates"] = [
        {
            "id": "cand-20260722-28",
            "title": "novel shape",
            "cause": "...",
            "area": "...",
            "first_seen": "2026-07-22",
            "disposition": disposition,
            "sightings": [
                {
                    "date": "2026-07-22",
                    "project": "dark_factory",
                    "session": "sess-old",
                    "origin_phase": "architect",
                    "manifested_phase": "verify",
                }
            ],
        }
    ]
    return codebook


def test_apply_coding_record_does_not_resurrect_rejected_candidate():
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["candidates"]) == 1  # NO pending twin fabricated
    candidate = result["candidates"][0]
    assert candidate["id"] == "cand-20260722-28"
    assert candidate["disposition"] == "rejected"  # census verdict untouched
    # recurrence signal preserved, append-only
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert stats["candidate_disposition_conflicts"] == 1
    assert stats["candidates_applied"] == 0
    assert mod.validate(result) == []


def test_apply_coding_record_does_not_duplicate_promoted_candidate():
    """The FALLBACK half of the promoted case: this fixture's candidate is
    stamped `promoted` but carries no `promoted_to` (a hand-edited or
    pre-`promote_candidate` record), so there is no entry to route the
    recurrence to — it lands on the candidate and counts as a conflict,
    exactly like the rejected case."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["id"] == "cand-20260722-28"
    assert candidate["disposition"] == "promoted"  # census verdict untouched
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert stats["candidate_disposition_conflicts"] == 1
    assert stats["candidates_applied"] == 0
    assert mod.validate(result) == []


def test_apply_coding_record_routes_promoted_recurrence_to_its_entry():
    """A promoted candidate's `sightings` list is a DEAD field for new
    signal: `census.promote_candidate` deep-copies it into the new entry
    once, at promotion time, and nothing re-reads it afterwards (the matrix
    path and the codebook index both read ENTRY sightings). So a recurrence
    of a promoted title must land on the entry named by `promoted_to` and
    count as a `matched` sighting — filing it on the candidate would report
    the signal as preserved while writing it where no consumer looks."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-a"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-new"]
    assert stats["matched"] == 1
    assert stats["candidate_disposition_conflicts"] == 0
    assert stats["candidates_applied"] == 0

    # The candidate itself is untouched — verdict AND sightings; no twin.
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["disposition"] == "promoted"
    assert len(candidate["sightings"]) == 1
    assert mod.validate(result) == []


def test_apply_coding_record_promoted_recurrence_dedupes_on_the_entry():
    """Session-level dedup on the ENTRY, mirroring the match path: a record
    that both matches the entry directly and re-mines the promoted title
    appends exactly one sighting, not two."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-a"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")
    record["matches"] = [
        {"entry_id": "entry-a", "origin_phase": "implement", "manifested_phase": "merge"}
    ]

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-new"]
    assert stats["matched"] == 1  # the match path's append; the candidate re-sighting deduped
    assert stats["candidate_disposition_conflicts"] == 0
    assert stats["candidates_applied"] == 0
    assert len(result["candidates"][0]["sightings"]) == 1
    assert mod.validate(result) == []


def test_apply_coding_record_promoted_with_dangling_promoted_to_falls_back():
    """`promoted_to` naming an entry that does not exist must NOT silently
    drop the recurrence: fall back to the candidate append + conflict
    counter (the merger never fabricates an entry — only the census does)."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-that-was-never-created"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["entries"]) == 1  # no entry fabricated
    assert result["entries"][0]["sightings"] == []
    candidate = result["candidates"][0]
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert stats["matched"] == 0
    assert stats["candidate_disposition_conflicts"] == 1
    assert mod.validate(result) == []


def test_apply_coding_record_adjudicated_conflict_is_idempotent():
    """The pre-existing `already_seen` guard already spans every
    disposition, so re-applying the same record appends nothing and reports
    no fresh conflict."""
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    once, first_stats = mod.apply_coding_record(codebook, record)
    twice, second_stats = mod.apply_coding_record(once, record)

    assert len(twice["candidates"]) == 1
    assert len(twice["candidates"][0]["sightings"]) == 2  # unchanged by the 2nd apply
    assert first_stats["candidate_disposition_conflicts"] == 1
    assert second_stats["candidate_disposition_conflicts"] == 0
    assert mod.validate(twice) == []


def test_apply_coding_record_still_creates_pending_for_a_genuinely_new_title():
    """Regression guard: the new conflict branch must not swallow genuinely
    novel titles — only an ALREADY-SEEN title takes it."""
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(
        title="a different shape", session="sess-new", date="2026-07-24"
    )

    result, stats = mod.apply_coding_record(codebook, record)

    assert len(result["candidates"]) == 2
    fresh = next(c for c in result["candidates"] if c["title"] == "a different shape")
    assert fresh["disposition"] == "pending"
    assert fresh["id"] == "cand-20260724-1"
    assert stats["candidates_applied"] == 1
    assert stats["candidate_disposition_conflicts"] == 0
    assert mod.validate(result) == []


# ---------------------------------------------------------------------------
# task 5198: RED — apply_coding_record() `corrections`, the third op.
#
# A sighting is an immutable dated observation, so both existing ops are pure
# appends. An entry's title/cause is instead the CURRENT best explanation of a
# cause, and it is the only part of an entry that
# `scripts/legibility/coder.py::build_codebook_index` renders to the coder —
# so a refuted framing can only be withdrawn by rewriting it there.
# ---------------------------------------------------------------------------

# apply_coding_record()'s full stats contract with every counter at rest: what
# the sole writer reports when it changed nothing. Shared with the live-corpus
# re-apply no-op pins below so the contract is declared once.
_NO_CHANGES = {
    "matched": 0,
    "skipped_unknown_entry": 0,
    "candidates_applied": 0,
    "candidate_disposition_conflicts": 0,
    "corrections_applied": 0,
    "correction_skipped": 0,
    "record_invalid": False,
}


def _correction_record(entry_id="entry-a", session="sess-correction", date="2026-07-14"):
    """A §7.3 record carrying one `corrections` op that names all three
    writable fields. Callers mutate the returned record to build variants."""
    return {
        "session": session,
        "date": date,
        "project": "dark_factory",
        "agent_class": "orchestrated-task",
        "corrections": [
            {
                "entry_id": entry_id,
                "origin_phase": "unknown",
                "manifested_phase": "verify",
                "title": "Retracted: the recorded cause was refuted",
                "cause": "The premise did not hold; superseded by entry-b.",
                "status": "retired",
                "note": "CORRECTION: the transcript refutes the recorded cause.",
            }
        ],
    }


def test_apply_coding_record_correction_rewrites_only_the_named_fields():
    """Writes title/cause/status, leaves every other entry field alone, and
    appends exactly one sighting built from the record header + the
    correction's own phase stamps and note."""
    codebook = _codebook_with_entry_a()
    before = copy.deepcopy(codebook["entries"][0])
    record = _correction_record()
    correction = record["corrections"][0]

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["title"] == correction["title"]
    assert entry["cause"] == correction["cause"]
    assert entry["status"] == correction["status"]

    untouched = {
        field: value
        for field, value in before.items()
        if field not in {"title", "cause", "status", "sightings"}
    }
    assert {field: entry[field] for field in untouched} == untouched

    assert entry["sightings"] == [
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-correction",
            "origin_phase": "unknown",
            "manifested_phase": "verify",
            "note": correction["note"],
        }
    ]
    assert stats == {**_NO_CHANGES, "corrections_applied": 1}
    assert mod.validate(result) == []


def test_apply_coding_record_correction_never_deletes_and_does_not_mutate_input():
    """Prior sightings survive a field rewrite, the new one lands after them,
    and the input codebook is not touched (deep-copy semantics, as both
    existing ops guarantee)."""
    codebook = _codebook_with_entry_a()
    codebook["entries"][0]["sightings"] = [
        {
            "date": "2026-07-01",
            "project": "dark_factory",
            "session": "sess-original",
            "origin_phase": "unknown",
            "manifested_phase": "verify",
        }
    ]
    original = copy.deepcopy(codebook)

    result, _ = mod.apply_coding_record(codebook, _correction_record())

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == [
        "sess-original",
        "sess-correction",
    ]
    mod.assert_no_deletion(original, result)
    assert codebook == original


def test_apply_coding_record_correction_re_apply_is_a_no_op(caplog):
    """The whole correction — field writes AND provenance sighting — is gated
    by the one `session` dedup the match path already uses, so `apply` re-run
    over the same file (nightly, or to resolve a rebase) changes nothing."""
    codebook = _codebook_with_entry_a()
    record = _correction_record()

    with caplog.at_level(logging.WARNING, logger="legibility.codebook"):
        once, _ = mod.apply_coding_record(codebook, record)
        twice, stats = mod.apply_coding_record(once, record)

    assert stats == _NO_CHANGES
    assert twice == once
    # SILENT by design, and that is the whole point of the distinction: an
    # already-merged record is the dedup doing its job. `correction_skipped`
    # and its WARNING are reserved for the sibling-op collision below, where
    # a correction genuinely failed to land.
    assert caplog.records == []


def _match_and_correction_record(entry_id="entry-a"):
    """One record that both matches and corrects the same entry — the shape in
    which the per-(session, entry) sighting dedup forces a choice between the
    two ops."""
    record = _correction_record(entry_id=entry_id)
    record["matches"] = [
        {
            "entry_id": entry_id,
            "origin_phase": "implement",
            "manifested_phase": "merge",
            "note": "the match's own note",
        }
    ]
    return record


def test_apply_coding_record_correction_outranks_a_sibling_match_on_one_entry():
    """Both ops write at most one sighting per (session, entry), so a record
    carrying both for one entry can land only one — and it must be the
    correction: a match's sighting is one interchangeable observation, while
    the correction is the only op that can withdraw a refuted framing."""
    codebook = _codebook_with_entry_a()
    record = _match_and_correction_record()
    correction = record["corrections"][0]

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["title"] == correction["title"]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert stats == {**_NO_CHANGES, "corrections_applied": 1}
    assert mod.validate(result) == []


def test_apply_coding_record_colliding_correction_is_counted_and_logged(caplog):
    """Two corrections for one entry in one record collide on that same single
    slot. The loser cannot land — but it must not be dropped SILENTLY, since a
    vanished framing withdrawal is indistinguishable from a successful one in
    the returned codebook."""
    codebook = _codebook_with_entry_a()
    record = _correction_record()
    loser = copy.deepcopy(record["corrections"][0])
    loser["title"] = "A second, conflicting retitle"
    record["corrections"].append(loser)

    with caplog.at_level(logging.WARNING, logger="legibility.codebook"):
        result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["title"] == record["corrections"][0]["title"]
    assert stats == {**_NO_CHANGES, "corrections_applied": 1, "correction_skipped": 1}

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "entry-a" in message and record["session"] in message
    assert mod.validate(result) == []


# The never-clear guard's input grid. ("status", "empty") is absent by
# construction: "" is not in STATUSES, so an emptied status never reaches the
# truthiness guard — `validate_coding_record` rejects the whole record first,
# which `_correction_with_an_empty_status` pins below.
_UNSUPPLIED_FIELD_SPELLINGS = [
    ("title", "empty"),
    ("title", "omitted"),
    ("cause", "empty"),
    ("cause", "omitted"),
    ("status", "omitted"),
]


@pytest.mark.parametrize(("field", "spelling"), _UNSUPPLIED_FIELD_SPELLINGS)
def test_apply_coding_record_correction_never_clears_a_field(field, spelling):
    """The never-delete boundary of the one op allowed to overwrite instead of
    append: a field the correction does not actually supply — spelled `""` or
    left out — keeps its pre-correction value, while the fields it does supply
    are still written and the provenance sighting is still appended."""
    codebook = _codebook_with_entry_a()
    # entry-a carries no `cause` by default; a correction that omits one can
    # only be shown to preserve it if there was something to preserve.
    codebook["entries"][0]["cause"] = "The original, not-yet-refuted explanation."
    before = copy.deepcopy(codebook["entries"][0])

    record = _correction_record()
    correction = record["corrections"][0]
    if spelling == "empty":
        correction[field] = ""
    else:
        del correction[field]

    result, stats = mod.apply_coding_record(codebook, record)

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry[field] == before[field]
    for still_supplied in {"title", "cause", "status"} - {field}:
        assert entry[still_supplied] == correction[still_supplied]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert stats == {**_NO_CHANGES, "corrections_applied": 1}
    assert mod.validate(result) == []


def test_apply_coding_record_correction_unknown_entry_id_is_skipped_and_counted():
    """Mirrors the match path: only the census creates entries, so an
    unresolvable entry_id is counted, never fabricated."""
    codebook = _codebook_with_entry_a()
    record = _correction_record(entry_id="entry-zzz")

    result, stats = mod.apply_coding_record(codebook, record)

    assert result["entries"] == _codebook_with_entry_a()["entries"]
    assert stats == {**_NO_CHANGES, "skipped_unknown_entry": 1}
    assert mod.validate(result) == []


@pytest.mark.parametrize("action", ["delete", "remove"])
def test_apply_coding_record_raises_on_correction_removal_action(action):
    """A removal-shaped correction is rejected before the codebook is copied,
    exactly as `_reject_deletion_directive` already treats `matches`."""
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _correction_record()
    record["corrections"][0]["action"] = action

    with pytest.raises(mod.NeverDeleteError):
        mod.apply_coding_record(codebook, record)

    assert codebook == original


def _correction_with_out_of_enum_status():
    record = _correction_record()
    record["corrections"][0]["status"] = "not-a-status"
    return record


def _correction_without_its_required_note():
    record = _correction_record()
    del record["corrections"][0]["note"]
    return record


def _correction_with_an_empty_note():
    """`""` is a missing audit trail spelled differently: `_build_sighting`
    emits `note` only when truthy, so an empty one would rewrite the entry's
    framing and leave a bare sighting that says nothing about why."""
    record = _correction_record()
    record["corrections"][0]["note"] = ""
    return record


def _correction_with_an_empty_status():
    """Unlike title/cause, an emptied `status` is not a silently-ignored
    "field not supplied": "" is outside STATUSES, so the record is rejected
    before the never-clear guard ever sees it."""
    record = _correction_record()
    record["corrections"][0]["status"] = ""
    return record


@pytest.mark.parametrize(
    "build_record",
    [
        _correction_with_out_of_enum_status,
        _correction_without_its_required_note,
        _correction_with_an_empty_note,
        _correction_with_an_empty_status,
    ],
)
def test_apply_coding_record_invalid_correction_is_skipped_whole(build_record):
    """An entry's framing may never change without an audit trail, and never
    into a status outside STATUSES. Every such failure is caught by
    `validate_coding_record`, which skips the record WHOLE — no partial field
    write survives."""
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = build_record()

    assert mod.validate_coding_record(record) != []

    result, stats = mod.apply_coding_record(codebook, record)

    assert stats == {**_NO_CHANGES, "record_invalid": True}
    assert result == original


# ---------------------------------------------------------------------------
# step-13: RED (§8.3 never-delete) — NeverDeleteError + assert_no_deletion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["delete", "remove", "retract"])
def test_apply_coding_record_raises_on_top_level_removal_directive(key):
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _match_record()
    record[key] = ["entry-a"]

    with pytest.raises(mod.NeverDeleteError):
        mod.apply_coding_record(codebook, record)

    assert codebook == original  # input untouched


def test_apply_coding_record_raises_on_match_delete_action():
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _match_record()
    record["matches"][0]["action"] = "delete"

    with pytest.raises(mod.NeverDeleteError):
        mod.apply_coding_record(codebook, record)

    assert codebook == original


def test_assert_no_deletion_raises_when_entry_missing():
    before = _codebook_with_entry_a()
    after = copy.deepcopy(before)
    after["entries"] = []

    with pytest.raises(mod.NeverDeleteError):
        mod.assert_no_deletion(before, after)


def test_assert_no_deletion_raises_when_sighting_shrinks():
    before = _codebook_with_entry_a()
    before["entries"][0]["sightings"] = [
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-1",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    ]
    after = copy.deepcopy(before)
    after["entries"][0]["sightings"] = []

    with pytest.raises(mod.NeverDeleteError):
        mod.assert_no_deletion(before, after)


def test_assert_no_deletion_allows_growth():
    before = _codebook_with_entry_a()
    before["entries"][0]["sightings"] = [
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-1",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    ]
    after = copy.deepcopy(before)
    after["entries"][0]["sightings"].append(
        {
            "date": "2026-07-15",
            "project": "dark_factory",
            "session": "sess-2",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    )

    mod.assert_no_deletion(before, after)  # must not raise


def test_assert_no_deletion_allows_retirement_via_status_change():
    before = _codebook_with_entry_a()
    after = copy.deepcopy(before)
    after["entries"][0]["status"] = "retired"

    mod.assert_no_deletion(before, after)  # must not raise

    assert after["entries"][0]["status"] == "retired"
    assert len(after["entries"]) == 1
    assert mod.validate(after) == []


# ---------------------------------------------------------------------------
# amendment: assert_no_deletion() must cover top-level candidates
# symmetrically with entries — candidates hold real mined signal (pending
# novel causes awaiting promotion), so a merge regression that drops or
# shrinks them must trip the same construction-independent safety net.
# ---------------------------------------------------------------------------

def _codebook_with_candidate() -> dict:
    codebook = _codebook_with_entry_a()
    codebook["candidates"] = [
        {
            "id": "cand-20260714-1",
            "title": "novel shape",
            "first_seen": "2026-07-14",
            "disposition": "pending",
            "sightings": [
                {
                    "date": "2026-07-14",
                    "project": "dark_factory",
                    "session": "sess-1",
                    "origin_phase": "architect",
                    "manifested_phase": "verify",
                }
            ],
        }
    ]
    return codebook


def test_assert_no_deletion_raises_when_candidate_missing():
    before = _codebook_with_candidate()
    after = copy.deepcopy(before)
    after["candidates"] = []

    with pytest.raises(mod.NeverDeleteError):
        mod.assert_no_deletion(before, after)


def test_assert_no_deletion_raises_when_candidate_sighting_shrinks():
    before = _codebook_with_candidate()
    after = copy.deepcopy(before)
    after["candidates"][0]["sightings"] = []

    with pytest.raises(mod.NeverDeleteError):
        mod.assert_no_deletion(before, after)


def test_assert_no_deletion_allows_candidate_growth():
    before = _codebook_with_candidate()
    after = copy.deepcopy(before)
    after["candidates"][0]["sightings"].append(
        {
            "date": "2026-07-15",
            "project": "dark_factory",
            "session": "sess-2",
            "origin_phase": "architect",
            "manifested_phase": "verify",
        }
    )

    mod.assert_no_deletion(before, after)  # must not raise


def test_assert_no_deletion_allows_candidate_promotion_via_disposition_change():
    before = _codebook_with_candidate()
    after = copy.deepcopy(before)
    after["candidates"][0]["disposition"] = "promoted"

    mod.assert_no_deletion(before, after)  # must not raise

    assert len(after["candidates"]) == 1
    assert mod.validate(after) == []


# ---------------------------------------------------------------------------
# amendment: migrate_v1_to_v2()/apply_coding_record() fail loudly with a
# clear ValueError on a non-dict codebook (e.g. an empty or malformed YAML
# file loads via yaml.safe_load as None), instead of an AttributeError deep
# in the mutation logic — mirrors validate()'s isinstance guard.
# ---------------------------------------------------------------------------

# Both functions are DECLARED `codebook: dict`, and these two tests
# deliberately violate that to exercise the runtime isinstance guard — the
# point being that an empty codebook file loads via yaml.safe_load as None,
# so this is a real production input, not a synthetic one. The Any-typed
# local states "statically unconstrained on purpose" without suppressing the
# diagnostic, and without widening the production signature (which would
# weaken it for every legitimate caller).
_NOT_A_CODEBOOK: Any = None


def test_migrate_v1_to_v2_raises_on_non_dict_codebook():
    with pytest.raises(ValueError):
        mod.migrate_v1_to_v2(_NOT_A_CODEBOOK)


def test_apply_coding_record_raises_on_non_dict_codebook():
    with pytest.raises(ValueError):
        mod.apply_coding_record(_NOT_A_CODEBOOK, _match_record())


# ---------------------------------------------------------------------------
# step-15: RED — main(argv) CLI end-to-end (tmp_path/capsys)
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_validate_good_file_returns_zero(self, tmp_path):
        path = tmp_path / "codebook.yaml"
        mod.dump(_minimal_v2(), path)

        ret = mod.main(["validate", str(path)])

        assert ret == 0

    def test_validate_broken_file_returns_nonzero_and_prints_errors(self, tmp_path, capsys):
        codebook = _minimal_v2()
        codebook["version"] = 1  # broken: v2 validator requires version == 2
        path = tmp_path / "codebook.yaml"
        mod.dump(codebook, path)

        ret = mod.main(["validate", str(path)])
        captured = capsys.readouterr()

        assert ret != 0
        assert captured.err.strip() != ""

    def test_migrate_rewrites_v1_file_to_v2_in_place(self, tmp_path):
        v1 = _v1_fixture()
        path = tmp_path / "codebook.yaml"
        mod.dump(v1, path)

        ret = mod.main(["migrate", str(path)])

        assert ret == 0
        reloaded = mod.load(path)
        assert reloaded["version"] == 2
        assert reloaded["candidates"] == []
        assert all("sightings" in e for e in reloaded["entries"])
        assert mod.validate(reloaded) == []

    def test_migrate_is_idempotent(self, tmp_path):
        v1 = _v1_fixture()
        path = tmp_path / "codebook.yaml"
        mod.dump(v1, path)

        first = mod.main(["migrate", str(path)])
        second = mod.main(["migrate", str(path)])

        assert first == 0
        assert second == 0
        assert mod.validate(mod.load(path)) == []

    def _write_apply_fixtures(self, tmp_path):
        codebook = _codebook_with_entry_a()
        codebook_path = tmp_path / "codebook.yaml"
        mod.dump(codebook, codebook_path)

        record = _match_record()
        record["candidates"] = [
            {
                "title": "novel shape",
                "cause": "...",
                "area": "...",
                "origin_phase": "architect",
                "manifested_phase": "verify",
                "evidence_quote": "...",
            }
        ]
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return codebook_path, records_path

    def test_apply_applies_match_and_candidate_and_rewrites_file(self, tmp_path):
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)

        ret = mod.main(["apply", str(codebook_path), str(records_path)])

        assert ret == 0
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert len(entry["sightings"]) == 1
        assert len(reloaded["candidates"]) == 1
        assert mod.validate(reloaded) == []

    def test_apply_is_idempotent_at_file_level(self, tmp_path):
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)

        first = mod.main(["apply", str(codebook_path), str(records_path)])
        second = mod.main(["apply", str(codebook_path), str(records_path)])

        assert first == 0
        assert second == 0
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert len(entry["sightings"]) == 1
        assert len(reloaded["candidates"]) == 1
        assert mod.validate(reloaded) == []

    def test_apply_skips_malformed_json_line_without_aborting_batch(self, tmp_path, capsys):
        """A single malformed JSONL line must not raise/abort the whole
        batch (amendment: json.loads() used to be unguarded, so one bad
        emitted record lost the entire nightly/trickle merge instead of
        just itself — mirrors apply_coding_record()'s skip-whole-record
        semantics, but at the line-parsing boundary)."""
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)
        good_line = records_path.read_text(encoding="utf-8")
        records_path.write_text("{not valid json\n" + good_line, encoding="utf-8")

        ret = mod.main(["apply", str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        assert "malformed JSON" in captured.err
        assert "malformed_json=1" in captured.out
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert len(entry["sightings"]) == 1  # the valid line still applied
        assert len(reloaded["candidates"]) == 1
        assert mod.validate(reloaded) == []

    def test_apply_skips_deletion_directive_record_without_aborting_batch(
        self, tmp_path, capsys
    ):
        """A single deletion-shaped record must be skipped and counted like
        a malformed JSON line — not abort the whole batch. dump() is AFTER
        the per-line loop, so a NeverDeleteError escaping _cmd_apply
        discarded every already-applied in-memory record along with the
        records that came after the bad line."""
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)
        good = json.loads(records_path.read_text(encoding="utf-8"))

        bad = _match_record()
        bad["matches"][0]["action"] = "delete"  # trips _reject_deletion_directive

        # A second good record under a DIFFERENT session, so it is not
        # deduped away by the session-keyed idempotency guard — its arrival
        # proves the records AFTER the bad line were not discarded.
        good_2 = copy.deepcopy(good)
        good_2["session"] = "sess-2"

        records_path.write_text(
            "\n".join(json.dumps(r) for r in (good, bad, good_2)) + "\n",
            encoding="utf-8",
        )

        ret = mod.main(["apply", str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        assert str(records_path) in captured.err
        assert ":2:" in captured.err  # the offending line number
        assert "deletion directive" in captured.err
        assert "deletion_directive=1" in captured.out

        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert {s["session"] for s in entry["sightings"]} == {"sess-1", "sess-2"}
        assert mod.validate(reloaded) == []

    def test_apply_summary_reports_candidate_disposition_conflicts(self, tmp_path, capsys):
        """A recurrence sighting appended to an already-adjudicated candidate
        must be REPORTED, not hidden. Secondary to the nightly fix: _cmd_apply
        dumps unconditionally after its loop, so the sighting was always
        persisted here — only the printed summary was blind, leaving the new
        stat vestigial at this merge site."""
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)

        cb = mod.load(codebook_path)
        cb["candidates"] = [
            {
                "id": "cand-20260722-28",
                "title": "recurring rejected cause",
                "first_seen": "2026-07-22",
                "disposition": "rejected",
                "sightings": [
                    {
                        "date": "2026-07-22",
                        "project": "dark_factory",
                        "session": "sess-old",
                        "origin_phase": "architect",
                        "manifested_phase": "verify",
                    }
                ],
            }
        ]
        mod.dump(cb, codebook_path)

        # session must be absent from the seeded sightings, or `already_seen`
        # short-circuits the conflict branch.
        records_path.write_text(
            json.dumps(
                _candidate_record(
                    title="recurring rejected cause", session="sess-new", date="2026-07-24"
                )
            )
            + "\n",
            encoding="utf-8",
        )

        ret = mod.main(["apply", str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        assert "candidate_disposition_conflicts=1" in captured.out
        # Reported SEPARATELY rather than conflated with applied candidates.
        assert "candidates_applied=0" in captured.out

        reloaded = mod.load(codebook_path)
        assert len(reloaded["candidates"]) == 1  # no fabricated pending twin
        candidate = reloaded["candidates"][0]
        assert candidate["disposition"] == "rejected"
        assert len(candidate["sightings"]) == 2
        assert candidate["sightings"][1]["session"] == "sess-new"
        assert mod.validate(reloaded) == []

    def test_apply_summary_reports_corrections_applied(self, tmp_path, capsys):
        """The third op reports itself in the same stdout summary as the other
        two, so an operator (and the nightly log) can see a correction landed
        rather than inferring it from a silent file rewrite."""
        codebook_path = tmp_path / "codebook.yaml"
        mod.dump(_codebook_with_entry_a(), codebook_path)
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(
            json.dumps(_correction_record()) + "\n", encoding="utf-8"
        )

        ret = mod.main(["apply", str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        assert "corrections_applied=1" in captured.out

        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert entry["status"] == "retired"
        assert entry["sightings"][0]["session"] == "sess-correction"
        assert mod.validate(reloaded) == []

    def test_migrate_empty_file_fails_loudly_instead_of_crashing(self, tmp_path, capsys):
        """An empty codebook file loads via yaml.safe_load() as None.
        _cmd_migrate must report a clear error and return 1, not raise an
        unhandled AttributeError/TypeError from inside migrate_v1_to_v2."""
        path = tmp_path / "codebook.yaml"
        path.write_text("", encoding="utf-8")

        ret = mod.main(["migrate", str(path)])
        captured = capsys.readouterr()

        assert ret == 1
        assert captured.err.strip() != ""

    def test_apply_empty_codebook_file_fails_loudly_instead_of_crashing(self, tmp_path, capsys):
        """Same guard, via the apply subcommand: an empty codebook file
        must not crash apply_coding_record() with an unhandled exception."""
        path = tmp_path / "codebook.yaml"
        path.write_text("", encoding="utf-8")
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(json.dumps(_match_record()) + "\n", encoding="utf-8")

        ret = mod.main(["apply", str(path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 1
        assert captured.err.strip() != ""

    def test_migrate_invalid_output_does_not_clobber_file(self, tmp_path, capsys):
        """If migrate_v1_to_v2()'s output still fails validate() (e.g. a
        v1 entry with an out-of-enum severity — a field migrate does not
        touch), _cmd_migrate must report the errors, return 1, and leave
        the on-disk file exactly as it was: the migrated-but-invalid
        result is never dumped over the live registry."""
        v1 = _v1_fixture()
        v1["entries"][0]["severity"] = "catastrophic"  # not in {high,medium,low}
        path = tmp_path / "codebook.yaml"
        mod.dump(v1, path)
        before = path.read_bytes()

        ret = mod.main(["migrate", str(path)])
        captured = capsys.readouterr()

        assert ret == 1
        assert captured.err.strip() != ""
        assert path.read_bytes() == before


# ---------------------------------------------------------------------------
# step-17: RED (live-file guard, PRD §11 γ observable) — committed codebook
# ---------------------------------------------------------------------------

# scripts/tests/test_codebook.py -> scripts/tests/ -> scripts/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_CODEBOOK_PATH = _REPO_ROOT / "docs" / "legibility" / "confusion-codebook.yaml"


def test_live_codebook_is_v2_and_validates_green():
    """Pins that the committed registry conforms post-migration and guards
    future hand-edits: version == 2 and validate() returns zero errors.
    RED until the in-place v1->v2 migration lands (file is still v1)."""
    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    assert codebook["version"] == 2
    assert mod.validate(codebook) == []


def test_live_codebook_has_no_pending_twin_of_an_adjudicated_candidate():
    """Live-file guard against the resurrection that already happened.

    (a) is the resurrection signature: the merger fabricated a *pending* twin
    for a title the census had already adjudicated (rejected cand-20260722-28
    came back as pending cand-20260724-2). (b) is the forward guard — already
    green today — asserted so the invariant is stated whole.

    This is the checkable invariant the merge fix guarantees going forward: a
    title with an existing pending record takes the `pending_match` branch; a
    title whose records are all adjudicated takes the conflict branch (sighting
    appended, disposition untouched, NO new pending); only a genuinely unseen
    title reaches the create branch. census.py only flips dispositions and
    never creates candidates, so it can only violate (a) by rejecting one of
    two same-title pendings — which (b) proves cannot currently arise.

    Deliberately asserted on the pending-vs-adjudicated split rather than on
    duplicate titles outright: the never-delete contract forbids removing the
    resurrected record, so the repair can only restore its verdict, not erase
    it, and a "no duplicate titles" assertion could never go green.
    """
    codebook = mod.load(_LIVE_CODEBOOK_PATH)

    by_title = {}
    for candidate in codebook.get("candidates") or []:
        by_title.setdefault(candidate.get("title"), []).append(candidate)

    def _describe(title):
        return "{!r}: {}".format(
            title,
            [
                (c.get("id"), c.get("disposition"), c.get("first_seen"))
                for c in by_title[title]
            ],
        )

    # (a) THE RESURRECTION SIGNATURE — a pending record alongside an
    #     already-adjudicated one for the byte-identical title.
    resurrected = [
        title
        for title, group in by_title.items()
        if any(c.get("disposition") == "pending" for c in group)
        and any(c.get("disposition") in {"rejected", "promoted"} for c in group)
    ]
    assert resurrected == [], (
        "pending candidate(s) coexist with an already-adjudicated candidate of "
        "the same title — the census verdict was resurrected: "
        + "; ".join(_describe(t) for t in resurrected)
    )

    # (b) forward guard — never two pending records for one title.
    duplicated = [
        title
        for title, group in by_title.items()
        if sum(1 for c in group if c.get("disposition") == "pending") > 1
    ]
    assert duplicated == [], (
        "more than one pending candidate shares a title: "
        + "; ".join(_describe(t) for t in duplicated)
    )


# ---------------------------------------------------------------------------
# task 4892 (live-file guard) — both corrections landed through the sole
# writer, and the committed registry still carries the committed record.
# ---------------------------------------------------------------------------

_T4892_ENTRY_ID = "guards-assert-unverified-diagnoses"
_T4892_ORIGINAL_SESSION = "e16af1c5-de98-4252-8504-ce5d13bee3d6"
_T4892_CORRECTION_SESSION = "task-4892-resume-sysprompt-correction"
_T4892_CANDIDATE_SESSION = "task-4892-lettered-option-collision"
_T4892_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-4892-corrections.jsonl"
)

# What apply_coding_record copies verbatim from a match/candidate payload into
# the sighting it appends. The optional three are emitted only when truthy, so
# a payload that omits one is compared as omitting it.
_SIGHTING_PAYLOAD_FIELDS = (
    "origin_phase",
    "manifested_phase",
    "invariant_violated",
    "note",
    "evidence_quote",
)


def _sightings_for(holder: dict, session: str) -> list[dict]:
    return [s for s in holder.get("sightings") or [] if s.get("session") == session]


def _t4892_carriers(codebook: dict, record: dict):
    """Each payload of *record* with (its key, the live records the sole writer
    appends it to): a match goes to the entry named by its `entry_id`, a
    candidate to the candidate(s) sharing its `title`."""
    for match in record.get("matches") or []:
        key = match.get("entry_id")
        yield key, match, [e for e in codebook.get("entries") or [] if e.get("id") == key]
    for candidate in record.get("candidates") or []:
        key = candidate.get("title")
        yield key, candidate, [
            c for c in codebook.get("candidates") or [] if c.get("title") == key
        ]


def test_live_codebook_carries_the_task_4892_corrections():
    """The committed §7.3 record and the committed registry still agree.

    Both corrections went through the sole writer, which can only APPEND: the
    now-false present-tense claim about `--system-prompt-file` is superseded by
    a sibling sighting rather than rewritten in place, and the
    lettered-option-collision pattern is filed as a candidate for the census to
    adjudicate. Four properties, each invariant under every transition the
    system is designed to make to this file:

    (1) AGREEMENT — every payload in the record is carried, exactly once, by
        exactly one live entry/candidate, field for field. A hand-edit of the
        YAML (the in-place rewrite the append-only contract exists to prevent)
        surfaces here as a mismatch. Two committed data artifacts compared
        against each other — not the prose pin esc-4892-4 removed, which
        asserted a substring of a note this record does not own.
    (2) NEVER-DELETE — the 2026-08-10 observation is still present, still so
        dated. It was true when made; only its present tense expired.
    (3) ORDER — the correction follows what it supersedes: its note points the
        reader at "the wording above".
    (4) NO-OP — re-applying the record appends nothing, which is what makes a
        rebase onto a nightly-rewritten main resolvable by re-running one CLI
        command instead of hand-editing 22k lines of generated YAML.

    Schema shape is not re-checked here: `test_live_codebook_is_v2_and_validates_green`
    owns `validate() == []` over this file. The candidate's `disposition` and
    its `cand-<yyyymmdd>-<n>` id are not pinned either — the census owns the
    first and the merger derives the second from the same-date candidate count,
    so both may legitimately change; each is located by its stable sighting
    session instead.
    """
    records = [
        json.loads(line)
        for line in _T4892_RECORD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sessions = {r.get("session") for r in records}
    assert {_T4892_CORRECTION_SESSION, _T4892_CANDIDATE_SESSION} <= sessions, (
        f"{_T4892_RECORD_PATH} no longer records both task-4892 corrections: "
        f"{sorted(sessions)}"
    )

    codebook = mod.load(_LIVE_CODEBOOK_PATH)

    # (1) AGREEMENT — the record is carried verbatim, once, by one holder.
    for record in records:
        session = record.get("session")
        assert mod.validate_coding_record(record) == [], (
            f"invalid §7.3 coding record for session {session!r}"
        )
        for key, payload, holders in _t4892_carriers(codebook, record):
            carrying = [h for h in holders if _sightings_for(h, session)]
            assert len(carrying) == 1, (
                f"expected exactly one live record under {key!r} to carry a "
                f"{session!r} sighting, found {[h.get('id') for h in carrying]}"
            )
            carried = _sightings_for(carrying[0], session)
            assert len(carried) == 1, (
                "the sole writer must not duplicate a sighting on a re-run — "
                f"{session!r} appears {len(carried)}x on {carrying[0].get('id')!r}"
            )
            drift = {
                field: (payload[field], carried[0].get(field))
                for field in _SIGHTING_PAYLOAD_FIELDS
                if payload.get(field) and payload[field] != carried[0].get(field)
            }
            assert drift == {}, (
                f"the {session!r} sighting on {carrying[0].get('id')!r} no longer "
                "matches the committed record — a committed sighting was "
                f"rewritten in place {{field: (record, codebook)}}: {drift}"
            )

    # (2) NEVER-DELETE and (3) ORDER, on the annotated entry.
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == _T4892_ENTRY_ID]
    assert len(entries) == 1, f"expected exactly one {_T4892_ENTRY_ID!r} entry"
    sightings = entries[0].get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}

    assert _T4892_ORIGINAL_SESSION in order, (
        f"the 2026-08-10 sighting was deleted from {_T4892_ENTRY_ID!r} — "
        "sightings are immutable dated observations"
    )
    original = sightings[order[_T4892_ORIGINAL_SESSION]]
    assert original.get("date") == "2026-08-10", (
        f"the {_T4892_ORIGINAL_SESSION!r} sighting was re-dated to "
        f"{original.get('date')!r} — a dated observation is not editable"
    )
    assert order.get(_T4892_CORRECTION_SESSION, -1) > order[_T4892_ORIGINAL_SESSION], (
        f"the {_T4892_CORRECTION_SESSION!r} sighting is missing from "
        f"{_T4892_ENTRY_ID!r} or does not follow the 2026-08-10 sighting it "
        "supersedes — its note points the reader at 'the wording above'"
    )

    # (4) NO-OP — the record is already fully absorbed, so re-running the sole
    #     writer over it appends nothing.
    for record in records:
        codebook, stats = mod.apply_coding_record(codebook, record)
        assert stats == _NO_CHANGES, (
            f"re-applying session {record.get('session')!r} appended something — "
            f"its sightings are not already absorbed by the live codebook: {stats}"
        )


# ---------------------------------------------------------------------------
# task 5198 (live-file guard, esc-5120-2) — the refuted task-5120 framing is
# withdrawn from the committed registry, through the sole writer.
# ---------------------------------------------------------------------------

_T5198_ENTRY_ID = "entry-cand-20260831-16"
_T5198_CORRECTION_SESSION = "task-5198-verify-summary-glob-refutation"
_T5198_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5198-corrections.jsonl"
)

# The three sightings the correction ANNOTATES. They observed a real symptom —
# agents genuinely did hit an opaque JSONDecodeError — and only the cause they
# were filed under was refuted, so each must survive, still carrying its date.
_T5198_ORIGINAL_SIGHTINGS = {
    "273579c5-e299-40c6-ab02-e4c0ab24213b": "2026-08-31",
    "8a483603-92b1-460b-8af1-65c02d70a598": "2026-09-05",
    "d83f509f-a7e5-4eeb-9e2b-74e80289204b": "2026-09-05",
}

# The token pair that made the entry self-ingesting: the census kept matching
# its own quoted error string back to this title.
_T5198_REFUTED_TOKENS = ("glob miss", "JSONDecodeError")


def test_live_codebook_withdraws_the_refuted_task_5120_framing():
    """The committed §7.3 record and the committed registry still agree, and
    the refuted framing is no longer asserted to the coder.

    `scripts/legibility/coder.py::build_codebook_index` renders one line per
    entry — `- {id}: {title} — {cause}` — and deliberately keeps retired
    entries in that index, so retirement alone would leave the refuted title
    in front of the very loop esc-5120-2 diagnoses as self-ingesting. Six
    properties:

    (1) AGREEMENT — the entry's title/cause/status equal the record's, and one
        sighting carries the record's payload field for field. This is the
        in-place-rewrite detector: a hand edit of the merger-owned YAML
        surfaces here as drift between two committed artifacts.
    (2) WITHDRAWN — the title no longer carries the distinctive token pair.
    (3) RETIRED — status is 'retired'.
    (4) NEVER-DELETE — all three original sightings survive, located by
        session, still so dated.
    (5) ORDER — the correction follows everything it annotates.
    (6) NO-OP — re-applying the record changes nothing, which is what makes a
        rebase onto a nightly-rewritten main resolvable by re-running one CLI
        command instead of hand-editing 22k lines of generated YAML.

    The entry's TOTAL sighting count is deliberately not pinned: retired
    entries stay in the coder's index, so the nightly census can legitimately
    append a fourth sighting after this amendment lands. Presence-of-each-
    original plus (5) carries never-delete without that coupling. Nor is any
    substring of the note's prose pinned — (1) already proves it is carried
    verbatim — and schema shape stays owned by
    `test_live_codebook_is_v2_and_validates_green`.
    """
    records = [
        json.loads(line)
        for line in _T5198_RECORD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_session = {r.get("session"): r for r in records}
    assert _T5198_CORRECTION_SESSION in by_session, (
        f"{_T5198_RECORD_PATH} no longer records the task-5198 correction: "
        f"{sorted(by_session)}"
    )
    record = by_session[_T5198_CORRECTION_SESSION]
    assert mod.validate_coding_record(record) == [], (
        f"invalid §7.3 coding record for session {_T5198_CORRECTION_SESSION!r}"
    )
    corrections = record.get("corrections") or []
    assert len(corrections) == 1, (
        f"expected exactly one correction in {_T5198_RECORD_PATH}, got {len(corrections)}"
    )
    correction = corrections[0]
    assert correction.get("entry_id") == _T5198_ENTRY_ID

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == _T5198_ENTRY_ID]
    assert len(entries) == 1, f"expected exactly one {_T5198_ENTRY_ID!r} entry"
    entry = entries[0]

    # (1) AGREEMENT — entry fields, then the provenance sighting.
    field_drift = {
        field: (correction[field], entry.get(field))
        for field in ("title", "cause", "status")
        if correction.get(field) and correction[field] != entry.get(field)
    }
    assert field_drift == {}, (
        f"{_T5198_ENTRY_ID!r} no longer carries the committed correction — a "
        f"merger-owned field was rewritten by hand {{field: (record, codebook)}}: "
        f"{field_drift}"
    )

    carried = _sightings_for(entry, _T5198_CORRECTION_SESSION)
    assert len(carried) == 1, (
        f"expected exactly one {_T5198_CORRECTION_SESSION!r} sighting on "
        f"{_T5198_ENTRY_ID!r}, found {len(carried)}"
    )
    sighting_drift = {
        field: (correction[field], carried[0].get(field))
        for field in _SIGHTING_PAYLOAD_FIELDS
        if correction.get(field) and correction[field] != carried[0].get(field)
    }
    assert sighting_drift == {}, (
        f"the {_T5198_CORRECTION_SESSION!r} sighting no longer matches the "
        f"committed record {{field: (record, codebook)}}: {sighting_drift}"
    )

    # (2) WITHDRAWN — the framing the census kept re-ingesting is gone from the
    #     one field build_codebook_index actually renders.
    title_lower = entry["title"].lower()
    still_asserted = [t for t in _T5198_REFUTED_TOKENS if t.lower() in title_lower]
    assert still_asserted == [], (
        f"{_T5198_ENTRY_ID!r} still asserts the refuted framing via "
        f"{still_asserted} — build_codebook_index renders this title to the "
        f"coder every night: {entry['title']!r}"
    )

    # (3) RETIRED.
    assert entry.get("status") == "retired", (
        f"{_T5198_ENTRY_ID!r} is {entry.get('status')!r}, not 'retired'"
    )

    # (4) NEVER-DELETE and (5) ORDER.
    sightings = entry.get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}
    for session, date in _T5198_ORIGINAL_SIGHTINGS.items():
        assert session in order, (
            f"the {session!r} sighting was deleted from {_T5198_ENTRY_ID!r} — "
            "sightings are immutable dated observations, and the correction "
            "annotates them rather than retracting them"
        )
        original = sightings[order[session]]
        assert original.get("date") == date, (
            f"the {session!r} sighting was re-dated to {original.get('date')!r} "
            f"(was {date!r}) — a dated observation is not editable"
        )
    assert order[_T5198_CORRECTION_SESSION] > max(
        order[s] for s in _T5198_ORIGINAL_SIGHTINGS
    ), (
        f"the {_T5198_CORRECTION_SESSION!r} sighting does not follow the "
        "sightings it annotates"
    )

    # (6) NO-OP — the committed YAML is exactly what the sole writer produces.
    _, stats = mod.apply_coding_record(codebook, record)
    assert stats == _NO_CHANGES, (
        f"re-applying session {_T5198_CORRECTION_SESSION!r} changed something — "
        f"the committed codebook has not fully absorbed the record: {stats}"
    )


# ---------------------------------------------------------------------------
# task 5687 (live-file guard, confusion census 2026-09-20 §1.2/§5) — the
# refuted em-dash framing is withdrawn from the committed registry, through
# the sole writer.
# ---------------------------------------------------------------------------

_T5687_ENTRY_ID = "entry-cand-20260919-2"
_T5687_CORRECTION_SESSION = "task-5687-embedded-triple-quote-reproduction"
_T5687_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5687-corrections.jsonl"
)

# The founding observation the correction ANNOTATES, and the quote it was
# filed with. The symptom was real — python3 genuinely did report an invalid
# U+2014 — and only the cause it was filed under was refuted, so the sighting
# survives unedited rather than being retracted.
_T5687_ORIGINAL_SIGHTING = {"fbd7b22b-949a-4047-8087-6a719edb1dc2": "2026-09-19"}
_T5687_ORIGINAL_EVIDENCE_QUOTE = "SyntaxError: invalid character '—' (U+2014)"

# The withdrawn framing, as WORD tokens only, all lowercase, compared against
# lowercased text.
#
# The bare U+2014 CHARACTER is deliberately NOT in this tuple; it is asserted
# against the TITLE alone, by property (2). `build_codebook_index` renders
# each entry as `- {id}: {title} — {cause}`, and that separator IS U+2014,
# emitted exactly when the cause is non-empty. Banning the bare character from
# the rendered line would therefore contradict property (3)'s demand for a
# non-empty cause — no implementation could satisfy both. A title is a leaf
# value the separator cannot contaminate, so the character check belongs
# there. Do not "helpfully" add it back here.
_T5687_REFUTED_TOKENS = ("em-dash", "em dash", "u+2014", "unicode")


def test_live_codebook_withdraws_the_refuted_em_dash_framing():
    """The committed §7.3 record and the committed registry still agree, and
    the refuted em-dash framing is no longer asserted to the coder.

    The entry named a SYMPTOM, not a mechanism. Measured against the archived
    transcript, replacing every U+2014 in the failing script with an ASCII
    hyphen leaves it failing — with a different error — while escaping or
    deleting the one embedded `\"\"\"` fixes it with every U+2014 left in
    place. So the reported error string is unstable across recurrences of the
    same defect, and a title keyed to it gives the nightly coder nothing to
    match on. Seven properties:

    (0) WELL-FORMED — the record file is JSONL, exactly one of its records
        carries this session, it validates, and it holds exactly one
        `corrections` op naming this entry. Both counts are scoped BY SESSION,
        not file-wide: the sanctioned way to re-correct an entry later is a
        NEW session in the same file (task 5686's record holds five ops across
        two), and a file-wide count would break the moment anyone appends.
        Exactly-one matters because the merger admits one sighting per
        (session, entry), so a second op for this entry in this session would
        be counted in `correction_skipped` and its field writes would vanish.
    (1) AGREEMENT — the entry's title/cause/status equal the record's, and one
        sighting carries the record's payload field for field. This is the
        in-place-rewrite detector for the merger-owned YAML, and it is what
        proves the full cause text is carried verbatim — which is why no later
        property needs to pin cause prose.
    (2) WITHDRAWN (TITLE) — no refuted word token, and no bare U+2014, in the
        title. This is the assertion the task exists for.
    (3) CODER-VISIBLE CAUSE — the cause is non-empty, and the 200-character
        window `_one_line_cause` actually hands the coder carries no refuted
        token. Checking the WINDOW rather than the whole cause is deliberate:
        that window is all the coder ever reads, while the cause BODY must
        stay free to name the em-dash in order to REFUTE it.
    (4) STILL OPEN — this correction replaces an explanation; it does not
        retire a live, unfixed defect. Pinned so a later sweep cannot quietly
        retire the entry and keep this guard green. (Diverges deliberately
        from the task-5198 guard above, which pins 'retired'.)
    (5) NEVER-DELETE and ORDER — the founding 2026-09-19 sighting survives,
        still so dated, still carrying its original quote, and the correction
        follows it.
    (6) NO-OP — re-applying the record changes nothing, which makes a rebase
        onto a nightly-rewritten main resolvable by re-running one CLI command
        instead of hand-editing 22k lines of generated YAML.

    The entry's TOTAL sighting count is deliberately not pinned — the nightly
    census may legitimately append more. Nor is any substring of the note's
    prose: (1) already proves it is carried verbatim, and schema shape stays
    owned by `test_live_codebook_is_v2_and_validates_green`.
    """
    # (0) WELL-FORMED.
    records = [
        json.loads(line)
        for line in _T5687_RECORD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [r for r in records if r.get("session") == _T5687_CORRECTION_SESSION]
    assert len(mine) == 1, (
        f"expected exactly one {_T5687_CORRECTION_SESSION!r} record in "
        f"{_T5687_RECORD_PATH}, found {len(mine)}; sessions present: "
        f"{sorted({r.get('session') for r in records})}"
    )
    record = mine[0]
    assert mod.validate_coding_record(record) == [], (
        f"invalid §7.3 coding record for session {_T5687_CORRECTION_SESSION!r}"
    )
    corrections = record.get("corrections") or []
    assert len(corrections) == 1, (
        f"expected exactly one correction in the {_T5687_CORRECTION_SESSION!r} "
        f"record, got {len(corrections)} — the merger admits one sighting per "
        "(session, entry), so a second op for this entry in this session would "
        "be dropped into `correction_skipped` with its field writes lost"
    )
    correction = corrections[0]
    assert correction.get("entry_id") == _T5687_ENTRY_ID

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == _T5687_ENTRY_ID]
    assert len(entries) == 1, f"expected exactly one {_T5687_ENTRY_ID!r} entry"
    entry = entries[0]

    # (1) AGREEMENT — entry fields, then the provenance sighting.
    field_drift = {
        field: (correction[field], entry.get(field))
        for field in ("title", "cause", "status")
        if correction.get(field) and correction[field] != entry.get(field)
    }
    assert field_drift == {}, (
        f"{_T5687_ENTRY_ID!r} no longer carries the committed correction — a "
        f"merger-owned field was rewritten by hand {{field: (record, codebook)}}: "
        f"{field_drift}"
    )

    carried = _sightings_for(entry, _T5687_CORRECTION_SESSION)
    assert len(carried) == 1, (
        f"expected exactly one {_T5687_CORRECTION_SESSION!r} sighting on "
        f"{_T5687_ENTRY_ID!r}, found {len(carried)}"
    )
    sighting_drift = {
        field: (correction[field], carried[0].get(field))
        for field in _SIGHTING_PAYLOAD_FIELDS
        if correction.get(field) and correction[field] != carried[0].get(field)
    }
    assert sighting_drift == {}, (
        f"the {_T5687_CORRECTION_SESSION!r} sighting no longer matches the "
        f"committed record {{field: (record, codebook)}}: {sighting_drift}"
    )

    # (2) WITHDRAWN (TITLE) — the refuted symptom is gone from the one field
    #     build_codebook_index renders unconditionally.
    title = entry["title"]
    title_lower = title.lower()
    still_asserted = [t for t in _T5687_REFUTED_TOKENS if t in title_lower]
    assert still_asserted == [], (
        f"{_T5687_ENTRY_ID!r} still asserts the refuted framing via "
        f"{still_asserted} — build_codebook_index renders this title to the "
        f"coder every night, and the symptom it names does not survive an "
        f"ASCII-only recurrence of the same defect: {title!r}"
    )
    assert "—" not in title, (
        f"{_T5687_ENTRY_ID!r} still spells the refuted character in its "
        f"title: {title!r}"
    )

    # (3) CODER-VISIBLE CAUSE — asserted against the REAL renderer, never a
    #     copy of its format string, so a change to either surfaces here.
    cause = entry.get("cause")
    assert cause, (
        f"{_T5687_ENTRY_ID!r} has no cause — with none, build_codebook_index "
        "shows the coder the title alone, which is the state this task exists "
        "to end"
    )
    window = coder._one_line_cause(cause).lower()
    still_rendered = [t for t in _T5687_REFUTED_TOKENS if t in window]
    assert still_rendered == [], (
        f"the {coder._INDEX_CAUSE_MAX_LEN}-character cause window the coder "
        f"actually reads still carries {still_rendered}: {window!r}"
    )
    rendered = coder.build_codebook_index({"entries": [entry]})
    assert _T5687_ENTRY_ID in rendered and title in rendered, (
        f"build_codebook_index no longer renders this entry as expected: {rendered!r}"
    )

    # (4) STILL OPEN — the defect is live and unfixed; only its explanation
    #     was wrong.
    assert entry.get("status") == "open", (
        f"{_T5687_ENTRY_ID!r} is {entry.get('status')!r}, not 'open' — this "
        "correction replaces an explanation, it does not retire a live defect"
    )

    # (5) NEVER-DELETE and ORDER.
    sightings = entry.get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}
    for session, date in _T5687_ORIGINAL_SIGHTING.items():
        assert session in order, (
            f"the {session!r} sighting was deleted from {_T5687_ENTRY_ID!r} — "
            "sightings are immutable dated observations, and the correction "
            "annotates them rather than retracting them"
        )
        original = sightings[order[session]]
        assert original.get("date") == date, (
            f"the {session!r} sighting was re-dated to {original.get('date')!r} "
            f"(was {date!r}) — a dated observation is not editable"
        )
        assert original.get("evidence_quote") == _T5687_ORIGINAL_EVIDENCE_QUOTE, (
            f"the {session!r} sighting's evidence_quote was rewritten to "
            f"{original.get('evidence_quote')!r} — the symptom it recorded was "
            "real and is not what the correction withdraws"
        )
    assert order[_T5687_CORRECTION_SESSION] > max(
        order[s] for s in _T5687_ORIGINAL_SIGHTING
    ), (
        f"the {_T5687_CORRECTION_SESSION!r} sighting does not follow the "
        "sighting it annotates"
    )

    # (6) NO-OP — the committed YAML is exactly what the sole writer produces.
    _, stats = mod.apply_coding_record(codebook, record)
    assert stats == _NO_CHANGES, (
        f"re-applying session {_T5687_CORRECTION_SESSION!r} changed something — "
        f"the committed codebook has not fully absorbed the record: {stats}"
    )
