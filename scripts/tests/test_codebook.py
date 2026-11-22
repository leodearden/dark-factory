"""Tests for scripts/legibility/codebook.py — confusion-codebook v2 schema,
validator, and deterministic sole-writer merger (task 2575).

See plans/confusion-reduction-prd.md §7.1 (codebook v2), §7.3 (coding
record), §8.2/§8.3 (idempotency + never-delete boundary tests).

Imported as a namespace package (`from legibility import codebook as mod`)
since scripts/legibility/ is a subdir of scripts/ (on sys.path via
scripts/tests/conftest.py) with no __init__.py — confirmed empirically to
resolve under pytest's --import-mode=importlib.

The guards over the committed codebook and coding records live in
scripts/tests/test_live_codebook.py.
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest
from codebook_merge_report import NO_SLUGS, clean_merge_report, merge_one, merge_report
from legibility import codebook as mod
from legibility import invariants
from shared.finding_key import finding_key
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
    """Amendment: under the sole-writer design, dump()/apply_coding_records()
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
# step-9: RED (§8.2 idempotency, match path) — apply_coding_records()
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


def test_merge_appends_one_sighting():
    codebook = _codebook_with_entry_a()
    record = _match_record()

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["sightings"] == [
        {
            "date": "2026-07-14",
            "project": "dark_factory",
            "session": "sess-1",
            "origin_phase": "implement",
            "manifested_phase": "merge",
        }
    ]
    assert mod.validate(merge.codebook) == []


def test_merge_match_is_idempotent_on_session_and_entry():
    codebook = _codebook_with_entry_a()
    record = _match_record()

    once = merge_one(codebook, record).codebook
    twice = merge_one(once, record).codebook

    entry = next(e for e in twice["entries"] if e["id"] == "entry-a")
    assert len(entry["sightings"]) == 1
    assert mod.validate(twice) == []


def test_merge_unknown_entry_id_is_skipped_and_counted():
    codebook = _codebook_with_entry_a()
    record = _match_record(entry_id="entry-zzz")

    merge = merge_one(codebook, record)

    assert len(merge.codebook["entries"]) == 1  # no entry fabricated
    assert merge.codebook["entries"][0]["sightings"] == []
    assert merge.counts.skipped_unknown_entry == 1
    assert mod.validate(merge.codebook) == []


def test_merge_does_not_mutate_input():
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _match_record()

    merge_one(codebook, record)

    assert codebook == original


# ---------------------------------------------------------------------------
# step-11: RED (§8.3 candidate append) — apply_coding_records() candidates
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


def test_merge_appends_one_candidate():
    codebook = _codebook_with_entry_a()
    record = _candidate_record()

    merge = merge_one(codebook, record)

    assert len(merge.codebook["candidates"]) == 1
    candidate = merge.codebook["candidates"][0]
    assert candidate["id"] == "cand-20260714-1"
    assert candidate["first_seen"] == "2026-07-14"
    assert candidate["disposition"] == "pending"
    assert candidate["title"] == "novel shape"
    assert len(candidate["sightings"]) == 1
    assert candidate["sightings"][0]["session"] == "sess-1"
    assert merge.counts.candidates_applied == 1
    assert mod.validate(merge.codebook) == []


def test_merge_candidate_is_idempotent_on_session_and_title():
    codebook = _codebook_with_entry_a()
    record = _candidate_record()

    once = merge_one(codebook, record).codebook
    twice = merge_one(once, record).codebook

    assert len(twice["candidates"]) == 1
    assert len(twice["candidates"][0]["sightings"]) == 1
    assert mod.validate(twice) == []


def test_merge_different_candidate_same_day_increments_id():
    codebook = _codebook_with_entry_a()
    record_a = _candidate_record(title="novel shape")
    once = merge_one(codebook, record_a).codebook

    record_b = _candidate_record(title="a different shape")
    twice = merge_one(once, record_b).codebook

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
    the shape `codebook_lifecycle.reject_candidate`/`promote_candidate` leave behind."""
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


def test_merge_does_not_resurrect_rejected_candidate():
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    merge = merge_one(codebook, record)

    assert len(merge.codebook["candidates"]) == 1  # NO pending twin fabricated
    candidate = merge.codebook["candidates"][0]
    assert candidate["id"] == "cand-20260722-28"
    assert candidate["disposition"] == "rejected"  # census verdict untouched
    # recurrence signal preserved, append-only
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert merge.counts.candidate_disposition_conflicts == 1
    assert merge.counts.candidates_applied == 0
    assert mod.validate(merge.codebook) == []


def test_merge_does_not_duplicate_promoted_candidate():
    """The FALLBACK half of the promoted case: this fixture's candidate is
    stamped `promoted` but carries no `promoted_to` (a hand-edited or
    pre-`promote_candidate` record), so there is no entry to route the
    recurrence to — it lands on the candidate and counts as a conflict,
    exactly like the rejected case."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    merge = merge_one(codebook, record)

    assert len(merge.codebook["candidates"]) == 1
    candidate = merge.codebook["candidates"][0]
    assert candidate["id"] == "cand-20260722-28"
    assert candidate["disposition"] == "promoted"  # census verdict untouched
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert merge.counts.candidate_disposition_conflicts == 1
    assert merge.counts.candidates_applied == 0
    assert mod.validate(merge.codebook) == mod.validate(codebook) != []


def test_merge_routes_promoted_recurrence_to_its_entry():
    """A promoted candidate's `sightings` list is a DEAD field for new
    signal: `codebook_lifecycle.promote_candidate` deep-copies it into the new entry
    once, at promotion time, and nothing re-reads it afterwards (the matrix
    path and the codebook index both read ENTRY sightings). So a recurrence
    of a promoted title must land on the entry named by `promoted_to` and
    count as a `matched` sighting — filing it on the candidate would report
    the signal as preserved while writing it where no consumer looks."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-a"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-new"]
    assert merge.counts.matched == 1
    assert merge.counts.candidate_disposition_conflicts == 0
    assert merge.counts.candidates_applied == 0

    # The candidate itself is untouched — verdict AND sightings; no twin.
    assert len(merge.codebook["candidates"]) == 1
    candidate = merge.codebook["candidates"][0]
    assert candidate["disposition"] == "promoted"
    assert len(candidate["sightings"]) == 1
    assert mod.validate(merge.codebook) == []


def test_merge_promoted_recurrence_dedupes_on_the_entry():
    """Session-level dedup on the ENTRY, mirroring the match path: a record
    that both matches the entry directly and re-mines the promoted title
    appends exactly one sighting, not two."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-a"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")
    record["matches"] = [
        {"entry_id": "entry-a", "origin_phase": "implement", "manifested_phase": "merge"}
    ]

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-new"]
    assert merge.counts.matched == 1  # the match path's append; the candidate re-sighting deduped
    assert merge.counts.candidate_disposition_conflicts == 0
    assert merge.counts.candidates_applied == 0
    assert len(merge.codebook["candidates"][0]["sightings"]) == 1
    assert mod.validate(merge.codebook) == []


def test_merge_promoted_with_dangling_promoted_to_falls_back():
    """`promoted_to` naming an entry that does not exist must NOT silently
    drop the recurrence: fall back to the candidate append + conflict
    counter (the merger never fabricates an entry — only the census does)."""
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-that-was-never-created"
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    merge = merge_one(codebook, record)

    assert len(merge.codebook["entries"]) == 1  # no entry fabricated
    assert merge.codebook["entries"][0]["sightings"] == []
    candidate = merge.codebook["candidates"][0]
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert merge.counts.matched == 0
    assert merge.counts.candidate_disposition_conflicts == 1
    assert mod.validate(merge.codebook) == mod.validate(codebook) != []


def test_merge_adjudicated_conflict_is_idempotent():
    """The pre-existing `already_seen` guard already spans every
    disposition, so re-applying the same record appends nothing and reports
    no fresh conflict."""
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    once = merge_one(codebook, record)
    twice = merge_one(once.codebook, record)

    assert len(twice.codebook["candidates"]) == 1
    assert len(twice.codebook["candidates"][0]["sightings"]) == 2  # unchanged by the 2nd apply
    assert once.counts.candidate_disposition_conflicts == 1
    assert twice.counts.candidate_disposition_conflicts == 0
    assert mod.validate(twice.codebook) == []


def test_merge_still_creates_pending_for_a_genuinely_new_title():
    """Regression guard: the new conflict branch must not swallow genuinely
    novel titles — only an ALREADY-SEEN title takes it."""
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(
        title="a different shape", session="sess-new", date="2026-07-24"
    )

    merge = merge_one(codebook, record)

    assert len(merge.codebook["candidates"]) == 2
    fresh = next(c for c in merge.codebook["candidates"] if c["title"] == "a different shape")
    assert fresh["disposition"] == "pending"
    assert fresh["id"] == "cand-20260724-1"
    assert merge.counts.candidates_applied == 1
    assert merge.counts.candidate_disposition_conflicts == 0
    assert mod.validate(merge.codebook) == []


# ---------------------------------------------------------------------------
# task 5198: RED — apply_coding_records() `corrections`, the third op.
#
# A sighting is an immutable dated observation, so both existing ops are pure
# appends. An entry's title/cause is instead the CURRENT best explanation of a
# cause, and it is the only part of an entry that
# `scripts/legibility/coder.py::build_codebook_index` renders to the coder —
# so a refuted framing can only be withdrawn by rewriting it there.
# ---------------------------------------------------------------------------

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


def test_merge_correction_rewrites_only_the_named_fields():
    """Writes title/cause/status, leaves every other entry field alone, and
    appends exactly one sighting built from the record header + the
    correction's own phase stamps and note."""
    codebook = _codebook_with_entry_a()
    before = copy.deepcopy(codebook["entries"][0])
    record = _correction_record()
    correction = record["corrections"][0]

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
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
    assert merge_report(merge) == clean_merge_report(corrections_applied=1)
    assert mod.validate(merge.codebook) == []


def test_merge_correction_never_deletes_and_does_not_mutate_input():
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

    result = merge_one(codebook, _correction_record()).codebook

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == [
        "sess-original",
        "sess-correction",
    ]
    mod.assert_no_deletion(original, result)
    assert codebook == original


def test_merge_correction_re_apply_is_a_no_op(caplog):
    """The whole correction — field writes AND provenance sighting — is gated
    by the one `session` dedup the match path already uses, so `apply` re-run
    over the same file (nightly, or to resolve a rebase) changes nothing."""
    codebook = _codebook_with_entry_a()
    record = _correction_record()

    with caplog.at_level(logging.WARNING, logger="legibility.codebook"):
        once = merge_one(codebook, record).codebook
        twice = merge_one(once, record)

    assert merge_report(twice) == clean_merge_report()
    assert twice.codebook == once
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


def test_merge_correction_outranks_a_sibling_match_on_one_entry():
    """Both ops write at most one sighting per (session, entry), so a record
    carrying both for one entry can land only one — and it must be the
    correction: a match's sighting is one interchangeable observation, while
    the correction is the only op that can withdraw a refuted framing."""
    codebook = _codebook_with_entry_a()
    record = _match_and_correction_record()
    correction = record["corrections"][0]

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["title"] == correction["title"]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert merge_report(merge) == clean_merge_report(corrections_applied=1)
    assert mod.validate(merge.codebook) == []


def test_merge_colliding_correction_is_counted_and_logged(caplog):
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
        merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["title"] == record["corrections"][0]["title"]
    assert merge_report(merge) == clean_merge_report(corrections_applied=1, correction_skipped=1)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "entry-a" in message and record["session"] in message
    assert mod.validate(merge.codebook) == []


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
def test_merge_correction_never_clears_a_field(field, spelling):
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

    merge = merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry[field] == before[field]
    for still_supplied in {"title", "cause", "status"} - {field}:
        assert entry[still_supplied] == correction[still_supplied]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert merge_report(merge) == clean_merge_report(corrections_applied=1)
    assert mod.validate(merge.codebook) == []


def test_merge_correction_unknown_entry_id_is_skipped_and_counted():
    """Mirrors the match path: only the census creates entries, so an
    unresolvable entry_id is counted, never fabricated."""
    codebook = _codebook_with_entry_a()
    record = _correction_record(entry_id="entry-zzz")

    merge = merge_one(codebook, record)

    assert merge.codebook["entries"] == _codebook_with_entry_a()["entries"]
    assert merge_report(merge) == clean_merge_report(skipped_unknown_entry=1)
    assert mod.validate(merge.codebook) == []


@pytest.mark.parametrize("action", ["delete", "remove"])
def test_merge_skips_a_correction_removal_action(action):
    """A removal-shaped correction is skipped whole, exactly as
    `_deletion_directive` already treats `matches`."""
    record = _correction_record()
    record["corrections"][0]["action"] = action

    skipped = _skipped_as_deletion(record)

    assert f"correction carries a removal action: {action!r}" in skipped.reason


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
def test_merge_invalid_correction_is_skipped_whole(build_record):
    """An entry's framing may never change without an audit trail, and never
    into a status outside STATUSES. Every such failure is caught by
    `validate_coding_record`, which skips the record WHOLE — no partial field
    write survives."""
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = build_record()

    assert mod.validate_coding_record(record) != []

    merge = merge_one(codebook, record)

    assert merge.counts == mod.MergeCounts()
    assert [(s.index, s.session) for s in merge.invalid_records] == [(0, record["session"])]
    assert merge.skipped_deletions == merge.merged_records == ()
    assert merge.codebook == original


# ---------------------------------------------------------------------------
# task 7448: a correction that names the task whose whole outcome for the
# entry it is (`outcome_of_task`). The merger records that task on the
# entry's `corrected_by`, which disposition reads, so the task's done is never
# taken for a fix of the entry's defect.
# ---------------------------------------------------------------------------

_T7448_UNIT_TASK_REF = {"project_root": "/r", "task_id": 5687}


def _outcome_correction_record(
    session="sess-outcome", task_ref: dict | None = None
) -> dict:
    """A note-only §7.3 correction naming *task_ref* (default task 5687 of
    root "/r") in `outcome_of_task`: no title, cause or status."""
    return {
        "session": session,
        "date": "2026-10-24",
        "project": "dark_factory",
        "agent_class": "task-agent",
        "corrections": [
            {
                "entry_id": "entry-a",
                "note": "The task's whole outcome was this record's correction.",
                "outcome_of_task": dict(task_ref or _T7448_UNIT_TASK_REF),
            }
        ],
    }


def test_validate_coding_record_accepts_a_correction_naming_its_outcome_task():
    assert mod.validate_coding_record(_outcome_correction_record()) == []


@pytest.mark.parametrize(
    "task_ref",
    [
        {"project_root": "/r", "task_id": "5687"},
        {"project_root": "/r", "task_id": True},
        {"task_id": 5687},
        {"project_root": "", "task_id": 5687},
        {"project_root": "/r"},
    ],
    ids=["str-task-id", "bool-task-id", "no-project-root", "empty-project-root", "no-task-id"],
)
def test_validate_coding_record_rejects_a_malformed_outcome_task(task_ref):
    record = _outcome_correction_record()
    record["corrections"][0]["outcome_of_task"] = task_ref

    errors = mod.validate_coding_record(record)

    assert errors and all("outcome_of_task" in error for error in errors)


def test_merge_outcome_correction_records_the_task_on_the_entry_only():
    codebook = _codebook_with_entry_a()
    before = copy.deepcopy(codebook["entries"][0])

    merge = merge_one(codebook, _outcome_correction_record())

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    for field in ("title", "status"):
        assert entry[field] == before[field]
    assert "cause" not in entry
    assert [s["session"] for s in entry["sightings"]] == ["sess-outcome"]
    assert "outcome_of_task" not in entry["sightings"][0]
    assert entry["sightings"][0]["note"]
    assert entry["corrected_by"] == [{"project_root": "/r", "task_id": 5687}]
    assert merge_report(merge) == clean_merge_report(corrections_applied=1)
    assert mod.validate(merge.codebook) == []


def test_merge_outcome_correction_re_apply_is_a_no_op():
    record = _outcome_correction_record()
    once = merge_one(_codebook_with_entry_a(), record).codebook

    twice = merge_one(once, record)

    assert twice.codebook == once
    assert merge_report(twice) == clean_merge_report()
    assert not twice.mutated


def test_merge_outcome_corrections_dedupe_the_task_and_append_new_ones_in_order():
    first = merge_one(_codebook_with_entry_a(), _outcome_correction_record("sess-1")).codebook
    same_task = merge_one(first, _outcome_correction_record("sess-2")).codebook
    entry = next(e for e in same_task["entries"] if e["id"] == "entry-a")
    assert entry["corrected_by"] == [{"project_root": "/r", "task_id": 5687}]

    other_task = merge_one(
        same_task,
        _outcome_correction_record("sess-3", {"project_root": "/r", "task_id": 5688}),
    ).codebook

    entry = next(e for e in other_task["entries"] if e["id"] == "entry-a")
    assert entry["corrected_by"] == [
        {"project_root": "/r", "task_id": 5687},
        {"project_root": "/r", "task_id": 5688},
    ]
    assert [s["session"] for s in entry["sightings"]] == ["sess-1", "sess-2", "sess-3"]


def test_merge_correction_without_an_outcome_task_writes_no_corrected_by():
    merge = merge_one(_codebook_with_entry_a(), _correction_record())

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert "corrected_by" not in entry


def test_validate_accepts_a_well_formed_corrected_by():
    codebook = _codebook_with_entry_a()
    codebook["entries"][0]["corrected_by"] = [{"project_root": "/r", "task_id": 5687}]

    assert mod.validate(codebook) == []


def test_validate_rejects_a_corrected_by_item_with_a_string_task_id():
    codebook = _codebook_with_entry_a()
    codebook["entries"][0]["corrected_by"] = [{"project_root": "/r", "task_id": "5687"}]

    errors = mod.validate(codebook)

    assert errors and all("corrected_by" in error for error in errors)


# ---------------------------------------------------------------------------
# step-13: RED (§8.3 never-delete) — NeverDeleteError + assert_no_deletion
# ---------------------------------------------------------------------------

def _skipped_as_deletion(record: dict):
    """Merge deletion-shaped *record* alone and hand back its SkippedRecord,
    after pinning that the batch neither raised, merged it, nor changed (or
    mutated) the codebook."""
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)

    merge = merge_one(codebook, record)

    [skipped] = merge.skipped_deletions
    assert (skipped.index, skipped.session) == (0, record["session"])
    assert merge.merged_records == merge.invalid_records == ()
    assert merge.counts == mod.MergeCounts()
    assert merge.codebook == original
    assert codebook == original
    return skipped


@pytest.mark.parametrize("key", ["delete", "remove", "retract", "drop"])
def test_merge_skips_a_top_level_removal_directive(key):
    record = _match_record()
    record[key] = ["entry-a"]

    skipped = _skipped_as_deletion(record)

    assert f"removal directive: top-level {key!r} key" in skipped.reason


def test_merge_skips_a_match_delete_action():
    record = _match_record()
    record["matches"][0]["action"] = "delete"

    skipped = _skipped_as_deletion(record)

    assert "match carries a removal action: 'delete'" in skipped.reason


def test_a_never_delete_breach_by_the_merger_itself_fails_the_batch(monkeypatch):
    """Only a deletion directive is the record's fault. The never-delete
    post-condition guards the merger, so its breach is a merger bug: it fails
    the batch rather than being filed as a skipped deletion."""
    def breached(before, after):
        raise mod.NeverDeleteError("entries dropped from codebook: ['entry-a']")

    monkeypatch.setattr(mod, "assert_no_deletion", breached)

    with pytest.raises(mod.NeverDeleteError, match="entries dropped"):
        merge_one(_codebook_with_entry_a(), _match_record())


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
    after["candidates"][0]["promoted_to"] = "entry-a"

    mod.assert_no_deletion(before, after)  # must not raise

    assert len(after["candidates"]) == 1
    assert mod.validate(after) == []


# ---------------------------------------------------------------------------
# task 6400: invariant_violated is screened against the project's slugs on
# write (plans/census-incremental-prd.md §4.8 row 15)
# ---------------------------------------------------------------------------

_KNOWN_SLUGS = ("known-slug", "other-slug")


def _slugged_record() -> dict:
    record = _match_record()
    record["matches"][0]["invariant_violated"] = "known-slug"
    record["candidates"] = [
        {"title": "novel shape", "invariant_violated": "free text not a slug"}
    ]
    return record


def _sighting_on(holder: dict, session: str) -> dict:
    return next(s for s in holder["sightings"] if s["session"] == session)


def test_a_known_slug_persists_and_an_unknown_one_is_stored_as_absent():
    merge = merge_one(_codebook_with_entry_a(), _slugged_record(), _KNOWN_SLUGS)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert _sighting_on(entry, "sess-1")["invariant_violated"] == "known-slug"
    candidate = next(c for c in merge.codebook["candidates"] if c["title"] == "novel shape")
    assert "invariant_violated" not in _sighting_on(candidate, "sess-1")
    assert merge.invariant_slugs == mod.SlugTally(
        valid=1, rejected=("free text not a slug",)
    )
    assert mod.validate(merge.codebook) == []


def test_a_correction_with_an_unknown_slug_lands_without_it():
    record = _correction_record()
    record["corrections"][0]["invariant_violated"] = "not-a-declared-slug"

    merge = merge_one(_codebook_with_entry_a(), record, _KNOWN_SLUGS)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["title"] == record["corrections"][0]["title"]
    assert "invariant_violated" not in _sighting_on(entry, "sess-correction")
    assert merge.counts.corrections_applied == 1
    assert merge.invariant_slugs.rejected == ("not-a-declared-slug",)


@pytest.mark.parametrize("value", [None, "", "<absent>"])
def test_an_empty_slug_is_neither_valid_nor_rejected(value):
    record = _match_record()
    if value == "<absent>":
        del record["matches"][0]["invariant_violated"]
    else:
        record["matches"][0]["invariant_violated"] = value

    merge = merge_one(_codebook_with_entry_a(), record, _KNOWN_SLUGS)

    assert merge.invariant_slugs == mod.SlugTally()


def test_a_project_without_slugs_rejects_every_value():
    merge = merge_one(_codebook_with_entry_a(), _slugged_record(), ())

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert "invariant_violated" not in _sighting_on(entry, "sess-1")
    assert merge.invariant_slugs == mod.SlugTally(
        valid=0, rejected=("known-slug", "free text not a slug")
    )


def test_the_slug_screen_does_not_mutate_the_record():
    record = _slugged_record()
    original = copy.deepcopy(record)

    merge_one(_codebook_with_entry_a(), record, ())

    assert record == original


def test_validate_still_accepts_a_legacy_free_text_slug():
    codebook = _codebook_with_entry_a()
    codebook["entries"][0]["sightings"] = [
        {
            "date": "2026-07-01",
            "project": "dark_factory",
            "session": "legacy",
            "origin_phase": "unknown",
            "manifested_phase": "unknown",
            "invariant_violated": "some legacy free text",
        }
    ]

    assert mod.validate(codebook) == []


def test_slug_tallies_add_up():
    total = mod.SlugTally(valid=1, rejected=("a",)).plus(
        mod.SlugTally(valid=2, rejected=("b", "a"))
    )

    assert total == mod.SlugTally(valid=3, rejected=("a", "b", "a"))


def test_unknown_slugs_are_warned_once_per_run(caplog):
    caplog.set_level(logging.DEBUG, logger="legibility.codebook")

    mod.warn_unknown_invariant_slugs(
        mod.SlugTally(valid=2, rejected=("unknown-x", "unknown-y", "unknown-x")),
        ("a",),
        run="run-1",
    )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == "legibility.codebook"
    message = warnings[0].getMessage()
    assert "run-1" in message
    assert "stored 3 " in message
    assert "['unknown-x', 'unknown-y']" in message
    assert message.count("unknown-x") == 1


def test_the_unknown_slug_warning_stays_one_bounded_line(caplog):
    """The values are LLM free text: a large run must not grow the line without
    bound, while the count stays exact."""
    caplog.set_level(logging.DEBUG, logger="legibility.codebook")
    rejected = tuple(f"free-text-{n:03d}-" + "z" * 1000 for n in range(50))

    mod.warn_unknown_invariant_slugs(
        mod.SlugTally(rejected=rejected), ("a",), run="run-1"
    )

    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    message = warning.getMessage()
    assert "stored 50 " in message
    assert "free-text-000-" in message
    assert "free-text-049-" not in message
    assert "more" in message
    assert len(message) < 2000


def test_a_slug_tallys_rejections_record_counts_every_value_and_lists_each_distinct_one_once():
    record = mod.SlugTally(
        valid=2, rejected=("unknown-y", "unknown-x", "unknown-y")
    ).rejections_record()

    assert record == {"count": 3, "values": ["unknown-x", "unknown-y"], "values_omitted": 0}
    assert json.loads(json.dumps(record)) == record


def test_a_tally_without_rejections_records_zero_not_absent():
    assert mod.SlugTally(valid=2).rejections_record() == {
        "count": 0, "values": [], "values_omitted": 0,
    }


def test_the_rejections_record_is_bounded_as_the_warning_is(caplog):
    caplog.set_level(logging.DEBUG, logger="legibility.codebook")
    tally = mod.SlugTally(
        rejected=tuple(f"free-text-{n:03d}-" + "z" * 1000 for n in range(50))
    )

    record = tally.rejections_record()

    assert record["count"] == 50
    assert len(record["values"]) == 10
    assert record["values_omitted"] == 40
    assert record["values"][0].startswith("free-text-000-")
    assert not any(value.startswith("free-text-049-") for value in record["values"])
    assert all(len(value) <= 83 and value.endswith("...") for value in record["values"])

    mod.warn_unknown_invariant_slugs(tally, ("a",), run="run-1")

    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert str(record["values"]) in warning.getMessage()


def test_values_that_truncate_alike_are_listed_once_and_the_rest_counted_omitted(caplog):
    caplog.set_level(logging.DEBUG, logger="legibility.codebook")
    prefix = "p" * 80
    tally = mod.SlugTally(rejected=(prefix + "-one", prefix + "-two", "short"))

    record = tally.rejections_record()
    mod.warn_unknown_invariant_slugs(tally, ("a",), run="run-1")

    assert record == {"count": 3, "values": [prefix + "...", "short"], "values_omitted": 1}
    [warning] = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert str(record["values"]) + " +1 more" in warning.getMessage()


def test_no_unknown_slugs_means_no_warning(caplog):
    caplog.set_level(logging.DEBUG, logger="legibility.codebook")

    mod.warn_unknown_invariant_slugs(mod.SlugTally(valid=2), ("a",), run="r1")

    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


# ---------------------------------------------------------------------------
# amendment: migrate_v1_to_v2()/apply_coding_records() fail loudly with a
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


def test_merge_raises_on_non_dict_codebook():
    with pytest.raises(ValueError):
        merge_one(_NOT_A_CODEBOOK, _match_record())


# ---------------------------------------------------------------------------
# task 6742: one MergeResult for every batch (nightly, census, CLI). A bad
# record is skipped and named; it never costs the rest of the batch.
# ---------------------------------------------------------------------------

def _deletion_shaped_record(session: str) -> dict:
    record = _match_record(session=session)
    record["matches"][0]["action"] = "delete"
    return record


def _schema_invalid_record(session: str) -> dict:
    record = _match_record(session=session)
    del record["agent_class"]
    return record


def test_a_deletion_shaped_record_is_skipped_and_the_rest_of_the_batch_lands():
    good_1 = _match_record(session="sess-1")
    good_3 = _match_record(session="sess-3")
    batch = [good_1, _deletion_shaped_record("sess-2"), good_3]

    merge = mod.apply_coding_records(_codebook_with_entry_a(), batch, invariant_slugs=NO_SLUGS)

    [skipped] = merge.skipped_deletions
    assert (skipped.index, skipped.session) == (1, "sess-2")
    assert "removal action" in skipped.reason
    assert skipped.describe() == f"session=sess-2: {skipped.reason}"
    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-1", "sess-3"]
    assert len(merge.merged_records) == 2
    assert all(m is g for m, g in zip(merge.merged_records, (good_1, good_3), strict=True))
    assert merge.counts == mod.MergeCounts(matched=2)


def test_a_schema_invalid_record_is_named_and_not_applied():
    invalid = _schema_invalid_record("sess-bad")

    merge = merge_one(_codebook_with_entry_a(), invalid)

    [skipped] = merge.invalid_records
    assert (skipped.index, skipped.session) == (0, "sess-bad")
    assert "'agent_class' is a required property" in skipped.reason
    assert merge.merged_records == merge.skipped_deletions == ()
    assert merge.codebook == _codebook_with_entry_a()


def test_a_non_dict_record_is_an_invalid_record_without_a_session():
    merge = merge_one(_codebook_with_entry_a(), ["not", "a", "record"])

    [skipped] = merge.invalid_records
    assert (skipped.index, skipped.session) == (0, None)
    assert skipped.reason
    assert merge.merged_records == ()


def test_the_batch_never_mutates_or_aliases_its_input_codebook():
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)

    _ = mod.apply_coding_records(
        codebook,
        [_match_record(), _deletion_shaped_record("sess-2"), _correction_record()],
        invariant_slugs=NO_SLUGS,
    )
    empty = mod.apply_coding_records(codebook, [], invariant_slugs=NO_SLUGS)

    assert codebook == original
    assert empty.codebook == codebook
    assert empty.codebook is not codebook
    assert empty.codebook["entries"] is not codebook["entries"]


def test_counts_and_slug_tallies_are_summed_across_the_batch():
    declared = _match_record(session="sess-1")
    declared["matches"][0]["invariant_violated"] = "known-slug"
    undeclared = _match_record(session="sess-2")
    undeclared["matches"][0]["invariant_violated"] = "x"

    merge = mod.apply_coding_records(
        _codebook_with_entry_a(), [declared, undeclared], invariant_slugs=_KNOWN_SLUGS
    )

    assert merge.counts == mod.MergeCounts(matched=2)
    assert merge.invariant_slugs == mod.SlugTally(valid=1, rejected=("x",))


def _adjudicated_title_case() -> tuple[dict, dict]:
    return (
        _codebook_with_adjudicated_candidate("rejected"),
        _candidate_record(title="novel shape", session="sess-new", date="2026-07-24"),
    )


@pytest.mark.parametrize(
    ("build_case", "count"),
    [
        pytest.param(lambda: (_codebook_with_entry_a(), _match_record()), "matched", id="match"),
        pytest.param(
            lambda: (_codebook_with_entry_a(), _candidate_record()),
            "candidates_applied",
            id="new-candidate",
        ),
        pytest.param(
            _adjudicated_title_case, "candidate_disposition_conflicts", id="adjudicated-title"
        ),
        pytest.param(
            lambda: (_codebook_with_entry_a(), _correction_record()),
            "corrections_applied",
            id="correction",
        ),
    ],
)
def test_every_mutating_op_marks_the_merge_mutated_and_its_re_apply_does_not(
    build_case, count
):
    """Task 6357's rule, now held once by the merger: each of the four ops that
    writes the codebook is a mutation, so a night whose only effect is one of
    them still dumps and commits."""
    codebook, record = build_case()

    first = merge_one(codebook, record)
    again = merge_one(first.codebook, record)

    assert first.counts == mod.MergeCounts(**{count: 1})
    assert first.counts.mutations == 1
    assert first.mutated is True
    assert again.counts == mod.MergeCounts()
    assert again.mutated is False


def test_an_unknown_entry_match_does_not_mark_the_merge_mutated():
    merge = merge_one(_codebook_with_entry_a(), _match_record(entry_id="entry-zzz"))

    assert merge.counts == mod.MergeCounts(skipped_unknown_entry=1)
    assert merge.mutated is False


def test_the_summary_is_one_line_naming_every_report_field():
    undeclared = _match_record(session="sess-1")
    undeclared["matches"][0]["invariant_violated"] = "x"
    batch = [undeclared, _deletion_shaped_record("sess-2"), _schema_invalid_record("sess-3")]

    summary = mod.apply_coding_records(
        _codebook_with_entry_a(), batch, invariant_slugs=_KNOWN_SLUGS
    ).summary()

    assert "\n" not in summary
    assert {
        "matched=1",
        "deletion_directive=1",
        "record_invalid=1",
        "candidate_disposition_conflicts=0",
        "invariant_slugs_rejected=1",
    } <= set(summary.split())


# ---------------------------------------------------------------------------
# task 6750: the merge names the candidate each candidate payload resolved to
# ---------------------------------------------------------------------------

def _ref(record: int, candidate: int = 0):
    return mod.PayloadRef(record=record, candidate=candidate)


def _two_payload_record(session: str = "sess-1") -> dict:
    record = _candidate_record(title="novel shape", session=session)
    record["candidates"].append({**record["candidates"][0], "title": "a different shape"})
    return record


def _codebook_with_pending_candidate() -> dict:
    codebook = _codebook_with_adjudicated_candidate("pending")
    codebook["candidates"][0]["id"] = "cand-20260701-3"
    return codebook


def _codebook_with_promoted_then_rejected_twin(*, promoted_to: str) -> dict:
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = promoted_to
    codebook["candidates"].append({
        "id": "cand-20260724-2",
        "title": "novel shape",
        "first_seen": "2026-07-24",
        "disposition": "rejected",
        "sightings": [],
    })
    return codebook


def test_an_unseen_title_resolves_to_the_candidate_the_merge_mints():
    merge = merge_one(_codebook_with_entry_a(), _candidate_record())

    assert merge.resolutions == (mod.NewPending(payload=_ref(0), candidate_id="cand-20260714-1"),)
    assert merge.resolution_at(_ref(0)) == merge.resolutions[0]


def test_resolutions_follow_batch_order_one_per_candidate_payload():
    batch = [_two_payload_record("sess-1"), _match_record(session="sess-2"), _candidate_record(session="sess-3")]

    merge = mod.apply_coding_records(_codebook_with_entry_a(), batch, invariant_slugs=NO_SLUGS)

    assert merge.resolutions == (
        mod.NewPending(payload=_ref(0, 0), candidate_id="cand-20260714-1"),
        mod.NewPending(payload=_ref(0, 1), candidate_id="cand-20260714-2"),
        mod.PendingAbsorbed(payload=_ref(2, 0), candidate_id="cand-20260714-1"),
    )
    assert merge.resolution_at(_ref(0, 1)) == merge.resolutions[1]
    assert merge.resolution_at(_ref(0, 2)) is None
    assert merge.resolution_at(_ref(1)) is None


def test_a_title_a_pending_candidate_carries_resolves_to_that_candidate():
    merge = merge_one(_codebook_with_pending_candidate(), _candidate_record(session="sess-new"))

    assert merge.resolutions == (mod.PendingAbsorbed(payload=_ref(0), candidate_id="cand-20260701-3"),)


def test_a_promoted_candidate_with_a_live_entry_outranks_a_later_rejected_twin():
    """The task-4144 shape, in the order the live codebook carries it: the
    recurrence goes to the promoted candidate's entry, not to the list-order
    last same-title record."""
    codebook = _codebook_with_promoted_then_rejected_twin(promoted_to="entry-a")

    merge = merge_one(codebook, _candidate_record(session="sess-new", date="2026-07-25"))

    assert merge.resolutions == (
        mod.RoutedToEntry(payload=_ref(0), candidate_id="cand-20260722-28", entry_id="entry-a"),
    )
    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert [s["session"] for s in entry["sightings"]] == ["sess-new"]


def test_a_rejected_only_title_resolves_to_the_rejected_candidate():
    codebook = _codebook_with_adjudicated_candidate("rejected")

    merge = merge_one(codebook, _candidate_record(session="sess-new", date="2026-07-24"))

    assert merge.resolutions == (
        mod.RoutedToCandidate(payload=_ref(0), candidate_id="cand-20260722-28", disposition="rejected"),
    )


@pytest.mark.parametrize(
    ("build_codebook", "expected"),
    [
        pytest.param(
            lambda: _codebook_with_adjudicated_candidate("promoted"),
            ("cand-20260722-28", "promoted"),
            id="lone-promoted",
        ),
        pytest.param(
            lambda: _codebook_with_promoted_then_rejected_twin(promoted_to="entry-never-created"),
            ("cand-20260724-2", "rejected"),
            id="promoted-then-rejected",
        ),
    ],
)
def test_a_promoted_candidate_whose_entry_dangles_falls_back_to_the_last_same_title_record(
    build_codebook, expected
):
    merge = merge_one(build_codebook(), _candidate_record(session="sess-new", date="2026-07-25"))

    candidate_id, disposition = expected
    assert merge.resolutions == (
        mod.RoutedToCandidate(payload=_ref(0), candidate_id=candidate_id, disposition=disposition),
    )


@pytest.mark.parametrize(
    "build_codebook",
    [
        pytest.param(_codebook_with_pending_candidate, id="pending-absorbed"),
        pytest.param(
            lambda: _codebook_with_promoted_then_rejected_twin(promoted_to="entry-a"),
            id="routed-to-entry",
        ),
        pytest.param(lambda: _codebook_with_adjudicated_candidate("rejected"), id="routed-to-candidate"),
    ],
)
def test_re_applying_a_merged_record_resolves_it_again_and_writes_nothing(build_codebook):
    record = _candidate_record(session="sess-new", date="2026-07-25")
    first = merge_one(build_codebook(), record)

    again = merge_one(first.codebook, record)

    assert again.resolutions == first.resolutions
    assert again.codebook == first.codebook
    assert again.mutated is False


def test_re_applying_a_minting_record_names_the_candidate_it_minted():
    record = _candidate_record()
    first = merge_one(_codebook_with_entry_a(), record)

    again = merge_one(first.codebook, record)

    assert again.resolutions == (mod.PendingAbsorbed(payload=_ref(0), candidate_id="cand-20260714-1"),)
    assert again.codebook == first.codebook


def test_a_skipped_record_resolves_no_payload():
    deletion = _candidate_record(session="sess-del")
    deletion["drop"] = True
    invalid = _candidate_record(session="sess-bad")
    del invalid["agent_class"]
    batch = [deletion, _candidate_record(session="sess-good"), invalid]

    merge = mod.apply_coding_records(_codebook_with_entry_a(), batch, invariant_slugs=NO_SLUGS)

    assert merge.resolutions == (mod.NewPending(payload=_ref(1), candidate_id="cand-20260714-1"),)
    assert merge.resolution_at(_ref(0)) is None
    assert merge.resolution_at(_ref(2)) is None


def test_a_record_without_candidate_payloads_resolves_nothing():
    merge = mod.apply_coding_records(
        _codebook_with_entry_a(), [_match_record(), _correction_record()], invariant_slugs=NO_SLUGS
    )

    assert merge.resolutions == ()


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

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])

        assert ret == 0
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert len(entry["sightings"]) == 1
        assert len(reloaded["candidates"]) == 1
        assert mod.validate(reloaded) == []

    def test_apply_is_idempotent_at_file_level(self, tmp_path):
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)

        first = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
        second = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])

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
        just itself — mirrors apply_coding_records()'s skip-whole-record
        semantics, but at the line-parsing boundary)."""
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)
        good_line = records_path.read_text(encoding="utf-8")
        records_path.write_text("{not valid json\n" + good_line, encoding="utf-8")

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
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
        bad["matches"][0]["action"] = "delete"  # trips _deletion_directive

        # A second good record under a DIFFERENT session, so it is not
        # deduped away by the session-keyed idempotency guard — its arrival
        # proves the records AFTER the bad line were not discarded.
        good_2 = copy.deepcopy(good)
        good_2["session"] = "sess-2"

        records_path.write_text(
            "\n".join(json.dumps(r) for r in (good, bad, good_2)) + "\n",
            encoding="utf-8",
        )
        expected = mod.apply_coding_records(
            mod.load(codebook_path), [good, bad, good_2], invariant_slugs=()
        )

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        [applied] = [line for line in captured.out.splitlines() if line.startswith("applied: ")]
        assert expected.summary() in applied
        assert "malformed_json=0" in applied
        [skipped] = expected.skipped_deletions
        assert [
            line
            for line in captured.err.splitlines()
            if line.startswith(f"{records_path}:2:") and "deletion directive" in line
            and skipped.describe() in line
        ]

        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert {s["session"] for s in entry["sightings"]} == {"sess-1", "sess-2"}
        assert mod.validate(reloaded) == []

    def test_apply_names_a_schema_invalid_record_without_aborting_batch(
        self, tmp_path, capsys
    ):
        """A record the validator rejects is skipped whole, and the operator is
        told which line and why rather than only seeing a counter move."""
        codebook_path, records_path = self._write_apply_fixtures(tmp_path)
        good = json.loads(records_path.read_text(encoding="utf-8"))
        invalid = _schema_invalid_record("sess-bad")
        records_path.write_text(
            "\n".join(json.dumps(r) for r in (good, invalid)) + "\n", encoding="utf-8"
        )

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
        captured = capsys.readouterr()

        assert ret == 0
        assert "record_invalid=1" in captured.out
        assert [
            line
            for line in captured.err.splitlines()
            if line.startswith(f"{records_path}:2:") and "invalid" in line
            and "session=sess-bad" in line
            and "'agent_class' is a required property" in line
        ]
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert [s["session"] for s in entry["sightings"]] == ["sess-1"]

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

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
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

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(codebook_path), str(records_path)])
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
        must not crash apply_coding_records() with an unhandled exception."""
        path = tmp_path / "codebook.yaml"
        path.write_text("", encoding="utf-8")
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(json.dumps(_match_record()) + "\n", encoding="utf-8")

        ret = mod.main(["apply", "--project-root", str(tmp_path), str(path), str(records_path)])
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

    def test_apply_screens_slugs_against_the_project_doc(self, tmp_path, capsys, caplog):
        caplog.set_level(logging.DEBUG)
        root = tmp_path / "project"
        doc = root / invariants.DOC_RELPATH
        doc.parent.mkdir(parents=True)
        doc.write_text("## INV-1 `known-slug`\n", encoding="utf-8")
        codebook_path = tmp_path / "codebook.yaml"
        mod.dump(_codebook_with_entry_a(), codebook_path)
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(json.dumps(_slugged_record()) + "\n", encoding="utf-8")

        ret = mod.main(
            ["apply", "--project-root", str(root), str(codebook_path), str(records_path)]
        )

        assert ret == 0
        reloaded = mod.load(codebook_path)
        entry = next(e for e in reloaded["entries"] if e["id"] == "entry-a")
        assert _sighting_on(entry, "sess-1")["invariant_violated"] == "known-slug"
        candidate = next(c for c in reloaded["candidates"] if c["title"] == "novel shape")
        assert "invariant_violated" not in _sighting_on(candidate, "sess-1")
        out = capsys.readouterr().out
        assert "invariant_slugs_valid=1" in out
        assert "invariant_slugs_rejected=1" in out
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


# ---------------------------------------------------------------------------
# task 6402 (census PRD leaf L3, C1): finding identity, run history and ticket
# fields on entries and candidates. Every field is optional; each rejection
# names the field path and carries the offending value (INV-2).
# ---------------------------------------------------------------------------

_C1_RUN = "census-dark_factory-20261010"
_C1_KEY = finding_key("orchestrator", "orchestrator/x.py::f", "h13")
_C1_OLD_KEY = finding_key("orchestrator", "orchestrator/x.py::old", "h13")
_C1_ENTRY_FIELDS = {
    "finding_key": _C1_KEY,
    "finding_area": "orchestrator",
    "finding_anchor": "orchestrator/x.py::f",
    "finding_tags": ["h13", "kind:confusion"],
    "severity_reason": "r",
    "first_seen_run": _C1_RUN,
    "last_seen_runs": [_C1_RUN],
    "supersedes": [_C1_OLD_KEY],
    "fixed_at_sha": "abc1234",
    "accepted_by": 42,
}


def _c1_ticket(**overrides: Any) -> dict:
    row = {
        "ticket": "tkt_1",
        "project_root": "/r",
        "run_id": "census-x-20261010",
        "finding_key": _C1_KEY,
        "task_id": 12,
        "resolution": "created",
        "step": "curator",
    }
    row.update(overrides)
    return row


def _c1_entry(**overrides: Any) -> dict:
    entry = _minimal_v2()["entries"][0]
    entry.update(copy.deepcopy(_C1_ENTRY_FIELDS))
    entry["status"] = "accepted"
    entry["filed_tickets"] = [
        _c1_ticket(),
        _c1_ticket(ticket=None, task_id=None, resolution=None),
    ]
    entry.update(overrides)
    return entry


def _c1_candidate(**overrides: Any) -> dict:
    candidate = {
        "id": "cand-20261010-1",
        "title": "a novel shape",
        "first_seen": "2026-10-10",
        "disposition": "rejected",
        "sightings": [],
        "finding_key": _C1_KEY,
        "finding_area": "orchestrator",
        "finding_anchor": "orchestrator/x.py::f",
        "finding_tags": ["h13", "kind:confusion"],
        "attached_by": {"method": "similarity", "score": 0.93, "run_id": "census-x-20261010"},
        "adjudicated_with": "cand-20261010-3",
    }
    candidate.update(overrides)
    return candidate


def _c1_codebook(*, entry: dict | None = None, candidate: dict | None = None) -> dict:
    codebook = _minimal_v2()
    codebook["entries"] = [entry if entry is not None else _c1_entry()]
    codebook["candidates"] = [candidate] if candidate is not None else []
    return codebook


def _c1_errors_naming(codebook: dict, path: str, offending: Any) -> list[str]:
    return [error for error in mod.validate(codebook) if path in error and repr(offending) in error]


def test_accepted_is_a_codebook_status():
    assert "accepted" in mod.STATUSES


def test_an_entry_carrying_every_c1_field_validates():
    assert mod.validate(_c1_codebook()) == []


def test_the_c1_fields_are_all_optional_on_an_entry():
    entry = _c1_entry(status="open")
    for field in [*_C1_ENTRY_FIELDS, "filed_tickets"]:
        del entry[field]
    assert mod.validate(_c1_codebook(entry=entry)) == []


@pytest.mark.parametrize(
    "field, value, offending",
    [
        ("finding_key", "fk-ZZZ", "fk-ZZZ"),
        ("finding_key", "fk-0123456789ab0", "fk-0123456789ab0"),
        ("finding_area", "orchestrator/merge-queue", "orchestrator/merge-queue"),
        ("finding_tags", [], []),
        ("finding_tags", "h13", "h13"),
        ("last_seen_runs", _C1_RUN, _C1_RUN),
        ("supersedes", ["not-a-key"], "not-a-key"),
        ("fixed_at_sha", "XYZ", "XYZ"),
        ("accepted_by", "42", "42"),
    ],
)
def test_validate_rejects_a_malformed_c1_entry_field(field, value, offending):
    codebook = _c1_codebook(entry=_c1_entry(**{field: value}))
    assert _c1_errors_naming(codebook, f"entries/0/{field}", offending)


@pytest.mark.parametrize(
    "overrides, offending",
    [
        ({"resolution": "maybe"}, "maybe"),
        ({"step": "guess"}, "guess"),
        ({"task_id": True}, True),
        ({"task_id": "12"}, "12"),
    ],
)
def test_validate_rejects_a_malformed_filed_ticket_row(overrides, offending):
    codebook = _c1_codebook(entry=_c1_entry(filed_tickets=[_c1_ticket(**overrides)]))
    assert _c1_errors_naming(codebook, "entries/0/filed_tickets/0", offending)


def test_validate_rejects_a_filed_ticket_row_missing_its_step():
    row = _c1_ticket()
    del row["step"]
    codebook = _c1_codebook(entry=_c1_entry(filed_tickets=[row]))
    assert _c1_errors_naming(codebook, "entries/0/filed_tickets/0", "step")


def test_a_back_link_filed_before_finding_keys_existed_validates():
    codebook = _c1_codebook(entry=_c1_entry(filed_tickets=[_c1_ticket(finding_key=None)]))
    assert mod.validate(codebook) == []


def test_a_malformed_finding_key_on_a_filed_ticket_row_is_still_rejected():
    codebook = _c1_codebook(entry=_c1_entry(filed_tickets=[_c1_ticket(finding_key="fk-ZZZ")]))
    assert _c1_errors_naming(codebook, "entries/0/filed_tickets/0", "fk-ZZZ")


def test_a_historical_backfill_marker_validates():
    codebook = _c1_codebook()
    codebook[mod.HISTORICAL_BACKFILL] = {"run_id": _C1_RUN, "unmatched_task_ids": [5330]}
    assert mod.validate(codebook) == []


@pytest.mark.parametrize(
    "marker, path, offending",
    [
        ({"run_id": "", "unmatched_task_ids": []}, "run_id", ""),
        ({"run_id": _C1_RUN, "unmatched_task_ids": ["5330"]}, "unmatched_task_ids/0", "5330"),
        ({"unmatched_task_ids": []}, "", "run_id"),
    ],
    ids=["empty-run", "string-task-id", "no-run"],
)
def test_validate_rejects_a_malformed_historical_backfill_marker(marker, path, offending):
    codebook = _c1_codebook()
    codebook[mod.HISTORICAL_BACKFILL] = marker
    assert _c1_errors_naming(codebook, f"{mod.HISTORICAL_BACKFILL}/{path}".rstrip("/"), offending)


def test_filed_ticket_errors_accepts_a_well_formed_row_keyed_or_not():
    assert mod.filed_ticket_errors(_c1_ticket()) == []
    assert mod.filed_ticket_errors(_c1_ticket(finding_key=None)) == []


def test_filed_ticket_errors_names_the_offending_field_or_value():
    unresolved = _c1_ticket()
    del unresolved["resolution"]

    assert any("guess" in error for error in mod.filed_ticket_errors(_c1_ticket(step="guess")))
    assert any("resolution" in error for error in mod.filed_ticket_errors(unresolved))


def test_validate_rejects_an_accepted_entry_without_accepted_by():
    entry = _c1_entry()
    del entry["accepted_by"]
    errors = mod.validate(_c1_codebook(entry=entry))
    assert [error for error in errors if "entry-a" in error and "accepted_by" in error]


def test_legacy_filed_tasks_stay_unconstrained():
    for filed_tasks in ("2547, 2548", [2559]):
        entry = _c1_entry(filed_tasks=filed_tasks)
        assert mod.validate(_c1_codebook(entry=entry)) == []


def test_a_candidate_carrying_every_c1_field_validates():
    assert mod.validate(_c1_codebook(candidate=_c1_candidate())) == []


def test_validate_rejects_a_candidate_attached_by_with_an_unknown_method():
    candidate = _c1_candidate(attached_by={"method": "vibes", "run_id": _C1_RUN})
    codebook = _c1_codebook(candidate=candidate)
    assert _c1_errors_naming(codebook, "candidates/0/attached_by/method", "vibes")


def test_validate_rejects_a_candidate_attached_by_with_a_string_score():
    candidate = _c1_candidate(attached_by={"method": "similarity", "score": "0.9", "run_id": _C1_RUN})
    codebook = _c1_codebook(candidate=candidate)
    assert _c1_errors_naming(codebook, "candidates/0/attached_by/score", "0.9")


def test_validate_rejects_a_candidate_attached_by_without_a_run_id():
    candidate = _c1_candidate(attached_by={"method": "key"})
    codebook = _c1_codebook(candidate=candidate)
    assert _c1_errors_naming(codebook, "candidates/0/attached_by", "run_id")


def test_validate_rejects_a_candidate_adjudicated_with_a_non_candidate_id():
    codebook = _c1_codebook(candidate=_c1_candidate(adjudicated_with="cand-x"))
    assert _c1_errors_naming(codebook, "candidates/0/adjudicated_with", "cand-x")


def test_a_candidates_finding_identity_is_validated_like_an_entrys():
    codebook = _c1_codebook(candidate=_c1_candidate(finding_area="orchestrator/merge-queue"))
    assert _c1_errors_naming(codebook, "candidates/0/finding_area", "orchestrator/merge-queue")


_C1_KEY_ONLY_DROPS = ("finding_area", "finding_anchor", "finding_tags")


def _key_only(record: dict) -> dict:
    return {field: value for field, value in record.items() if field not in _C1_KEY_ONLY_DROPS}


@pytest.mark.parametrize(
    "codebook, path",
    [
        (_c1_codebook(entry=_key_only(_c1_entry())), "entries/0"),
        (_c1_codebook(candidate=_key_only(_c1_candidate())), "candidates/0"),
    ],
)
def test_validate_rejects_a_partial_finding_identity(codebook, path):
    errors = mod.validate(codebook)
    for missing in _C1_KEY_ONLY_DROPS:
        assert [error for error in errors if f"{path}/{missing}" in error and "finding_key" in error]


@pytest.mark.parametrize(
    "field, value",
    [
        ("finding_anchor", "orchestrator/x.py:12"),
        ("finding_anchor", "/abs/x.py"),
        ("finding_tags", ["h99"]),
        ("finding_tags", ["h13", ""]),
    ],
)
def test_validate_holds_the_identity_to_the_contract_anchor_and_tag_vocabulary(field, value):
    codebook = _c1_codebook(entry=_c1_entry(**{field: value}))
    assert _c1_errors_naming(codebook, f"entries/0/{field}", value)


# ---------------------------------------------------------------------------
# task 6405 (census PRD leaf L6): every attach sets attached_by. The merger
# copies a payload's attached_by onto the sighting it lands, wherever its
# existing routing lands it; validate() and validate_coding_record() hold the
# one attached_by shape, and validate() the sighting's verifier_tags.
# ---------------------------------------------------------------------------

_L6_RUN = "census-dark_factory-20260714"
_L6_TITLE_ATTACH = {"method": "title", "run_id": _L6_RUN}


def _attached(record: dict, op: str) -> dict:
    record[op][0]["attached_by"] = dict(_L6_TITLE_ATTACH)
    return record


def _recurrence_of_novel_shape() -> dict:
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")
    return _attached(record, "candidates")


def test_a_title_attach_on_a_rejected_candidate_lands_its_attached_by():
    merge = merge_one(_codebook_with_adjudicated_candidate("rejected"), _recurrence_of_novel_shape())

    candidate = merge.codebook["candidates"][0]
    assert _sighting_on(candidate, "sess-new")["attached_by"] == _L6_TITLE_ATTACH
    assert mod.validate(merge.codebook) == []


def test_a_title_attach_on_a_promoted_candidate_lands_its_attached_by_on_the_entry():
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0]["promoted_to"] = "entry-a"

    merge = merge_one(codebook, _recurrence_of_novel_shape())

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert _sighting_on(entry, "sess-new")["attached_by"] == _L6_TITLE_ATTACH
    assert mod.validate(merge.codebook) == []


def test_a_match_carrying_attached_by_lands_it_on_the_entry():
    merge = merge_one(_codebook_with_entry_a(), _attached(_match_record(), "matches"))

    entry = merge.codebook["entries"][0]
    assert _sighting_on(entry, "sess-1")["attached_by"] == _L6_TITLE_ATTACH
    assert mod.validate(merge.codebook) == []


def test_a_payload_without_attached_by_lands_a_sighting_without_the_key():
    merge = merge_one(
        _codebook_with_adjudicated_candidate("rejected"),
        _candidate_record(title="novel shape", session="sess-new", date="2026-07-24"),
    )

    assert "attached_by" not in _sighting_on(merge.codebook["candidates"][0], "sess-new")


def _codebook_with_a_sighting(**fields: Any) -> dict:
    codebook = _codebook_with_entry_a()
    sighting = {
        "date": "2026-07-14",
        "project": "dark_factory",
        "session": "sess-1",
        "origin_phase": "implement",
        "manifested_phase": "merge",
    }
    sighting.update(fields)
    codebook["entries"][0]["sightings"] = [sighting]
    return codebook


def test_validate_accepts_a_sighting_carrying_attached_by_and_verifier_tags():
    codebook = _codebook_with_a_sighting(
        attached_by={"method": "key", "run_id": _L6_RUN},
        verifier_tags=["h11", "kind:confusion"],
    )
    assert mod.validate(codebook) == []


@pytest.mark.parametrize(
    "fields, path, offending",
    [
        ({"attached_by": {"method": "vibes", "run_id": _L6_RUN}}, "attached_by/method", "vibes"),
        ({"attached_by": {"method": "title"}}, "attached_by", "run_id"),
        ({"verifier_tags": []}, "verifier_tags", []),
        ({"verifier_tags": "h11"}, "verifier_tags", "h11"),
        ({"verifier_tags": ["h11", 7]}, "verifier_tags/1", 7),
    ],
    ids=["unknown-method", "no-run-id", "empty-tags", "string-tags", "non-string-tag"],
)
def test_validate_rejects_a_malformed_sighting_attach(fields, path, offending):
    codebook = _codebook_with_a_sighting(**fields)
    assert _c1_errors_naming(codebook, f"entries/0/sightings/0/{path}", offending)


@pytest.mark.parametrize("op", ["matches", "candidates"])
@pytest.mark.parametrize(
    "attached_by, offending",
    [({"method": "vibes", "run_id": _L6_RUN}, "vibes"), ({"method": "title"}, "run_id")],
    ids=["unknown-method", "no-run-id"],
)
def test_validate_coding_record_rejects_a_malformed_payload_attach(op, attached_by, offending):
    record = _match_record() if op == "matches" else _candidate_record()
    record[op][0]["attached_by"] = attached_by

    errors = mod.validate_coding_record(record)

    assert [error for error in errors if f"{op}/0/attached_by" in error and repr(offending) in error]


@pytest.mark.parametrize("op", ["matches", "candidates"])
def test_validate_coding_record_accepts_a_well_formed_payload_attach(op):
    record = _match_record() if op == "matches" else _candidate_record()
    assert mod.validate_coding_record(_attached(record, op)) == []


# ---------------------------------------------------------------------------
# task 6750: validate cross-references a promoted candidate's entry and holds
# an entry's filing marker to its vocabulary
# ---------------------------------------------------------------------------

def _codebook_with_promoted_candidate(**promotion: Any) -> dict:
    codebook = _codebook_with_adjudicated_candidate("promoted")
    codebook["candidates"][0].update(promotion)
    return codebook


def _errors_naming(codebook: dict, *fragments: str) -> list[str]:
    return [error for error in mod.validate(codebook) if all(fragment in error for fragment in fragments)]


@pytest.mark.parametrize(
    ("promotion", "offending"),
    [
        pytest.param({}, None, id="missing"),
        pytest.param({"promoted_to": "entry-never-created"}, "entry-never-created", id="dangling"),
    ],
)
def test_validate_rejects_a_promoted_candidate_whose_entry_does_not_resolve(promotion, offending):
    codebook = _codebook_with_promoted_candidate(**promotion)

    assert _errors_naming(codebook, "cand-20260722-28", repr(offending))


def test_validate_accepts_a_promoted_candidate_whose_entry_resolves():
    assert mod.validate(_codebook_with_promoted_candidate(promoted_to="entry-a")) == []


@pytest.mark.parametrize("filing", ["withheld", "owed", "filed"])
def test_validate_accepts_an_entry_filing_marker_of_the_vocabulary(filing):
    codebook = _minimal_v2()
    codebook["entries"][0]["filing"] = filing

    assert mod.validate(codebook) == []


@pytest.mark.parametrize("filing", ["held", True])
def test_validate_rejects_an_entry_filing_marker_outside_the_vocabulary(filing):
    codebook = _minimal_v2()
    codebook["entries"][0][mod.ENTRY_FILING_KEY] = filing

    assert _errors_naming(codebook, "entries/0/filing", repr(filing))


def test_validate_accepts_an_entry_without_a_filing_marker():
    codebook = _minimal_v2()

    assert mod.ENTRY_FILING_KEY not in codebook["entries"][0]
    assert mod.validate(codebook) == []


def test_validate_cli_fails_on_a_dangling_promoted_to(tmp_path, capsys):
    path = tmp_path / "codebook.yaml"
    mod.dump(_codebook_with_promoted_candidate(promoted_to="entry-never-created"), path)

    assert mod.main(["validate", str(path)]) != 0
    assert "entry-never-created" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# task 6787: a coder match quotes its evidence, and an entry records the
# census's fit verdict on each attached sighting as a `sighting_fits` row.
# ---------------------------------------------------------------------------

_FIT_RUN = "census-dark_factory-20261116"


def test_validate_coding_record_accepts_a_match_quoting_its_evidence():
    record = _well_formed_record()
    record["matches"][0]["evidence_quote"] = "ls scripts/legibility 2>/dev/null -> [exit 2]"

    assert mod.validate_coding_record(record) == []


def test_validate_coding_record_rejects_a_match_whose_evidence_quote_is_not_a_string():
    record = _well_formed_record()
    record["matches"][0]["evidence_quote"] = 7

    assert [error for error in mod.validate_coding_record(record) if "matches/0/evidence_quote" in error]


def _fit_row(**overrides) -> dict:
    row = {"session": "sess-1", "exhibits": True, "run_id": _FIT_RUN, "reason": "same probe"}
    row.update(overrides)
    return {key: value for key, value in row.items() if value is not None}


def _codebook_with_fit_rows(*rows: dict) -> dict:
    codebook = _codebook_with_a_sighting()
    codebook["entries"][0][mod.SIGHTING_FITS_KEY] = list(rows)
    return codebook


def test_validate_accepts_an_entry_whose_fit_row_names_its_own_sighting():
    assert mod.validate(_codebook_with_fit_rows(_fit_row())) == []


_MALFORMED_FIT_ROWS = [
    pytest.param((_fit_row(session=None),), id="no-session"),
    pytest.param((_fit_row(exhibits=None),), id="no-exhibits"),
    pytest.param((_fit_row(run_id=None),), id="no-run-id"),
    pytest.param((_fit_row(exhibits="yes"),), id="string-exhibits"),
    pytest.param((_fit_row(exhibits=1),), id="int-exhibits"),
    pytest.param((_fit_row(session="sess-never-sighted"),), id="unknown-session"),
    pytest.param((_fit_row(), _fit_row(exhibits=False)), id="two-rows-one-session"),
]


@pytest.mark.parametrize("rows", _MALFORMED_FIT_ROWS)
def test_validate_rejects_a_malformed_fit_row_naming_the_entry(rows):
    assert _errors_naming(_codebook_with_fit_rows(*rows), "entry-a", mod.SIGHTING_FITS_KEY)


@pytest.mark.parametrize("rows", _MALFORMED_FIT_ROWS)
def test_sighting_fit_errors_states_the_row_invariant_for_the_entry_alone(rows):
    codebook = _codebook_with_fit_rows(*rows)

    errors = mod.sighting_fit_errors(codebook["entries"][0])

    assert errors
    assert all("entry-a" in error for error in errors)
    assert set(errors) <= set(mod.validate(codebook))


def test_sighting_fit_errors_accepts_an_entry_without_fit_rows():
    assert mod.sighting_fit_errors(_codebook_with_a_sighting()["entries"][0]) == []
