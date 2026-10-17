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
from legibility import coder, invariants
from shared.finding_key import finding_key

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The vocabulary of a project declaring no invariant slugs, for every merge
# test that is not about the slug screen.
_NO_SLUGS: tuple[str, ...] = ()

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

def _merge_one(codebook: dict, record: Any, slugs: tuple[str, ...] = _NO_SLUGS):
    return mod.apply_coding_records(codebook, [record], invariant_slugs=slugs)


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

    merge = _merge_one(codebook, record)

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

    once = _merge_one(codebook, record).codebook
    twice = _merge_one(once, record).codebook

    entry = next(e for e in twice["entries"] if e["id"] == "entry-a")
    assert len(entry["sightings"]) == 1
    assert mod.validate(twice) == []


def test_merge_unknown_entry_id_is_skipped_and_counted():
    codebook = _codebook_with_entry_a()
    record = _match_record(entry_id="entry-zzz")

    merge = _merge_one(codebook, record)

    assert len(merge.codebook["entries"]) == 1  # no entry fabricated
    assert merge.codebook["entries"][0]["sightings"] == []
    assert merge.counts.skipped_unknown_entry == 1
    assert mod.validate(merge.codebook) == []


def test_merge_does_not_mutate_input():
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)
    record = _match_record()

    _merge_one(codebook, record)

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

    merge = _merge_one(codebook, record)

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

    once = _merge_one(codebook, record).codebook
    twice = _merge_one(once, record).codebook

    assert len(twice["candidates"]) == 1
    assert len(twice["candidates"][0]["sightings"]) == 1
    assert mod.validate(twice) == []


def test_merge_different_candidate_same_day_increments_id():
    codebook = _codebook_with_entry_a()
    record_a = _candidate_record(title="novel shape")
    once = _merge_one(codebook, record_a).codebook

    record_b = _candidate_record(title="a different shape")
    twice = _merge_one(once, record_b).codebook

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


def test_merge_does_not_resurrect_rejected_candidate():
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    merge = _merge_one(codebook, record)

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

    merge = _merge_one(codebook, record)

    assert len(merge.codebook["candidates"]) == 1
    candidate = merge.codebook["candidates"][0]
    assert candidate["id"] == "cand-20260722-28"
    assert candidate["disposition"] == "promoted"  # census verdict untouched
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert merge.counts.candidate_disposition_conflicts == 1
    assert merge.counts.candidates_applied == 0
    assert mod.validate(merge.codebook) == []


def test_merge_routes_promoted_recurrence_to_its_entry():
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

    merge = _merge_one(codebook, record)

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

    merge = _merge_one(codebook, record)

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

    merge = _merge_one(codebook, record)

    assert len(merge.codebook["entries"]) == 1  # no entry fabricated
    assert merge.codebook["entries"][0]["sightings"] == []
    candidate = merge.codebook["candidates"][0]
    assert len(candidate["sightings"]) == 2
    assert candidate["sightings"][1]["session"] == "sess-new"
    assert merge.counts.matched == 0
    assert merge.counts.candidate_disposition_conflicts == 1
    assert mod.validate(merge.codebook) == []


def test_merge_adjudicated_conflict_is_idempotent():
    """The pre-existing `already_seen` guard already spans every
    disposition, so re-applying the same record appends nothing and reports
    no fresh conflict."""
    codebook = _codebook_with_adjudicated_candidate("rejected")
    record = _candidate_record(title="novel shape", session="sess-new", date="2026-07-24")

    once = _merge_one(codebook, record)
    twice = _merge_one(once.codebook, record)

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

    merge = _merge_one(codebook, record)

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

def _report(merge) -> tuple:
    """Everything a MergeResult reports besides its codebook, compared whole so
    the no-op pins below cannot miss a report field."""
    return (merge.counts, merge.invariant_slugs, merge.skipped_deletions, merge.invalid_records)


def _clean_report(**counts: int) -> tuple:
    """The `_report` of a merge that skipped nothing and counted only *counts*.
    With no *counts* it is what the sole writer reports when it changed nothing,
    shared with the live-corpus re-apply no-op pins below."""
    return (mod.MergeCounts(**counts), mod.SlugTally(), (), ())


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

    merge = _merge_one(codebook, record)

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
    assert _report(merge) == _clean_report(corrections_applied=1)
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

    result = _merge_one(codebook, _correction_record()).codebook

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
        once = _merge_one(codebook, record).codebook
        twice = _merge_one(once, record)

    assert _report(twice) == _clean_report()
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

    merge = _merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["title"] == correction["title"]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert _report(merge) == _clean_report(corrections_applied=1)
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
        merge = _merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry["title"] == record["corrections"][0]["title"]
    assert _report(merge) == _clean_report(corrections_applied=1, correction_skipped=1)

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

    merge = _merge_one(codebook, record)

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert entry[field] == before[field]
    for still_supplied in {"title", "cause", "status"} - {field}:
        assert entry[still_supplied] == correction[still_supplied]
    assert [s["note"] for s in entry["sightings"]] == [correction["note"]]
    assert _report(merge) == _clean_report(corrections_applied=1)
    assert mod.validate(merge.codebook) == []


def test_merge_correction_unknown_entry_id_is_skipped_and_counted():
    """Mirrors the match path: only the census creates entries, so an
    unresolvable entry_id is counted, never fabricated."""
    codebook = _codebook_with_entry_a()
    record = _correction_record(entry_id="entry-zzz")

    merge = _merge_one(codebook, record)

    assert merge.codebook["entries"] == _codebook_with_entry_a()["entries"]
    assert _report(merge) == _clean_report(skipped_unknown_entry=1)
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

    merge = _merge_one(codebook, record)

    assert merge.counts == mod.MergeCounts()
    assert [(s.index, s.session) for s in merge.invalid_records] == [(0, record["session"])]
    assert merge.skipped_deletions == merge.merged_records == ()
    assert merge.codebook == original


# ---------------------------------------------------------------------------
# step-13: RED (§8.3 never-delete) — NeverDeleteError + assert_no_deletion
# ---------------------------------------------------------------------------

def _skipped_as_deletion(record: dict):
    """Merge deletion-shaped *record* alone and hand back its SkippedRecord,
    after pinning that the batch neither raised, merged it, nor changed (or
    mutated) the codebook."""
    codebook = _codebook_with_entry_a()
    original = copy.deepcopy(codebook)

    merge = _merge_one(codebook, record)

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
        _merge_one(_codebook_with_entry_a(), _match_record())


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
    merge = _merge_one(_codebook_with_entry_a(), _slugged_record(), _KNOWN_SLUGS)

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

    merge = _merge_one(_codebook_with_entry_a(), record, _KNOWN_SLUGS)

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

    merge = _merge_one(_codebook_with_entry_a(), record, _KNOWN_SLUGS)

    assert merge.invariant_slugs == mod.SlugTally()


def test_a_project_without_slugs_rejects_every_value():
    merge = _merge_one(_codebook_with_entry_a(), _slugged_record(), ())

    entry = next(e for e in merge.codebook["entries"] if e["id"] == "entry-a")
    assert "invariant_violated" not in _sighting_on(entry, "sess-1")
    assert merge.invariant_slugs == mod.SlugTally(
        valid=0, rejected=("known-slug", "free text not a slug")
    )


def test_the_slug_screen_does_not_mutate_the_record():
    record = _slugged_record()
    original = copy.deepcopy(record)

    _merge_one(_codebook_with_entry_a(), record, ())

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
        _merge_one(_NOT_A_CODEBOOK, _match_record())


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

    merge = mod.apply_coding_records(_codebook_with_entry_a(), batch, invariant_slugs=_NO_SLUGS)

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

    merge = _merge_one(_codebook_with_entry_a(), invalid)

    [skipped] = merge.invalid_records
    assert (skipped.index, skipped.session) == (0, "sess-bad")
    assert "'agent_class' is a required property" in skipped.reason
    assert merge.merged_records == merge.skipped_deletions == ()
    assert merge.codebook == _codebook_with_entry_a()


def test_a_non_dict_record_is_an_invalid_record_without_a_session():
    merge = _merge_one(_codebook_with_entry_a(), ["not", "a", "record"])

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
        invariant_slugs=_NO_SLUGS,
    )
    empty = mod.apply_coding_records(codebook, [], invariant_slugs=_NO_SLUGS)

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

    first = _merge_one(codebook, record)
    again = _merge_one(first.codebook, record)

    assert first.counts == mod.MergeCounts(**{count: 1})
    assert first.counts.mutations == 1
    assert first.mutated is True
    assert again.counts == mod.MergeCounts()
    assert again.mutated is False


def test_an_unknown_entry_match_does_not_mark_the_merge_mutated():
    merge = _merge_one(_codebook_with_entry_a(), _match_record(entry_id="entry-zzz"))

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
# Shared machinery for the live-file guards below (tasks 4892, 5198, 5687,
# 5729).
# Each asserts that a committed §7.3 coding record and the merger-owned
# registry still agree; the comparison itself is single-sourced here, so each
# guard states only what is distinctive about its own correction and a fix to
# the comparison lands once rather than once per guard.
# ---------------------------------------------------------------------------

# What apply_coding_records copies verbatim from a match/candidate/correction
# payload into the sighting it appends. The optional three are emitted only
# when truthy, so a payload that omits one is compared as omitting it.
_SIGHTING_PAYLOAD_FIELDS = (
    "origin_phase",
    "manifested_phase",
    "invariant_violated",
    "note",
    "evidence_quote",
)

# What apply_coding_records copies from the record's header into every
# sighting it appends (the session is how a sighting is found, so it is the
# lookup key rather than a compared field).
_SIGHTING_HEADER_FIELDS = ("date", "project")


def _sightings_for(holder: dict, session: str) -> list[dict]:
    return [s for s in holder.get("sightings") or [] if s.get("session") == session]


def _session_record(record_path: Path, session: str) -> dict:
    """The one §7.3 coding record in JSONL *record_path* that carries
    *session*, asserted valid."""
    records = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mine = [r for r in records if r.get("session") == session]
    assert len(mine) == 1, (
        f"expected exactly one {session!r} record in {record_path}, found "
        f"{len(mine)}; sessions present: "
        f"{sorted({r.get('session') for r in records})}"
    )
    assert mod.validate_coding_record(mine[0]) == [], (
        f"invalid §7.3 coding record for session {session!r}"
    )
    return mine[0]


def _live_entry(codebook: dict, entry_id: str) -> dict:
    entries = [e for e in codebook.get("entries") or [] if e.get("id") == entry_id]
    assert len(entries) == 1, (
        f"expected exactly one {entry_id!r} entry, found {len(entries)}"
    )
    return entries[0]


def _assert_sighting_carries(record: dict, payload: dict, holder: dict) -> None:
    """Assert *holder* carries exactly one sighting for *record*'s session,
    and that it is what the sole writer appends for *payload*: the record's
    header fields and the payload's fields, verbatim. This is the
    in-place-rewrite detector for a committed sighting."""
    session = record["session"]
    carried = _sightings_for(holder, session)
    assert len(carried) == 1, (
        f"expected exactly one {session!r} sighting on {holder.get('id')!r}, "
        f"found {len(carried)}"
    )
    expected = {field: record[field] for field in _SIGHTING_HEADER_FIELDS} | {
        field: payload[field] for field in _SIGHTING_PAYLOAD_FIELDS if payload.get(field)
    }
    drift = {
        field: (want, carried[0].get(field))
        for field, want in expected.items()
        if carried[0].get(field) != want
    }
    assert drift == {}, (
        f"the {session!r} sighting on {holder.get('id')!r} no longer matches the "
        "committed record — a committed sighting was rewritten in place "
        f"{{field: (record, codebook)}}: {drift}"
    )


def _assert_correction_absorbed(
    record_path: Path,
    session: str,
    entry_id: str,
    *,
    refuted_tokens: tuple[str, ...],
    expected_status: str,
    original_sightings: dict[str, str],
) -> tuple[dict, dict, dict]:
    """Assert the *session* correction committed in *record_path* is fully
    absorbed by the committed registry, and hand back its
    ``(record, correction, entry)`` triple for whatever else a caller pins.

    A correction REWRITES merger-owned fields in place — the one op in the
    §7.3 vocabulary that is not a pure append — so every guard over one needs
    the same six properties, each invariant under every transition the system
    is designed to make to these two files:

    (0) WELL-FORMED — *record_path* is JSONL, exactly one of its records
        carries *session*, it validates, and it holds exactly one
        `corrections` op naming *entry_id*. Both counts are scoped BY
        SESSION, not file-wide: the sanctioned way to re-correct an entry
        later is a NEW session in the same file (task 4892's record holds
        several ops across two), and a file-wide count would break the moment
        anyone appends. Exactly-one matters because the merger admits one
        sighting per (session, entry), so a second op for this entry in this
        session would be counted in `correction_skipped` with its field
        writes lost.
    (1) AGREEMENT — the entry's title/cause/status equal the record's, and one
        sighting carries the record field for field. This is the
        in-place-rewrite detector for the merger-owned YAML, and it is what
        proves the correction's full text is carried verbatim — which is why
        no caller needs to pin cause or note prose.
    (2) WITHDRAWN — no *refuted_tokens* member survives in the title, the one
        field `scripts/legibility/coder.py::build_codebook_index` renders to
        the nightly coder unconditionally.
    (3) STATUS — the entry is *expected_status*, pinned so a later sweep can
        neither retire a live defect nor reopen a settled one while leaving
        the guard green.
    (4) NEVER-DELETE and ORDER — every session in *original_sightings*
        survives, still carrying its mapped date, and the correction's own
        sighting follows all of them.
    (5) NO-OP — re-applying the record changes nothing, which makes a rebase
        onto a nightly-rewritten main resolvable by re-running one CLI command
        instead of hand-editing 22k lines of generated YAML.

    An entry's TOTAL sighting count is deliberately never pinned: the nightly
    census may legitimately append more, to retired entries too (they stay in
    the coder's index). Presence-of-each-original plus ORDER carries
    never-delete without that coupling. Nor is any substring of a note's prose
    pinned — (1) already proves it is carried verbatim — and schema shape
    stays owned by `test_live_codebook_is_v2_and_validates_green`.
    """
    # (0) WELL-FORMED.
    record = _session_record(record_path, session)
    corrections = record.get("corrections") or []
    assert len(corrections) == 1, (
        f"expected exactly one correction in the {session!r} record, got "
        f"{len(corrections)} — the merger admits one sighting per (session, "
        "entry), so a second op for this entry in this session would be "
        "dropped into `correction_skipped` with its field writes lost"
    )
    correction = corrections[0]
    assert correction.get("entry_id") == entry_id

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    entry = _live_entry(codebook, entry_id)

    # (1) AGREEMENT — entry fields, then the provenance sighting.
    field_drift = {
        field: (correction[field], entry.get(field))
        for field in ("title", "cause", "status")
        if correction.get(field) and correction[field] != entry.get(field)
    }
    assert field_drift == {}, (
        f"{entry_id!r} no longer carries the committed correction — a "
        f"merger-owned field was rewritten by hand {{field: (record, codebook)}}: "
        f"{field_drift}"
    )
    _assert_sighting_carries(record, correction, entry)

    # (2) WITHDRAWN — the refuted framing is gone from the one field
    #     build_codebook_index renders unconditionally.
    title_lower = entry["title"].lower()
    still_asserted = [t for t in refuted_tokens if t.lower() in title_lower]
    assert still_asserted == [], (
        f"{entry_id!r} still asserts the refuted framing via {still_asserted} "
        f"— build_codebook_index renders this title to the coder every night: "
        f"{entry['title']!r}"
    )

    # (3) STATUS.
    assert entry.get("status") == expected_status, (
        f"{entry_id!r} is {entry.get('status')!r}, not {expected_status!r}"
    )

    # (4) NEVER-DELETE and ORDER.
    sightings = entry.get("sightings") or []
    order = {s.get("session"): i for i, s in enumerate(sightings)}
    for original_session, date in original_sightings.items():
        assert original_session in order, (
            f"the {original_session!r} sighting was deleted from {entry_id!r} "
            "— sightings are immutable dated observations, and a correction "
            "annotates them rather than retracting them"
        )
        original = sightings[order[original_session]]
        assert original.get("date") == date, (
            f"the {original_session!r} sighting was re-dated to "
            f"{original.get('date')!r} (was {date!r}) — a dated observation is "
            "not editable"
        )
    assert order[session] > max(order[s] for s in original_sightings), (
        f"the {session!r} sighting does not follow the sightings it annotates"
    )

    # (5) NO-OP — the committed YAML is exactly what the sole writer produces.
    merge = _merge_one(codebook, record)
    assert _report(merge) == _clean_report(), (
        f"re-applying session {session!r} changed something — the committed "
        f"codebook has not fully absorbed the record: {merge.summary()}"
    )

    return record, correction, entry


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
        merge = _merge_one(codebook, record)
        codebook = merge.codebook
        assert _report(merge) == _clean_report(), (
            f"re-applying session {record.get('session')!r} appended something — "
            f"its sightings are not already absorbed by the live codebook: "
            f"{merge.summary()}"
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
    in front of the very loop esc-5120-2 diagnoses as self-ingesting. The
    withdrawal therefore had to be a rewritten title AND a 'retired' status,
    which is exactly the pair `_assert_correction_absorbed` pins; nothing
    about this correction is distinctive enough to assert beyond it.
    """
    _assert_correction_absorbed(
        _T5198_RECORD_PATH,
        _T5198_CORRECTION_SESSION,
        _T5198_ENTRY_ID,
        refuted_tokens=_T5198_REFUTED_TOKENS,
        expected_status="retired",
        original_sightings=_T5198_ORIGINAL_SIGHTINGS,
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
# `unicode` is deliberately BROADER than the two em-dash spellings and the
# codepoint beside it. python3 reported the symptom as an `invalid character`,
# so the refuted claim respells itself just as naturally as "a unicode
# character in the pasted source breaks the script" — which the three narrow
# tokens would let through unchallenged. The cost is bounded and visible: the
# tuple is scanned against two short surfaces this correction owns (the title,
# and the index line the coder reads), so a future re-framing that genuinely
# needs the word can drop the token in the same commit that needs it.
#
# The bare U+2014 CHARACTER is deliberately NOT in this tuple; it is asserted
# against the TITLE alone, by (b) below. `build_codebook_index` renders each
# entry as `- {id}: {title} — {cause}`, and that separator IS U+2014, emitted
# exactly when the cause is non-empty. Banning the bare character from the
# rendered line would therefore contradict (c)'s demand for a rendered cause —
# no implementation could satisfy both. A title is a leaf value the separator
# cannot contaminate, so the character check belongs there. Do not "helpfully"
# add it back here.
_T5687_REFUTED_TOKENS = ("em-dash", "em dash", "u+2014", "unicode")

# How much of the entry's own collapsed cause must reach the rendered index
# line. Short enough to survive `build_codebook_index`'s truncation under any
# cap it may adopt, so this pins that the cause is carried, never how far.
_T5687_RENDERED_CAUSE_PREFIX_LEN = 80


def test_live_codebook_withdraws_the_refuted_em_dash_framing():
    """`_assert_correction_absorbed` pins the mechanics; this adds the three
    things distinctive to withdrawing a refuted SYMPTOM from a still-open
    entry.

    The entry named a symptom, not a mechanism. Measured against the archived
    transcript, replacing every U+2014 in the failing script with an ASCII
    hyphen leaves it failing — with a different error — while escaping or
    deleting the one embedded `\"\"\"` fixes it with every U+2014 left in
    place. So the reported error string is unstable across recurrences of the
    same defect, and a title keyed to it gives the nightly coder nothing to
    match on.

    (a) STILL OPEN, not retired — passed to the helper as the expected status.
        This correction replaces an explanation; it does not retire a live,
        unfixed defect, and pinning that stops a later sweep quietly retiring
        the entry. (Diverges deliberately from the task-5198 guard above.)
    (b) THE FOUNDING SIGHTING IS ANNOTATED, NOT RETRACTED. The helper pins
        that it survives and keeps its date; its `evidence_quote` is the part
        a later editor is most likely to mistake for the refuted claim and
        "clean up", so the exact quote is pinned here. The symptom it records
        was real — only the cause it was filed under was wrong.
    (c) THE CODER'S VIEW carries the mechanism and none of the symptom. The
        cause is non-empty and reaches the rendered index line, and no refuted
        token survives anywhere in that line. Asserted through the PUBLIC
        renderer over a one-entry codebook, whose output IS the single line
        the coder reads: one scan then covers the title and the truncated
        cause window together, a renderer change surfaces here instead of
        being mirrored in a copy of its format string, and a regression to the
        cause-less `- {id}: {title}` fallback — the very state this task
        exists to end — cannot pass. The cause BODY stays free to name the
        em-dash in order to REFUTE it; only what the coder actually reads is
        policed.
    """
    _, _, entry = _assert_correction_absorbed(
        _T5687_RECORD_PATH,
        _T5687_CORRECTION_SESSION,
        _T5687_ENTRY_ID,
        refuted_tokens=_T5687_REFUTED_TOKENS,
        expected_status="open",  # (a)
        original_sightings=_T5687_ORIGINAL_SIGHTING,
    )
    title = entry["title"]
    assert "—" not in title, (
        f"{_T5687_ENTRY_ID!r} still spells the refuted character in its "
        f"title: {title!r}"
    )

    # (b) THE FOUNDING SIGHTING IS ANNOTATED, NOT RETRACTED.
    for session in _T5687_ORIGINAL_SIGHTING:
        quotes = [s.get("evidence_quote") for s in _sightings_for(entry, session)]
        assert quotes == [_T5687_ORIGINAL_EVIDENCE_QUOTE], (
            f"the {session!r} sighting of {_T5687_ENTRY_ID!r} now carries "
            f"{quotes!r} — the symptom it recorded was real, and is not what "
            "the correction withdraws"
        )

    # (c) THE CODER'S VIEW.
    cause = entry.get("cause")
    assert cause, (
        f"{_T5687_ENTRY_ID!r} has no cause — with none, build_codebook_index "
        "shows the coder the title alone, which is the state this task exists "
        "to end"
    )
    opening = " ".join(str(cause).split())[:_T5687_RENDERED_CAUSE_PREFIX_LEN]
    rendered = coder.build_codebook_index({"entries": [entry]})
    missing = [
        part
        for part, needle in (
            ("id", _T5687_ENTRY_ID),
            ("title", title),
            ("cause", opening),
        )
        if needle not in rendered
    ]
    assert missing == [], (
        f"build_codebook_index left the {missing} of {_T5687_ENTRY_ID!r} out "
        f"of the index line the nightly coder reads: {rendered!r}"
    )
    still_rendered = [t for t in _T5687_REFUTED_TOKENS if t in rendered.lower()]
    assert still_rendered == [], (
        f"the index line the coder actually reads still asserts the refuted "
        f"framing via {still_rendered}: {rendered!r}"
    )


# ---------------------------------------------------------------------------
# task 5729 (live-file guard) — the ef7f6a5c census-verifier sighting misfiled
# on entry-cand-20260919-2 is annotated there and re-routed to
# entry-cand-20260722-18, through the sole writer.
# ---------------------------------------------------------------------------

_T5729_RECORD_PATH = (
    _REPO_ROOT / "docs" / "legibility" / "coding-records" / "task-5729-corrections.jsonl"
)
_T5729_ANNOTATION_SESSION = "task-5729-ef7f6a5c-reroute"
_T5729_MISFILED_SESSION = "ef7f6a5c-3ea0-43b3-b0ee-73fe34d67ccf"
_T5729_MISFILED_DATE = "2026-09-20"
_T5729_HOME_ENTRY_ID = "entry-cand-20260722-18"
_T5729_MISFILED_NOTE = (
    "The periodic-census verifier encountered the em-dash issue itself while "
    "verifying the cluster. Turns 42, 46, 54, 83, 97 show repeated "
    "ansi_c_string encoding rejections when constructing heredocs with "
    "embedded em-dashes. Error neighborhoods confirm SyntaxError with U+2014 "
    "character and multiple failed Bash commands attempting to work with the "
    "problematic em-dash pattern."
)


def test_live_codebook_reroutes_the_misfiled_census_verifier_sighting():
    """The misfiled sighting is annotated where it was filed and re-routed to
    the entry whose mechanism it shows, both through the sole writer.

    (1) ANNOTATION — a note-only correction on the misfiled entry, absorbed as
        `_assert_correction_absorbed` defines it, the entry still open.
    (2) REPLACES NO EXPLANATION — the annotation supplies no title, cause or
        status.
    (3) ANNOTATED, NOT RETRACTED — the misfiled sighting keeps its own note.
    (4) LINK — the annotation names the session it annotates and its new home.
    (5) ROUTE — one match op, under the real session and date, on the home
        entry, naming where it was misfiled; carried once, field for field.
    (6) ROUTE NO-OP — re-applying the route changes nothing.
    """
    _, correction, entry = _assert_correction_absorbed(
        _T5729_RECORD_PATH,
        _T5729_ANNOTATION_SESSION,
        _T5687_ENTRY_ID,
        refuted_tokens=_T5687_REFUTED_TOKENS,
        expected_status="open",
        original_sightings={_T5729_MISFILED_SESSION: _T5729_MISFILED_DATE},
    )

    # (2) REPLACES NO EXPLANATION.
    rewritten = {"title", "cause", "status"} & correction.keys()
    assert rewritten == set(), (
        f"the {_T5729_ANNOTATION_SESSION!r} correction writes {sorted(rewritten)} "
        f"of {_T5687_ENTRY_ID!r} — that entry's framing is owned by task 5687's "
        "record; this one only annotates a sighting"
    )

    # (3) ANNOTATED, NOT RETRACTED.
    notes = [s.get("note") for s in _sightings_for(entry, _T5729_MISFILED_SESSION)]
    assert notes == [_T5729_MISFILED_NOTE], (
        f"the {_T5729_MISFILED_SESSION!r} sighting of {_T5687_ENTRY_ID!r} now "
        f"carries {notes!r} — a dated observation is immutable: it is corrected "
        "by the sibling annotation sighting, never rewritten"
    )

    # (4) LINK.
    unnamed = [
        ident
        for ident in (_T5729_MISFILED_SESSION, _T5729_HOME_ENTRY_ID)
        if ident not in correction["note"]
    ]
    assert unnamed == [], (
        f"the annotation note does not name {unnamed} — naming them is the only "
        "link from an annotation to the sighting it annotates and its new home"
    )

    # (5) ROUTE.
    route = _session_record(_T5729_RECORD_PATH, _T5729_MISFILED_SESSION)
    assert route.get("date") == _T5729_MISFILED_DATE, (
        f"the route must keep the observation's own date, not {route.get('date')!r}"
    )
    assert not route.get("corrections") and not route.get("candidates"), (
        "the route only adds a sighting; it rewrites no entry and files nothing new"
    )
    matches = route.get("matches") or []
    assert len(matches) == 1, f"expected exactly one match op, got {len(matches)}"
    match = matches[0]
    assert match.get("entry_id") == _T5729_HOME_ENTRY_ID
    assert _T5687_ENTRY_ID in match.get("note", ""), (
        f"the route note does not name {_T5687_ENTRY_ID!r}, where the sighting "
        "was misfiled and still stands"
    )

    codebook = mod.load(_LIVE_CODEBOOK_PATH)
    _assert_sighting_carries(route, match, _live_entry(codebook, _T5729_HOME_ENTRY_ID))

    # (6) ROUTE NO-OP.
    merge = _merge_one(codebook, route)
    assert _report(merge) == _clean_report(), (
        f"re-applying the route changed something — the committed codebook has "
        f"not fully absorbed it: {merge.summary()}"
    )


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
