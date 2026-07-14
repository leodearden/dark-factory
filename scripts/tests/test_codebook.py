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
from pathlib import Path

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

def test_migrate_v1_to_v2_raises_on_non_dict_codebook():
    with pytest.raises(ValueError):
        mod.migrate_v1_to_v2(None)


def test_apply_coding_record_raises_on_non_dict_codebook():
    with pytest.raises(ValueError):
        mod.apply_coding_record(None, _match_record())


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
